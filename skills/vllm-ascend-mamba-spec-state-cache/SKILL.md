---
name: vllm-ascend-mamba-spec-state-cache
description: 定位或修改 vLLM-Ascend Mamba/GDN 投机推理的状态缓存、拒绝后状态恢复、align 前缀缓存、跨 block 拷贝和按层 KV 加载逻辑。适用于跨块、请求重排、prefix hit 或 PD 恢复后才出错的场景。
---

# Mamba 投机推理状态缓存与回滚

目标是证明每个请求在下一次 forward 读到的状态对应正确的已提交 token 前缀。不要只凭最终文本或接受率判断状态正确。

## 定位当前实现

找到包含 `vllm/`、`vllm-ascend/` 的工作区；若仓库另放，分别确定上游根目录 `V` 和 Ascend 根目录 `A`。下面代码路径以各仓库为基准。先读待修改目录适用的 `AGENTS.md`。

核对双方 commit、运行时实际导入位置、V1/V2 runner、模型 recurrent 层类型、`mamba_cache_mode`、prefix caching、spec method/K、TP/PP/DCP/PCP、connector/PD role。GDN、Mamba2、short-conv 可以共用 `MambaSpec`，但 state types、shape、copy funcs 并不相同。

本技能基于 2026-10-04 本地快照：vLLM `d61081dc3d3f1740a5d8bf82608b62974393c2de`；Ascend `9b8fc5d728e1ea54dc277562b296112d905e15c1`。版本变动时按符号重定位，不能照搬注释中的旧版本判断。

先读 [references/state-contract.md](references/state-contract.md) 中与现象相关的部分，它列出实际源码、计数公式、拷贝边界和测试入口。

## 从一次出错迭代重建状态账本

每一行对应一个真实请求在一个 scheduler step 中的一次执行，至少记录：

| 类别 | 需要的字段 |
| --- | --- |
| 身份 | request ID、step、rank、batch row、persistent request slot、`idx_mapping` |
| 阶段 | 真正的 prefill/decode/spec 行、实际和 padded request/token 数、capture/replay |
| 计数 | step 前 computed、scheduled、draft 数；采样后实际 committed 数；state 恢复使用的 accepted 数 |
| 状态 | Mamba group ID、layer/state type、逻辑源/目标列、各 group 的物理 block ID |
| 布局 | shape、dtype、element size、stride、storage offset、page bytes、有效数据 bytes |
| 时序 | load 完成、pre-copy、forward 写入、sample/reject、post-copy、下一次 pre-copy |

批次位置不能充当持久请求槽位；多次重排后必须仍能追溯到同一个 request ID。设备数据只对缩小后的样例在明确的同步点导出，不要每层每步 `.item()` 或 `print(tensor)`。

## 按现象选择检查

### 拒绝 draft 后下一步才错

1. 查 sampler 输出的实际样本数，以及 runner 如何转换成 recurrent state 的 accepted count。中性 state count 通常为 `1`；chunked prefill 的 `num_sampled=0` 在 V2 被归一为 `1`。不能把它当成接受了一个 draft。
2. 对齐 verifier 的 base token、K 个 draft、bonus token、EOS/停止截断语义；逐个变量追查生产者和消费者。
3. 在相同 accepted 前缀下比状态：非投机顺序执行的 state，和投机拒绝恢复后下一轮真正读取的 state。不同执行阶段的 state 不可直接比较。
4. 检查 conv window 偏移和 temporal speculative block 选择，不能把 recurrent state 当作普通 attention KV 仅缩短 `seq_len`。
5. 核对 accepted 输入快照与输出 buffer，避免一个 state CTA 写回计数后，另一个 CTA 读到已经变化的决策值。

### 只在 block 边界、prefix hit 或长 prompt 出错

- 用运行时解析后的 Mamba block size `B` 构造 `B-1/B/B+1`、`2B-1/2B/2B+1` 的 token 长度。不要用字符数或假定 attention kernel block 等于 Mamba checkpoint interval。
- 先做冷缓存，再相同前缀重放，再共享前缀但后缀不同；分别核对各 group 命中长度和实际物化的 checkpoint。
- 检查 `_mamba_block_aligned_split` 与 `MambaManager.allocate_new_blocks` 的边界契约。只有 hash 相同不足以证明 recurrent checkpoint 可恢复。
- PD consumer 已有 computed prefix 时需要保留完整 verifier window；`kv_both` 的冷 prefill 不能笼统绕过对齐。以当前 patch 的 role 和 prefix 条件为准。
- cache mode 的默认值、KV store 强制 align、hidden-state extraction 例外、最终 `mamba_block_size` 都要从 resolved config 判断。

### 只在 V2、多请求或图重放出错

- `idx_mapping` 是 batch row 到持久槽位；postprocess 中状态计数按槽位寻址，block table 按 batch row 寻址。负映射是跳过标记，不得当作 Python 负下标。
- padded 行不能写真实请求的状态。核对纯 spec capture 后，真实 batch 的 metadata 是否刷新了同一块持久 buffer。
- 查新请求复用槽位时 accepted count 是否重置、恢复请求是否重新播种 state index。V1 还需检查 finished/preempted/resumed 清理和 load-only step。
- 检查 `UniformTypeKVCacheSpecs`：只解包同构的 Mamba specs；异构 shape/dtype/mamba type 不得被首层 spec 吞掉。
- 图捕获后不能仅替换 Python tensor 对象；确认 graph 使用的地址、buffer 生命周期和更新顺序。

### 只在按层 KV 加载或 PD 恢复后出错

- 先画出 load completion → 该层 state pre-copy → 该层 forward → save 的实际事件/stream 顺序。
- V2 的 `AscendMambaHybridModelState.preprocess_state` 可以只生成决策，把拷贝延迟到 `do_mamba_copy_for_layer`。提早执行全层 bulk copy 可能读取半加载状态。
- 查 `_finish_previous_layerwise_mamba_copy` 的 pending layers；它在下一次 preprocess 检查，不要因 sample 路径异步而错误移动到同一步末尾。
- V1 查 `prepare_mamba_copy_by_layer`、`do_mamba_copy_block_for_layer`、`finish_mamba_copy_by_layer`；CPU staging 数据不得在消费者尚未使用时复用。
- attention DCP 分片规则不能直接套到 recurrent state。当前 `AscendMambaManager.find_longest_cache_hit` 为 Mamba lookup 传入 `dcp_world_size=1`，其它并行条件仍需单独追踪。

## 修改与验证

优先修正产生错误元数据/状态的最小层级，保留 request 生命周期、group 独立分配、NPU stream 顺序。不要用清空全部 state、关闭 prefix cache 或强制 eager 掩盖恢复错误；这些可作为隔离实验。

为此次问题选择有因果关系的用例，不做所有开关的笛卡尔积：

- accepted draft 为 0/部分/全部；同时区分实际 sampler count 和 state count。
- 跨块与不跨块，`src==dst`、不同 group 物理 ID、SD/DS conv 布局、padded page。
- 请求重排、槽位复用、抢占恢复；若修改 PP 路径，再测 `idx_mapping=-1`。
- 冷 prefill、prefix 命中后续写、混合 prefill/decode；若修改 connector，再测 load-only 和延迟逐层加载。
- eager、重复 graph replay 且每轮 metadata 不同。

按参考中的已有测试扩展：CPU/mock 验证决策，NPU 算子测试验证真实 copy/guards，模型测试验证恢复后的 token/logits。无 NPU 时交付可运行命令与未验证项，不能把 mock 通过视为设备精度通过。

交付应包含：第一条不成立的状态不变量、触发 step/请求、源码符号、最小反例、修改及回归证据；尚未证实的原因单列为假设。
