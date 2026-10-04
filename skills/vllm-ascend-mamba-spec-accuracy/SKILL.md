---
name: vllm-ascend-mamba-spec-accuracy
description: 定位 vLLM 与 vLLM Ascend 中 Mamba/GDN 混合模型投机推理的 token、logits、递归状态精度首差；适用于 spec 开关、ACL Graph、混合批次、chunked prefill、prefix cache 或状态回滚引起的输出偏差。
---

# Mamba 类投机推理精度首差定位

把“输出不一样”收敛到同一请求、同一逻辑 token 位置、同一层的首个异常边界，
区分输入/元数据错误、状态生命周期错误、算子数值误差和采样差异，再设计回归验证。
默认在项目根目录的 `vllm/` 与 `vllm-ascend/` 双仓布局中工作；目录不同则先定位仓库。

## 先确认实现与证据

1. 读取将要修改的仓库 `AGENTS.md`，记录两个仓库 HEAD 与工作区 diff。
   本技能源码基线：vLLM `d61081dc3d3f1740a5d8bf82608b62974393c2de`，
   vLLM Ascend `9b8fc5d728e1ea54dc277562b296112d905e15c1`。
   版本变化后用 `rg` 重定位符号和调用者；当前工作区的实际实现优先。
2. 记录模型/权重与 tokenizer revision、chat template、输入 token IDs、dtype、量化、
   NPU/CANN/torch/torch_npu 版本、TP/PP/DP/PCP、运行器版本、spec method/K、
   graph 模式、chunked prefill、prefix caching、`mamba_cache_mode`、同步/异步调度。
   从实际配置对象或启动日志取值，不依赖未确认的默认值。
3. 确认 `MambaSpec.mamba_type`、模型层类型和实际注册的 NPU custom op。
   `AscendGatedDeltaNetAttention` 的 GDN recurrence 与 `MambaMixer2` 的 SSD/SSU 是不同数学路径；
   不能因为共用 Mamba cache 管理就套用相同 state shape、转置或 kernel oracle。
   上游 Mamba2/ReplaySSM 路径存在，不代表该 NPU 环境已经支持。
4. 阅读 [源码定位与验证入口](references/code-and-tests.md) 中与当前运行器、模型对应的部分。
   先确定 runtime 真的命中了候选代码，再下断言或插入 dump。

## 建立可比较的最小基线

先把相同配置运行两次，确认基线自身是否稳定。使用相同输入 token IDs、greedy
(`temperature=0`)、相同停止/EOS/max_tokens 条件；只固定 seed 不能保证随机采样逐 token 相等。
核对 chat 模板、BOS/EOS、prompt truncation、logits processors 和 penalties。
缩短输出到能稳定复现首差的范围，不用完整自然语言文本相似度代替 token 比较。

按失败特征逐项增加变量，避免一口气改变整个配置矩阵：

| 对照 | 主要回答的问题 |
| --- | --- |
| 同一 NPU，spec off / spec on，eager，单请求 | 偏差是否由投机路径引入 |
| spec on，K=1 / 故障 K | 多候选 token 的索引或回滚是否出错 |
| 固定 spec 配置，eager / 故障 graph 模式 | 捕获的输入地址、padding 或重放元数据是否陈旧 |
| 单请求 / 不等长批次 / 请求退出和新请求加入 | token partition、请求行重排、state slot 是否串位 |
| 完整 prefill / 强制实际发生的 chunked prefill | 初始状态、chunk 边界、最后一 token chunk 是否一致 |
| 冷缓存 / 相同前缀命中 / 请求结束后的新请求 | state restore、block 复用、生命周期是否正确 |
| 固定前述条件，同步 / 异步调度 | accepted count 与请求映射、事件时序是否错位 |

若关闭某项被配置校验拒绝，记录“不支持该对照”，用支持的最近组合隔离；不要绕开校验。
记录实际 chunk 长度和 cache hit，启用开关本身不证明执行过目标分支。
跨设备/HF 对照可帮助判断非投机基线，但先确认权重、数值路径和语义一致，
不要把跨设备浮点误差直接归因于 NPU 投机实现。

## 找到首个分歧边界

1. 先输出每个请求的首个不同 token 位置、前后 token IDs、结束原因和输出长度。
   spec 一轮输出多个 token，按请求的逻辑位置对齐，不按 engine step 或 batch 行号直接相减。
2. 在首差位置比较 **target** 原始 logits、processor 后 logits、top-k 和 top1/top2 margin；
   别把 draft logits 与 target logits 当作同一 oracle。
   logprobs 只有 top-k 时无法证明其余词表 logits 相同。
3. 以同一已确认 token 前缀作 teacher forcing/受控 replay，比较相同逻辑位置的 target 计算。
   一旦生成前缀已经不同，后续 hidden state 差异没有定位首因的价值。
   额外 prefill 会改变执行路径，应标明它是辅助 oracle；保留故障 decode 路径复现。
4. 若 target logits 相同而 token 不同，追踪 sampling metadata、target/bonus logits indices、
   draft token IDs、reject 结果、logits processors、RNG 消耗；不要先改 GDN kernel。
5. 若 logits 首差来自模型内部，按层二分：层输入 → projection → conv 输出 →
   q/k 归一化与 g/beta → recurrent/chunk 输出和更新后 state → gate/norm/output projection。
   在同一份输入和初始 state 上重放最早有差异的算子，再讨论容差。
6. 同步保存 conv state 与 SSM state 的更新前/更新后/提交或恢复后快照；
   只保存 Python tensor 引用会被后续原地写覆盖，应对选中的小片段做独立快照。

## 优先检查的状态与元数据

- 核对 `query_start_loc`、真实/填充 request/token 数、seq_lens、position、block table、
  `spec_token_indx` / `non_spec_token_indx`、state indices 与 request ID 的对应关系。
  分组后的 token 顺序必须可逆；逻辑内容正确但写回错行，也会表现为“数值误差”。
- GDN spec conv 和 recurrence 消费 accepted counts；当前 builder 明确为 one-based 计数。
  V1 Ascend 的异步 hybrid 路径在 `_update_states_after_model_execute` 用
  `(output_token_ids != -1).sum(dim=1)` 计数；同步路径转交父类。
  V2 由 sampler 产生 `num_sampled`，经 `GPUModelRunner.postprocess_sampled` 传给
  `MambaHybridModelState.postprocess_state`，再按 `idx_mapping` 写入持久请求槽位。
  未采样的 chunked prefill 会传 0，该 state 方法将它转为中性 accepted count 1，
  不表示新增一个输出 token。有效输出数也不等于“draft 被接受个数”；确认消费者语义后再改 `+1/-1`。
- 检查零 draft 接受、部分接受、全部接受、bonus token、EOS/停止、请求移出后的下一轮。
  根据当前算法跟踪保留哪一份已验证前缀状态，排除 rejected draft 污染下一轮 state。
  `align`、`all` 和其他实际支持的 cache 模式有不同提交/复制路径，分别核对。
- graph padding 的虚请求应产生零长度 GDN query 段；dummy accepted count 为 1
  并不表示应消费一个 token。对齐 PAD/NULL 约定，检查 padding 没有修改真实请求 state。
- 一个 token 的有状态 prompt chunk 与首 token prefill 的处理不同；
  检查 `has_initial_state`、chunk 的起始位置与最后 state scatter，而非只看 query 长度。
- 异步运行器检查 accepted counts 的上一轮行顺序、`prev_positions`、事件同步和新请求初始化。
  远端/按层 KV 加载场景还要确认 state copy 在对应层 load 完成之后、conv/recurrence 之前。
- state cache 可能是 page-strided view；对照 shape、stride、storage offset 与布局规范。
  GDN fallback 的转置、fused prefill 的 dtype 转换都是实际分支差异，不能盲目统一。

## Dump 和数值判据

仅采集能定位首差的请求、层和位置；优先 eager 小样本，设备同步与 D2H dump 会影响时序和性能。
graph 问题需保留原重放路径，在合法的边界采样；不要在被捕获 forward 内直接加 Python IO。
整数 token/indices/counts 用精确比较；浮点用明确的 `abs(candidate-ref) <= atol + rtol*abs(ref)`。
阈值依据 dtype、算子 reference、基线波动与模型输出影响设定，不给所有状态套固定阈值。
报告 shape/dtype/layout、非有限值、首差坐标、最大绝对误差、token margin；cosine 高并不能排除局部异常。
NaN 必须定位来源；mask 后 logits 的同位 `-Inf` 可以是预期行为，原始 logits/state 的 Inf 需另查。

需要离线比较时用 [dump 格式与脚本](references/dump-format.md) 以及
`scripts/compare_dumps.py`；它只读取 JSON/JSONL，无 torch/NPU 依赖，不加载 pickle。
使用环境中已确认的 Python 解释器；如果在 vLLM 仓库执行其测试，遵守该仓库的 uv/venv 规则。
脚本给出的“首差”是参考文件的记录顺序，应按逻辑执行顺序导出。

## 闭环验证与交付

选择已有 metadata/state/sampler 单测扩展回归，再在真实 NPU 上跑故障组合和必要对照。
验证至少覆盖首差消失、未接受分支的状态不污染后续 token、请求重排/边界场景、graph/eager 一致性。
恢复真实采样与长输出后做模型评测；greedy 最小复现通过并不等于完整模型精度验证。
acceptance rate 是投机效率信号，不是 target 正确性的证明；既要看输出/任务指标，也要看状态不变量。
测试文件名或 docstring 写着 correctness 不代表实际有比较断言，先读函数内容。
交付最小复现配置、首差证据、源码符号、根因与排除项、回归命令及实际结果。
没有 NPU、权重或环境时提供可运行步骤并写明未验证项，不能把静态检查或 mock 单测称为 NPU 精度通过。
