---
name: vllm-ascend-lora-kernel
description: 开发和优化 vLLM Ascend 的 PEFT LoRA 算子，联合验证 A/B 布局、缩放、bgmv/sgmv、packed matmul、Triton split-K、TP 与 MoE 路由的数值正确性和 NPU 性能。适用于 kernel 替换、shape 分派、精度回归及微基准；不用于把模型内生 MLA 低秩投影当成 LoRA adapter。
---

# LoRA 算子精度与性能联合验证

先证明同一输入、同一 adapter 映射产生正确 delta，再证明实际 serving 分支在目标 shape 上更快。
输出应能回答：优化了哪个调用路径、适用什么 shape、误差来自哪里、收益是否包含 metadata/通信/图重放成本。

## 确认代码、环境和实际分派

- 默认项目根包含 `vllm/`、`vllm-ascend/`；先读适用 `AGENTS.md`，记录两个 HEAD、dirty 状态、已安装包来源。
- 本技能基于 vLLM `d61081dc3d3f1740a5d8bf82608b62974393c2de`、Ascend `9b8fc5d728e1ea54dc277562b296112d905e15c1`。
- 下文路径相对项目根；版本变化后按符号重定位，以当前代码和实际 callable 为准。
- 记录 NPU 型号、CANN/torch/torch_npu/Triton Ascend、base/LoRA dtype、量化方式、TP/EP、fully-sharded、runner 和图模式。
- 明确训练 rank、配置 `max_lora_rank`、实际 local rank、packed total rank，不能只写一个 r。
- 记录输入/输出 shape、stride、token 数、output slices、活跃 adapter 数、slot 映射与 base-only/padding 行。
- 查看 `vllm-ascend/vllm_ascend/lora/punica_npu.py::PunicaWrapperNPU` 实例绑定的 bgmv/sgmv callable 和 `_single_lora_slot`。
- 再沿 `vllm-ascend/vllm_ascend/lora/lora_ops.py::lora_linear` 的真实分支追踪；不能仅凭日志中启用了 LoRA 推断 kernel。

| 实际入口 | 当前选择条件及限制 |
| --- | --- |
| 通用 shrink/expand | `PunicaWrapperNPU.__init__` 按 `LORA_CUSTOM_OPS` 和 `max_lora_rank >= 128` 选择 AscendC 或 torch backend |
| bgmv / sgmv | `_use_sgmv` 读取 wrapper 的 `is_prefill`；该 mapping 字段未必等于实际请求阶段，310P capture 有回退 |
| 单 slot packed 路径 | `_single_lora_slot` 要求 A2/A3、`max_loras == 1`、非 fully-sharded；多 slice 还要 packed A/B |
| Triton dense 路径 | `_try_lora_linear_triton` 要求 forward context 正在 capture、连续 BF16、支持 rank/token/输出宽度 |
| packed matmul | single-slot 满足但 Triton 未命中时；一般 eager 走这里 |
| MoE adapter | 经过 expert dispatch 后的逐行 combined index，不能直接套 dense single-slot 路径 |

`max_lora_rank >= 128` 的 backend 回退不代表所有 dense 调用都走 torch bgmv；single-slot 仍可能走 packed matmul。
上游 `LoRAModelRunnerMixin._set_active_loras` 当前固定创建 `is_prefill=True` 的 mapping；此入口下 decode 也可能走 sgmv。
因此记录实际 mapping/调用者，不把 `_shrink_decode` 方法存在当作 decode 必经 bgmv 的证据。
kernel 直接调用测试绕过上述 gate，只证明算术实现，不能证明 serving 会采用该 kernel。
详细契约见 [references/kernel-contracts.md](references/kernel-contracts.md)。

## 建立未混淆的数学 oracle

对每个 token i、adapter slot s 和输出 slice j，记录逻辑关系：
`delta[i,j] = ((x[i] @ A[s,j].T) * scale) @ B[s,j].T`。
原始 A 为 `[r,in]`，B 为 `[out,r]`；核对物理 stack、TP-local shard 和 packed 布局后才转换。
从相同低精度输入/权重上转 FP32 建 oracle；这比较运算误差，不把权重预先量化误差混入 kernel 差异。

- `vllm/vllm/lora/lora_weights.py::LoRALayerWeights.optimize` 将 scaling 原地折入 B 并设为 1。
- `PackedLoRALayerWeights.pack` 会先 optimize 各 slice；构造 oracle 时确认拿到的是折叠前还是折叠后 B。
- 非 1 的 scale 测试很必要，但生产层以 1 调用可能因为已折叠，不能判为漏乘。
- 同时比较 shrink、LoRA delta 和最终 y；base y 很大时，最终输出 relative L2 很小也可能掩盖完全错误的 delta。
- `add_inputs=True` 比较 `y_initial + delta`；overwrite 合约比较 delta，并检查未命中的行应该保留还是清零。
- 使用非零 y_initial 区分残差加法和覆盖；只用 zeros 会让错误实现通过。
- 精确核对 adapter IDs、slice 边界、shape；浮点报告最大绝对误差、relative L2、首差坐标、NaN/Inf 和 per-row/slice 误差。
- residual-add 下 inactive/base-only 行及输出 slice 外区域做 bit-exact 不变性检查；overwrite 则按明确的覆盖/清零合约检查。
- 浮点误差阈值按 dtype、路径和基线实测设定。

## 不可从 API 名字推断的语义

当前 `lora_ops.py` 的 AscendC bgmv/sgmv expand wrappers 接受 `add_inputs`，但没有传给底层算子。
`csrc/kernels/bgmv_expand.cpp` 和 `sgmv_expand.cpp` 读取旧 y 后加 delta；现有 UT 明确验证 wrapper 丢弃该参数。
高层通用 dense 路径也有限制：`lora_linear` 的 `use_kernel` 分支没有传递 `add_inputs`，
`_lora_linear_kernel` 及 `add_lora_linear` 的显式 buffer 分支固定以 True 调用 expand；
所以即使底层切成 torch ops，高层 False 也不会自动获得覆盖语义。`no_lora` 分支直接返回。
`_lora_linear_matmul` 则真正区分 `y.add_` / `y.copy_`；Triton expand 始终 residual add。
因此新增调用方需要 overwrite 时必须检查分派或构造适当输出，不能宣称各 backend 的 False 参数等价。

`output_slices` 是各逻辑输出宽度，`slice_offset`/`offset_start` 是写入位置；核对累计偏移和 TP 后宽度。
sgmv wrapper 传 `seq_len_tensor`，没有把所有上游 batch metadata 都传给 AscendC；从实际消费字段构造 oracle。
负 adapter index 的 skip 语义与 Python `weights[-1]` 完全不同，参考实现必须显式处理 inactive 行。

## split-K、packing 与图缓存检查

按 [kernel-contracts](references/kernel-contracts.md) 核对物理布局，再修改 tile/split 参数。
当前 Triton shrink 沿 hidden K 分四段，以 FP32 workspace 存 partial；expand FP32 相加后才做 mask/scale，随后转到 B dtype 做投影。
packed A 沿 rank 拼接，packed B 是 block diagonal；sliced expand 只计算所属 slice 的 rank 块，避免零块乘法。
不要将“逻辑 slice 数”乘错成 local rank，或将 `B_TRANSPOSED` 当作对所有权重的全局布局开关。

最小有意义的边界集合按改动选择：

- token 数：1、tile 边界前后、捕获边界及 128/129；区分真实 token 与 graph padding。
- rank：当前支持的 8/16/32/64，以及实际触发 fallback 的配置；记录 padding 后物理 rank。
- 输出：单 slice、QKV 三 slice、QKVZ 四 slice、不等宽 slice、完整宽度的边缘 mask。
- adapter：有效 slot、`-1`、mixed base/adapter、全 base、slot reset 后换权重；连续 replay 更换 token mask。
- shape/stride：真实 TP-local 输入、3D flatten、非连续输入；未支持布局应正确 fallback，不能静默按 contiguous 寻址。
- graph：同一 graph replay 内更新 A/B 或 mask 时保持 storage 地址稳定；对比首次 capture 与后续 replay。

改变 `_LORA_TRITON_K_SPLIT`、BLOCK_M/N/K 或 fallback 阈值时，同时检查数值归约顺序和 workspace 分配/复用。
新增更快 kernel 不能留下不同 shape 共用旧 workspace、inactive 行沿用上一轮值或捕获了过期 mask 的问题。

## TP、fully-sharded 与 MoE

读 `vllm/vllm/lora/layers/row_parallel_linear.py`、`column_parallel_linear.py` 的 slice 和通信逻辑。
普通 row-parallel A 切 input，B 保留完整；普通 column-parallel A 保留，B 按 output 分片，QKV/GQA 有专门规则。
fully-sharded column 路径还切 A 的 rank 并 all-gather shrink；row 路径切 B 输出并对 shrink all-reduce 后按 rank offset expand。
必须在应完成 collective 的边界比较全量结果，不能用一个 rank 的 partial output 对全模型 oracle。
Ascend packed wrapper 带 `_not_fully_sharded_can_replace`，不能由某一非分片测试通过推断 fully-sharded 也支持。

MoE 先读 `vllm-ascend/vllm_ascend/lora/fused_moe.py`：token/adapter indices 要随 TP split、top-k 展开、permutation 和 EP all-to-all 一起移动。
`PunicaWrapperNPU.add_lora_fused_moe` 当前要求预展开行 `top_k_num=1`，索引为 `slot * num_local_experts + expert`，inactive 为 -1。
验证 gate/up、activation、down、routed weight 的注入位置，避免把 gate/up 的 LoRA 加在 activation 后或重复乘 top-k weight。
使用 MoE 自身的 `moe_parallel_config.tp_size/tp_rank`，EP 可能已将它缩为 1；全局 TP 再切一次会错。
动态 EPLB 和 FusedMC2 在当前 guard 中被拒绝；共享 expert 走 dense wrapper；量化 MoE 只认已注册的实现。
`fully_sharded` 参数存在或被丢弃都不能证明该 MoE 组合受支持，须追上层实际分片和 NPU 测试。

## 精度先过，再测实际成本

执行 [references/tests-and-timing.md](references/tests-and-timing.md) 的最相关现有测试，必要时扩展对应用例。
先独立预热/JIT/捕获/分配；计时区间不包含 oracle、日志、`.cpu()`、`.item()` 或每次分配 workspace。
使用 NPU event 或同步包围的 wall-clock；只用主机计时包围异步 launch 会测到提交速度。
in-place y 每次必须恢复到同一初始值，或完整 base layer 每轮覆盖 y；不可反复累加导致误差、饱和和工作集改变。
分别报告 raw kernel、实际 dispatcher 路径、层含通信、serving 延迟/吞吐；标明 graph replay 与 eager、shape 与活跃 adapter 分布。
不要把 direct Triton 的快于 matmul 数字直接当 serving 收益，也不要照搬 upstream CUDA benchmark 作为 NPU 实测。

交付版本/配置、实际 callable、shape 分派表、oracle 结果、选定容差依据、warmup/repeats/同步方法和延迟分布。
优化需要回跑最初瓶颈场景并检查 fallback shape；无 NPU 时仅提交源码契约和待执行命令，明确未实测性能。
