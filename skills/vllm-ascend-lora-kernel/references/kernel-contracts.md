# 源码契约：dense、packed、Triton 与 MoE

路径均相对包含两个仓库的项目根；本文件是快照导航，开始任务时复核实际版本。

## 1. 权重、scale 与 packed layout

`vllm/vllm/lora/lora_weights.py::LoRALayerWeights` 的原始 A/B 是 `[rank,input]` / `[output,rank]`。
`optimize()` 原地执行 `B *= scaling`，然后 `scaling=1`；`PackedLoRALayerWeights.pack` 会优化子 adapter。
模型的 rank-stabilized 等 scaling 应从加载后的 PEFTHelper/LoRALayerWeights 查，不能始终硬编码 alpha/r。

`vllm-ascend/vllm_ascend/lora/utils.py::_PackedLoRAAWeightsMixin` 先让父类执行 TP 切片、padding、slot 写入，再打包：

| 对象 | 物理形状或作用 |
| --- | --- |
| `lora_a_stacked[j]` | 一个 slice 的 `[max_loras,1,rank,input_local]` |
| `lora_b_stacked[j]` | `[max_loras,1,output_local_j,rank]`，可能有配置级 padding |
| `lora_a_packed` | `[max_loras,1,n_slices*rank,input_local]`，按 slice 拼 A 行 |
| `lora_b_packed` | `[max_loras,1,n_slices*rank,sum(output_slices)]`，每个对角块写入 B_j 的转置 |
| `_TransposedLoRABMixin.lora_b_packed` | 单 slice 的 `[max_loras,1,rank,output_local]` 转置副本 |

`set_lora()` 重建对应 slot；`reset_lora()` 清理 stacked 和 packed 内容。检查动态换 adapter 是否同步更新两份权重。
`_apply_packed_lora` 传 scale=1，是生产路径预折叠约定；3D x/y 会 flatten 再恢复输出形状。
单 slice 的 matmul 实现直接取 stacked B 转置；Triton 可选择预转置副本，不能以“存在 packed B”推断 matmul 用了它。

## 2. 通用 AscendC shrink/expand

入口：`vllm-ascend/vllm_ascend/lora/lora_ops.py::{bgmv,sgmv}_{shrink,expand,expand_slice}`。
实现：`vllm-ascend/csrc/kernels/bgmv_shrink.cpp`、`bgmv_expand.cpp`、`sgmv_shrink.cpp`、`sgmv_expand.cpp`。

- shrink 的 X/W 取 kernel scalar dtype，输出 buffer 是 FP32；scale 在 shrink 结果上应用。
- expand 的输入是 FP32，W/Y 是 kernel scalar dtype，读取旧 y 后 residual add。
- expand 当前公开支持 rank 8/16/32/64；更大的配置在 wrapper backend 选择处可能落 torch 路径。
- bgmv 按每 token adapter index，sgmv 根据 sequence lengths 对分段 token 找 adapter；均核实下游索引类型和排列。
- negative index 跳过，输出保留原内容。复用 shrink workspace 时审计 skip 行是否会被后续路径消费。
- expand 写入区间为 `[slice_offset, slice_offset + slice_size)`，输出 stride 依据 full output width，不是 slice width。
- Python wrapper 的 `add_inputs` 没有向 AscendC 传递；不要用 overwrite 用例调用该路径并假设和 torch 一致。

通用 dense 高层同样固定 residual add：`lora_linear` 的 `use_kernel` 分支不传 `add_inputs`，
`_lora_linear_kernel` 和 `add_lora_linear` 的显式 buffer 分支都固定传 True。底层换成 torch ops
也不能恢复这条高层路径的 overwrite；单槽 matmul 的 add/copy 与 Triton 的始终 add 需分别验证。

`PunicaWrapperNPU._get_shrink_buffer` 按 `(n_slices,rank)` 缓存 FP32 buffer；设备值更新和复用行为应一起验证。
`_use_sgmv` 依据 metadata 的 `is_prefill`，并对 310P capture 特殊回退 bgmv。
`vllm/vllm/v1/worker/lora_model_runner_mixin.py::LoRAModelRunnerMixin._set_active_loras` 当前固定写 `is_prefill=True`，
因此用该入口建立的 decode mapping 也能走 sgmv；不能把 kernel 名称与请求阶段一一对应。
测试时记录实际 mapping 值、构造入口、device 和 capture 条件。

## 3. Triton capture 路径的实际 gate

入口：`lora_ops.py::_try_lora_linear_triton`，它被 `lora_linear` custom op 调用，后者声明 `mutates_args={"y"}`。
先要求 single-slot packed 路径，再逐项满足：

- forward context 的 `capturing` 为真，Triton 可导入；这与一般“启用了图模式”不同。
- token 数最多 128；x/y/A/B 连续 BF16；每 slice rank 属于 `{8,16,32,64}`。
- 多 slice 必须提供 packed A/B，且 `sum(output_slices)==y.size(1)`。
- output >=8192 时，rank<=16 的 token>32 或较高 rank 的 token>64 回 matmul。
- output >=5120 且 rank>=16、token>64 同样回 matmul。

shape 阈值是当前实现的分派，不是硬件不支持声明；变更前对阈值两侧同时做精度和性能验证。
直调 kernel 绕过 gate 时，调用者仍要满足容量、正 token 数和物理布局；没有 gate 不等于支持零长度或任意 stride。

`_lora_shrink_splitk_kernel`：K_SPLIT=4、BLOCK_K=256，分段使用 ceil-div，partial 写 `[4,128,total_rank]` FP32。
`_lora_expand_kernel`：FP32 汇总 partial，乘 adapter mask 和 scale，再 cast 到 B dtype 做 dot，最终加旧 y 后写 BF16。
`B_TRANSPOSED=True` 时 B 是 `[output,rank]`；False 时 B 是 `[rank,output]`。
`_lora_expand_sliced_kernel` 处理 2–4 个 slice 且 per-slice rank>=32；其余使用通用 block diagonal 展开。
检查总 rank、slice rank、tile 累积边界和 output 累积边界，尤其不同宽度 Q/K/V/Z。

mask buffer 来自 `PunicaWrapperNPU._update_base_metadata`，由 token slot 是否为 0 更新；这是 slot，不是用户 adapter ID。
图 replay 依赖其固定地址和正确更新；不要重绑 Python tensor 后以为捕获 graph 自动读取新 storage。

## 4. 分片与通信

普通 row：`vllm/vllm/lora/layers/row_parallel_linear.py::RowParallelLinearWithLoRA` 切 A 的 input 维，B 不切，随后按 base 配置 all-reduce 输出。
普通 column：`column_parallel_linear.py::ColumnParallelLinearWithLoRA` 切 B output；merged QKV 还按 q/kv head 分片和复制规则处理。
fully-sharded column：`_mcp_apply` 对 local-rank shrink 做 all-gather 后 expand；buffer 的 contiguous stride 是契约。
fully-sharded row：`RowParallelLinearWithShardedLoRA.apply` 对 shrink all-reduce，按 TP rank 输出 offset expand，最终依赖 row 输出 reduction。
读取真实 wrapper class、层的 `reduce_results`/`gather_output` 及 collective 结果，不能仅按配置名比较未归约 local 输出。

## 5. MoE 的 adapter index 跟随 expert 行

`vllm-ascend/vllm_ascend/lora/fused_moe.py`：

- `prepare_lora_indices` 截断/填充 inactive slot，按需要 TP split。
- `preprocess_lora_indices` 随 top-k 展开和排序；`all2all_lora_indices` 用相同 splits 做 EP 交换。
- `_recover_moe_lora_routing_allgather` / `_recover_moe_lora_routing_all2all` 恢复每个 dispatched row 的 expert 和 adapter。
- `moe_lora_apply_w13` / `moe_lora_apply_w2` 在 gate/up 和 down 边界注入，空 EP rank 提前返回。
- `AscendFusedMoEWithLoRA` 用 MoE 自身的 TP 配置，避免 EP 下把 local expert 再按全局 TP 切一次。

`PunicaWrapperNPU.add_lora_fused_moe` 将 `[adapter,expert,...]` 展成一个索引轴，构造 `adapter*num_experts+expert`。
disabled adapter 与 base-only 行设 -1；使用 FP32 shrink buffer；要求输入已经展开为每行一个路由 (`top_k_num==1`)。
若 `mul_routed_weight=True`，乘在 shrink buffer 上再 expand；在完整 caller 中确认 routed weight 是否已由其它步骤应用。
MoE callable 的 `fully_sharded` 参数被删除未消费，不表示具备完整分片协议；确认上游分片对象和实际测试。
`_assert_ascend_moe_lora_supported` 拒绝动态 EPLB 和启用 FusedMC2 的组合；共享 expert 有独立 dense LoRA。

量化 MoE 查 `vllm-ascend/vllm_ascend/lora/quant_moe.py::quant_apply_mlp_with_moe_lora` 及注册表。
当前文件注册 dynamic-int8 路径，并在浮点边界注入；未注册 quant type 抛 `NotImplementedError`。
基模量化不能推导 adapter 也量化，不能把浮点 adapter delta 加到未反量化的整型结果。
