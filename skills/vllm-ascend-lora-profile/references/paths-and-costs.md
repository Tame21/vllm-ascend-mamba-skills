# LoRA 路径、阈值和成本

路径相对于含 `vllm/` 与 `vllm-ascend/` 的工作区；以实际运行环境 import 路径和当前源码复核。

## 路由图的源码锚点

| 文件/符号 | 性能分析用途 |
| --- | --- |
| `vllm-ascend/vllm_ascend/platform.py:get_punica_wrapper` | 平台选择 `PunicaWrapperNPU` |
| `vllm-ascend/vllm_ascend/lora/utils.py:refresh_all_lora_classes` | Ascend wrapper 排在可替换层列表前面；检查实际实例 |
| 同文件 `_PackedLoRAAWeightsMixin`、`_TransposedLoRABMixin` | `set_lora/reset_lora` 构造/清零 packed A/B；merged B 是 block diagonal |
| `vllm-ascend/vllm_ascend/lora/punica_npu.py:PunicaWrapperNPU` | `no_lora`、single-slot gate、底层算子选择、metadata、workspace |
| `vllm-ascend/vllm_ascend/lora/lora_ops.py:lora_linear` | `buffer` 以外的 custom-op 分发：通用 kernel 分支，或单 slot 的 Triton 尝试/packed matmul 分支 |
| `vllm/vllm/v1/worker/lora_model_runner_mixin.py:_set_active_loras` | 真实 `LoRAMapping.is_prefill=True` 的生产者 |
| `vllm/vllm/v1/worker/gpu/lora_utils.py` | V2 active requests、capture cases；继续追实际 runner 调用 |
| `vllm/vllm/lora/punica_wrapper/punica_base.py` | 索引与分段元数据，区分 Python 逻辑和设备搬运 |
| `vllm/vllm/lora/ops/torch_ops/lora_ops.py` | fallback 的 gather、dtype conversion、einsum、repeat_interleave |
| `vllm/vllm/lora/layers/column_parallel_linear.py`、`row_parallel_linear.py` | TP slicing、fully-sharded shrink/expand 和 all-gather/all-reduce |

`max_loras=1` 的 wrapper 不由“本轮 active 数等于 1”触发；A2/A3 gate 在初始化时确定。
通用底层算子初始化：不支持 `HardwareCapability.LORA_CUSTOM_OPS` 或 `max_lora_rank>=128`
时选上游 torch ops，否则选 AscendC ops。之后 single-slot dense 分发仍可能走 matmul；
因此“rank 上限 128 → 所有层一定走 torch fallback”也是错误简化。

显式 buffer、embedding、fused MoE、fully-sharded 的调用接口不同，先看真实 caller。
MoE 用 `(lora_slot * num_experts + expert_id)` 组合索引后走 bgmv 的路径，不是 dense packed linear。
不要把 dense 的优化门槛和成本模型直接套到 MoE、量化 expert 或多模态 tower。

## 单 slot Triton 的确切门槛

`lora_ops.py:_try_lora_linear_triton` 当前要求：

- Triton 可导入、`get_forward_context().capturing` 为真；不是任意 eager forward。
- x/y/A/B 都为连续 BF16，flatten 后 token 数不大于 128。
- per-slice rank 为 8/16/32/64；merged projections 需 packed A/B 且 slice 总宽度等于 y 宽度。
- output width ≥8192 时，rank≤16 且 tokens>32、或更高 rank 且 tokens>64 回退。
- output width ≥5120、rank≥16、tokens>64 也回退。

rank 是 stacked tensor 的有效维度，而非只读 adapter 配置文件；记录 TP-local shape。
若满足：`_lora_shrink_splitk_kernel` → `_lora_expand_kernel` 或 `_lora_expand_sliced_kernel`。
split-K 为 4；workspace 按 packed rank 缓存为 FP32。sliced expand 还要求多 slice、
每 slice rank≥32、slice 数≤4；其它 eligible 情况使用普通 expand。
捕获后的图 replay 直接执行这些设备工作，Python capturing 判断不在每次 replay 重新运行。

不满足时 `_lora_linear_matmul` 执行两次 matmul，加 adapter mask、可选 scale 和残差。
merged packed B 含零块；普通 packed matmul 仍以实际 dense shape 工作，不能只按非零参数量估算成本。
具体算子边界与测试说明可读 `vllm-ascend/vllm_ascend/ops/triton/docs/lora_linear.md`，
但最终选择以当前代码和 trace 为准。

## trace 中寻找什么

| 观察点 | 源码或解释 |
| --- | --- |
| `_vllm_ascend_lora::lora_linear` | 自定义算子壳，继续展开真实 NPU kernel；壳的 CPU 时间不等于设备执行时间 |
| `bgmv_shrink` / `bgmv_expand` / `sgmv_shrink` / `sgmv_expand` | `csrc/torch_binding.cpp` 的 `OpCommand.Name`；设备实现是 `csrc/kernels/` 下对应 `.cpp` |
| `_lora_shrink_splitk_kernel` / `_lora_expand_kernel` / `_lora_expand_sliced_kernel` | 单 slot Triton；trace 名可能带编译后缀 |
| matmul + mask + add | `_lora_linear_matmul`；不能把同时间附近的 base GEMM 全算进 LoRA |
| gather/index、einsum、cast、repeat_interleave | 上游 torch fallback；是否显著以 trace 为准 |
| packed copy / zero | `utils.py:set_lora/reset_lora`，定位到激活还是热 forward |
| 通信与较小 GEMM 交错 | fully-sharded 的本地计算/通信交换，不能只比较 kernel 总和 |

`_use_sgmv` 看 wrapper 的 `is_prefill`，其来源可由 runner 强制为真；
310P graph capture 时会改选 BGMV。用 scheduler 的 scheduled tokens/query lengths 判断真实阶段。
`bgmv_expand` 和 `sgmv_expand` 的 C++ `y_out = y` 是别名，不能从返回 tensor 推断又分配一份输出。
flatten/reshape/transpose 也需检查 strides，而 `copy_`、cast、gather 是更直接的搬运候选。
新增 instrumentation 应在缩小用例的阶段边界，不在每层强制同步。

## 容量、加载与调度

`vllm/vllm/config/lora.py:LoRAConfig`：默认 max rank 16、max_loras 1、fully-sharded false；
`max_cpu_loras=None` 解析成 max_loras，且不能小于 max_loras。记录 resolved values，别把 None 当无限容量。

`vllm/vllm/lora/worker_manager.py:LRUCacheWorkerLoRAManager.add_adapter`：
adapter 缺失时 `_load_adapter` 从 checkpoint 读到 CPU、验证并转换；加载成功后才处理 CPU LRU 淘汰，
再调用 model manager 激活。CPU cache hit 不等于 NPU slot hit。
`model_manager.py:LoRAModelManager.activate_adapter` 遍历模块 `set_lora/reset_lora`；
`LRUCacheLoRAModelManager.activate_adapter` 在 active slots 满时移除最旧 active adapter。
`set_adapter_mapping` 还比较 slot layout，ID 未变也可能因外部加载重分配而刷新映射。

`vllm/vllm/v1/core/sched/scheduler.py:Scheduler.schedule` 会跳过使 active LoRA 数超过
`max_loras` 的 waiting 请求。TTFT 上升可能来自 adapter 槽约束改变调度与 batching，
不要仅看 matrix kernel。诊断时记录请求到达顺序、active IDs、scheduled tokens 和等待时间。

`vllm/vllm/entrypoints/openai/models/serving.py:OpenAIServingModels.init_static_loras`
在启动时加载静态模块；首个 HTTP 请求不是天然的“首次磁盘加载”。
要测 CPU 冷加载需证明发生 eviction/reload，或把独立服务启动/API load 的耗时单独计入。
不为基准默认清 OS page cache；区分“CPU adapter cache 冷”与“文件系统缓存冷”。

## graph 与验证入口

`vllm/vllm/v1/cudagraph_dispatcher.py:_get_lora_cases` 区分 LoRA 未启用、base/LoRA
specialization、active-count capture cases。相关配置在 `config/compilation.py` 和 `config/lora.py`。
Ascend MRV1 `_determine_batch_execution_and_padding` 传递 has_lora/active count；
MRV2 使用 `worker/v2/aclgraph_utils.py` 及上游 capture hooks。配置存在不保证每个 NPU 模型组合有效。
不要声称 `specialize_active_lora` 必然优化 Ascend kernel；先确认实际 dispatch 和生成的图。

局部变更可扩展 `vllm-ascend/tests/ut/lora/test_punica_npu.py`、`test_lora_ops.py`、`test_utils.py`。
真实 arithmetic 入口为 Ascend 仓库的 `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_lora_linear.py`、
`tests/e2e/nightly/single_node/ops/singlecard_ops/test_bgmv_shrink.py` 和
`tests/e2e/nightly/single_node/ops/singlecard_ops/test_bgmv_expand.py`；读取参数化、skip 和精度阈值再运行。
mock 验证路由不等于 NPU 数值通过；单算子正确也不能证明服务在 slot 重用后仍应用正确 adapter。
