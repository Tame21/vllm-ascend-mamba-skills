# 测试路由和 NPU 微基准

命令用于已有 Linux NPU 开发环境；不在本地无卡环境声称验证算子。
从双仓项目根设置解释器，遵守 vLLM 的 uv/venv 管理要求；先确认该解释器导入的两个包来自预期 checkout。

```bash
export LORA_PY="$PWD/vllm/.venv/bin/python"
test -x "$LORA_PY"
git -C vllm rev-parse HEAD
git -C vllm-ascend rev-parse HEAD
rg -n 'def _try_lora_linear_triton|def lora_linear|_LORA_TRITON_' vllm-ascend/vllm_ascend/lora/lora_ops.py
cd vllm-ascend
"$LORA_PY" -m pytest -q tests/ut/lora/test_lora_ops.py tests/ut/lora/test_punica_npu.py
"$LORA_PY" -m pytest -q tests/ut/lora/test_lora.py -k 'packed_single_lora_matmul or single_lora_mask'
```

这些 UT 大量使用 mock/CPU tensor，证明参数和 Python 分派，不证明 AscendC/Triton 数值或 NPU 性能。
对 MoE 索引/量化边界改动定向运行：

```bash
"$LORA_PY" -m pytest -q tests/ut/lora/test_lora.py -k 'routing or indices or empty_ep_rank'
"$LORA_PY" -m pytest -q tests/ut/lora/test_quant_moe.py
```

真实单卡算子验证：

```bash
"$LORA_PY" -m pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_bgmv_shrink.py
"$LORA_PY" -m pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_bgmv_expand.py
"$LORA_PY" -m pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_lora_linear.py
```

先阅读实际断言和参数化，再判断是否覆盖本次变更：

| 用例 | 当前覆盖及不足 |
| --- | --- |
| `test_lora_ops.py::test_bgmv_expand_uses_full_output_width_and_drops_add_inputs` | mock 验证 add_inputs 没有传给 C op，不能当 overwrite 正确性 |
| `test_lora.py::test_packed_single_lora_matmul` | 非零初始 y、混合 mask、add/overwrite 数学结果；CPU 计算而非 graph |
| `test_punica_npu.py::test_add_lora_fused_moe_builds_graph_safe_combined_index` | combined slot/expert index 及接口；不证明真实图捕获成功 |
| `test_bgmv_expand.py::test_bgmv_expand` | 单 batch、rank16、FP16 weights、FP32 shrink、slice0；未覆盖所有 rank/offset/inactive |
| `test_lora_linear.py::test_lora_linear_triton_accuracy` | 12 个组合，直接 launch Triton；不经过生产 gate |

Triton 的 12 个 case 是 rank `{8,16,32,64}` 与以下三个**配对**的笛卡尔积：

- token=1，O projection `(hidden=1536, outputs=(5120,))`。
- token=31，QKV `(hidden=5120, outputs=(1024,512,512))`。
- token=128，QKVZ `(hidden=5120, outputs=(1536,512,512,1536))`。

它不是全部 token×projection 的全组合。该测试 relative L2 要求小于 0.005，并要求 inactive 行与初始 y 精确一致。
这个阈值是现有 BF16 测试选择，不是新 kernel、dtype 或 adapter delta 的通用阈值。
测试 helper `_launch_lora_kernels` 每次创建 workspace；不要直接复用它循环计时并声称测得 kernel-only latency。
按改动补 threshold 两侧、slice 偏移、inactive、slot reset、异步 metadata 和连续 graph replay 场景。

## 微基准：先选择测量边界

分别定义并报告三类成本，避免一个数字混用：

1. raw kernel：预分配 workspace，给定已打包 A/B 和 token mask；只测 shrink+expand。
2. 生产 dispatcher/layer：包括实际选择的 matmul 或 Triton、metadata 更新、必要 packing/通信，注明哪些只在加载时发生。
3. serving：使用相同请求分布、base/adapter 比例、rank 配置、TP/EP 和 graph 模式，比较端到端收益。

已有 `vllm/benchmarks/kernels/benchmark_lora.py` 是上游 GPU 基准入口，不能无检查地在 NPU 执行。
可参考 Ascend `benchmarks/quantize_indexer_query.py` 的 NPU event 用法，但其 workload 不是 LoRA，不引用它的性能数字。

## 正确计时步骤

- 用 fixed seed 预生成 x、A、B、初始 y 和 adapter mapping，候选/reference 共用相同输入。
- 先完成模型加载/packing/JIT/warmup/capture 与 workspace 分配，确认 warmup 路径和待计时路径一致。
- raw Triton launch 与 custom-op 路由不同；后者 eager 会回 matmul，不能混名比较。
- 单流使用同一 stream 上的 NPU event；记录 start、launch、end，再 `end.synchronize()` 后取 `start.elapsed_time(end)`。
- 如果测 wall-clock，开始和结束都 synchronize；明确包括 host dispatch。异步多流则由结束事件依赖全部相关流，或使用同步包围的端到端时钟。
- 预热外单独记录首次编译/捕获耗时；steady-state 多轮采样报告中位数与分位数，不拿一轮最小值定论。
- 设备调频/并发其它任务会影响结果，交替执行 baseline/candidate 并记录硬件环境。

in-place residual 特别注意：

- 每次测试前在同一 stream `y.copy_(y_initial)`，reset 放在 start event 之前，使 restore 不进入 kernel-only 区间。
- graph 必须保留捕获的 y storage 地址，不能 `y = y_initial.clone()` 后重放旧 graph。
- 若完整层的 base matmul 本就每轮覆盖 y，计时完整层时无需另加人工 reset；注明所测边界。
- 一次测多个 launch 时，可准备多个输出 buffer，在开始前恢复并保证每个 launch 使用正确 buffer；图固定地址时不能随意替换。
- 长循环直接累加同一个 y 会改变数值甚至溢出，不是可靠的重复执行方式。
- restore 会改变 cache 工作集，记录这一测量约定；不要未经验证通过相减两次独立 benchmark 的时间推导 kernel latency。

正确性采样在计时区间外执行，比较每条路径独立的 shrink/delta/y 与原始 reference。
tile/split-K 优化还报告 workspace bytes、实际 token 数/物理 padded token 数，以及 raw kernel gain 是否被通信或 launch overhead 抵消。
没有执行 NPU 测量时，不输出估计加速倍数；给出待测 shape、命令、计时边界和判定标准。
