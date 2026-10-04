---
name: vllm-ascend-lora-profile
description: 根据本地 vLLM 与 vLLM Ascend 的 LoRA 实现，设计性能对照、采集 NPU profiling 并定位单适配器、多适配器、rank、TP、graph 或加载淘汰引起的 TTFT/TPOT/吞吐回归。适用于 LoRA 性能分析与优化验证；保留精度检查，不把降低 adapter 计算量造成的输出变化当作性能收益。
---

# LoRA 性能采集与归因

把服务开销分成 adapter 加载/激活、请求调度、映射更新和实际 LoRA 计算，再确定热点。
先证明测到的是预期 adapter 与路径，不凭 `enable_lora=True`、kernel 名称或吞吐数字下结论。

## 固定基线

定位包含 `vllm/`、`vllm-ascend/` 的工作区；若分别存放，确定两个根路径并读适用的 `AGENTS.md`。
本技能依据 vLLM `d61081dc3d3f1740a5d8bf82608b62974393c2de` 与
Ascend `9b8fc5d728e1ea54dc277562b296112d905e15c1` 附近工作树，当前 HEAD 和未提交修改仍需核对。
记录实际导入包路径、NPU/CANN/torch/torch_npu/triton 版本、runner、graph、并行与调度配置。

另记录 base/tokenizer revision、各 adapter revision/路径、真实 rank、target modules、alpha/scale、
LoRA dtype、量化、`max_lora_rank`、`max_loras`、`max_cpu_loras` 和 `fully_sharded_loras`。
配置的 rank 上限、权重真实 rank、TP 后本地 rank、packed total rank 是不同量。
优先沿用已工作的部署命令；模型、硬件或环境缺失时交付源码结论和待执行方案，不假装实测。

## 先分清四种负载

| 组别 | 服务与请求 | 要回答的问题 |
| --- | --- | --- |
| 无 LoRA / 启用但请求 base | 两组服务配置；都请求 base 模型名 | wrapper、调度、图与预分配的基线成本 |
| 热单 adapter | adapter 已加载且 NPU slot 已激活；持续请求该名字 | 稳态单 adapter 计算成本 |
| 混合 adapter | 多个真实 adapter 名称交错并发，记录每批实际 active 数 | 调度、映射与多 slot kernel 成本 |
| 冷加载 / 淘汰 | adapter 不在 CPU cache，或仅 CPU resident 而不在 NPU slot | 文件读取、权重转换、H2D、repack、LRU 与排队 |

四类数据不能混在同一均值中。`max_loras` 是一批可用 adapter 槽数，不是服务注册总数；
仅列多个名称不能证明同批混合，必须查 scheduler 和 mapping 的实际 active IDs。
冷热程度分别描述 CPU cache、NPU slot、graph、编译/首次 workspace 和 prefix cache。

## 采集顺序

1. 确认 base 请求和 adapter 请求均成功，检查实际名称、应用的目标层及可重复输出。
2. 先测无 profiler 基准：固定实际输入输出 token 数、到达率、并发、采样与 cache 策略。
3. 对目标 adapter 和形状显式暖机，再做热组；冷组保留首次加载，不先暖机消除问题。
4. 采少量代表请求的 CPU+NPU trace，查每个相关 TP/PP rank 的时间线及实际输出产物。
5. 用一个最小对照区分假设；优化后先复核输出，再重跑无 profiler 的性能基准。

命令、暖机与负载分布陷阱见 [采集模板](references/benchmark-recipes.md)。
源码和配置门槛见 [路径与成本](references/paths-and-costs.md)，只读当前问题涉及的部分。

## 先确认实际 linear 路径

`vllm_ascend/lora/punica_npu.py:PunicaWrapperNPU` 与 `lora_ops.py:lora_linear` 决定分支：

- `no_lora` 时普通 `add_lora_linear` 提前返回；base 请求仍可能经历 wrapper/metadata/图相关成本。
- 单 slot 路径要求 A2/A3、配置 `max_loras == 1`、非 fully-sharded，且提供所需 packed weights。
  配置 `max_loras=4` 而本批仅一个 active adapter，不满足这个 fast path。
- 符合单 slot 条件时，graph capture 内可尝试 BF16 小 token 的专用 Triton；
  不满足 dtype、rank、连续性或形状门槛则走 packed `torch.matmul`。
- 其它普通 linear 路径采用 shrink/expand；底层可能是 AscendC SGMV/BGMV 或上游 torch fallback。
  初始化时硬件能力及配置 `max_lora_rank >= 128` 会影响底层选择，但不能据此跳过单 slot 分支判断。
- 调用传入显式 shrink buffer 时会直接走 shrink/expand；embedding、MoE、fully-sharded 单独追踪。

**SGMV 不等于真实 prefill。** 上游 `lora_model_runner_mixin.py:_set_active_loras`
当前固定构造 `LoRAMapping(..., is_prefill=True)`。检查实际调用时 mapping，
不能因 `_shrink_decode` 存在就声称真实 decode 使用 BGMV；310P capture 还有特殊选择。
真实阶段根据 scheduler/query 长度确定，同一 batch 也可混合 prefill/decode。

## 按时间线逐层排除

先区分以下成本，不把整个 `forward` 范围都记成 LoRA：

1. 加载与激活：CPU checkpoint、LRU、`activate_adapter`、逐层 `set_lora/reset_lora`、H2D 和 packed copies。
2. 元数据：request adapter ID → slot、token mapping、prefill 分段、single-slot mask 刷新。
3. 计算：base linear 与 LoRA shrink/expand、mask/scale/add、merged slices 分开对应。
4. 配套数据处理：dtype conversion、gather、`repeat_interleave`、flatten/reshape、workspace 和输出布局。
5. 并行与调度：fully-sharded 的通信、其它 TP collective、adapter 槽上限造成的排队和批大小变化。

`view`/`transpose` 本身未必搬数据；用真实 stride、连续性及 trace 判断，
不要仅从名字推断发生复制。`set_lora` 中的 packed weight copy 通常属于激活成本，
若热轮反复出现，先查 slot churn/重新加载，不先优化每 token matmul。
cached FP32 shrink workspace 和 Triton workspace 可能使首次运行更慢；保留首次与稳态区分。

## rank、TP 与 graph 对照

保持 adapter 权重与应用层不变，先比较必要的配置差异；减少 target modules 会改变模型行为。
提高 `max_lora_rank` 不只是扩大准入上限，也可能改变 storage shape、kernel 路由和内存。
`fully_sharded_loras` 改变 layer replacement 和通信模式，并排除当前单 slot fast path；
不能把上游“可能更快”的配置说明当作当前 NPU、rank、TP 组合的性能保证。

先保留真实 graph 配置，再用 eager 做隔离诊断。单 slot Triton 在 capture 时选定并进入图；
replay 不要求每次重新进入 Python 选择函数。因此 eager 与 graph 的 kernel 不同可能是预期行为。
核对 capture cases、实际/补齐 token、active adapter 数及 base/LoRA 图分支，
不能仅因服务启动成功或一个 batch replay 正确就断言所有 adapter 混合都覆盖。

## NPU profiling 配置

使用 `--profiler-config` 的 `profiler: "torch"` 与独立 worker 可写的目录；
Ascend wrapper 只接受此 profiler，固定 CPU+NPU，不移植 CUDA/Proton 参数。
初查关闭 `torch_profiler_with_stack` 与 memory；前者在该 wrapper 映射 `with_modules`，
底层 `with_stack` 固定 false。shapes/FLOPs/gzip/activities 和 wait/warmup/active schedule 未完整透传。
delay/max iterations 属于 worker 步数，不是请求数；与 `msmonitor_use_daemon` 同开会报错。
`/start_profile`、`/stop_profile` 仅在 profiler 已配置时注册，停止后等待 flush 并核对 worker 日志。

不在正式热路径加入 `.item()`、`.cpu()`、逐层 print 或 `synchronize()` 来计时。
仅在隔离算子实验的测量边界同步；设备异步提交的 CPU 墙钟不等于 kernel 执行耗时。
需要 MRV1 阶段标记时，在启动前启用源码已有的 `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`，
记录额外扰动，最终基准恢复关闭。

## 结论与交付

给出每组请求分布、冷热定义、实际路由、错误数、TTFT/TPOT/ITL/输出吞吐和内存证据。
输出 1 token 的组适合首 token 分析，不产生有意义的 decode TPOT；投机输出按真实 token 数比较。
当前 benchmark 的 TPOT 为 `(latency - ttft) / (actual_output_tokens - 1)`；ITL 是流式 chunk 间隔，
投机场景不必对应逐 token 间隔。并发限制前的客户端等待另存 `queue_times`，不包含在 HTTP TTFT/latency 中。
小样本 trace 不能支撑 p99 结论，多 rank/stream 的 kernel 时间也不能简单相加成为请求时延。
交付命令、版本、结果/trace 路径、确认的瓶颈及一个验证过或待执行的最小实验。
明确源码核对、CPU/mock 和真实 NPU 执行的边界；没有 trace 或硬件结果时不编造加速比。
