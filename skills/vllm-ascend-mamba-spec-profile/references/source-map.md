# 源码入口与版本核对

路径相对于同时包含 `vllm/` 和 `vllm-ascend/` 的工作区根目录。
先运行 `git -C vllm rev-parse HEAD`、`git -C vllm-ascend rev-parse HEAD`，
并查看各自 `git status --short`；如运行环境为安装包，还要核对 import 的实际路径。
本 skill 根据本地源码写成，未声称本机执行过 NPU 采集。

## 服务与 profiler

| 文件与符号 | 用途与实际边界 |
| --- | --- |
| `vllm/vllm/engine/arg_utils.py:EngineArgs.add_cli_args` | `--profiler-config`、chunk token budget、prefix caching、Mamba cache mode 等 CLI 注册 |
| `vllm/vllm/config/profiler.py:ProfilerConfig` | 上游配置能解析哪些字段；不等于插件都实现 |
| `vllm/vllm/entrypoints/serve/profile/api_router.py:attach_router` | 仅 profiler 非空时注册 start/stop 路由 |
| `vllm/vllm/v1/engine/async_llm.py:AsyncLLM.start_profile/stop_profile` | API 进程 CPU profiler 与 engine 调用 |
| `vllm/vllm/v1/engine/core.py:EngineCore.profile` | 到 executor 的 profile RPC |
| `vllm-ascend/vllm_ascend/worker/worker.py:NPUWorker.profile` | 延迟创建 wrapper、rank trace 名、重复 start 使用第一次初始化的 trace 名 |
| `vllm-ascend/vllm_ascend/worker/worker.py:NPUWorker.execute_model` | 调用 runner 前的 `profiler.step()` |
| `vllm-ascend/vllm_ascend/profiler/torch_npu_profiler.py:TorchNPUProfilerWrapper` | CPU+NPU、Text、Level1、PipeUtilization、trace handler 与 start/stop |
| `vllm/vllm/profiler/wrapper.py:WorkerProfiler` | delay/max 计数和启动/停止状态；异常可能仅记 warning |
| `vllm-ascend/tests/ut/profiler/test_torch_npu_profiler.py` | mocked 配置映射、start/stop、无目录及 msmonitor 互斥覆盖；不是硬件采集测试 |

### 配置透传判据

NPU wrapper 直接使用 `torch_profiler_dir`、`torch_profiler_with_memory`，
将 `torch_profiler_with_stack` 用作 `with_modules`。
它不向 `torch_npu.profiler.profile` 传入 `record_shapes`、`with_flops`、
用户选择的 activities 或上游 wait/warmup/active schedule，
也不向 NPU trace handler 传入 `torch_profiler_use_gzip`。
其 `_profiler_step()` 只返回 `True`，没有调用底层 profiler 的 `step()`。
因此上游可解析配置、API CPU trace 和 NPU worker trace 的行为必须分开验证。

`WorkerProfiler.step()` 在 `_profiling_for_iters > _max_iters` 时停止，
结合当前 worker 的调用位置解释采集窗口，不用“请求数 × token 数”估算精确边界。
若 API 返回 200，仍检查 worker 的 `Failed to start profiler` / `Failed to stop profiler`
日志以及非空产物，因为包装器内部会捕获一部分异常。

## benchmark 与 metrics

- `vllm/vllm/benchmarks/serve.py:add_cli_args`：`--profile`、`--num-warmups`、
  `--max-concurrency`、`--request-rate`、保存文件和 percentile 配置。
- `vllm/vllm/benchmarks/datasets/datasets.py`：dataset、seed、num-prompts 和 random 长度参数。
  当前 `--random-range-ratio 0` 表示不作长度范围扰动，允许范围为 `[0, 1)`；不要照搬旧示例的 `1.0`。
- `vllm/vllm/benchmarks/serve.py:benchmark`：CLI 的 `--ready-check-timeout-sec` 默认 0，
  仅正值时先做 endpoint check；配置了 warmup 则在 start profiler 前执行 warmup。
  benchmark 完成后取 metrics 差值，再 stop profiler。
- `vllm/vllm/benchmarks/serve.py:fetch_spec_decode_metrics`：读取 `/metrics`，累加找到的 `_total` counters。
  该函数不按 model/engine label 精细隔离，共享服务上的其他流量可能污染差值。
- `vllm/vllm/benchmarks/serve.py:calculate_metrics`：客户端 TTFT、TPOT、ITL、吞吐口径。
- `vllm/vllm/v1/spec_decode/metrics.py:SpecDecodingStats/SpecDecodingLogging/SpecDecodingProm`：
  accepted 与 draft 计数、mean acceptance length、Prometheus 名称。

## Mamba 与 prefill

| 文件与符号 | 适合追踪的成本 |
| --- | --- |
| `vllm/vllm/v1/core/sched/scheduler.py:Scheduler.schedule` | 实际 token budget、已计算 token、投机 token 和混合调度 |
| `vllm-ascend/vllm_ascend/worker/model_runner_v1.py:NPUModelRunner.execute_model/sample_tokens` | 输入准备、forward、sample、draft 与异步状态更新 |
| 同文件 `_sync_num_accepted_tokens`、`_update_states_after_model_execute` | CPU/NPU accepted count、状态维护；trace 需跟实际调用者关联 |
| `vllm/vllm/v1/worker/mamba_utils.py` | preprocess/postprocess 的 cache mode 分支 |
| `vllm-ascend/vllm_ascend/ops/gdn_attn_builder.py:AscendGDNAttentionMetadataBuilder` | spec/non-spec 拆分、chunk metadata、padding 与单 token prompt 分类 |
| `vllm-ascend/vllm_ascend/ops/gdn.py:AscendGatedDeltaNetAttention._forward_core` | conv、fused gating、spec/non-spec recurrent、prefill 三种分支、state gather/scatter |
| 同文件 `_probe_fused_chunk` | 算子接口存在不代表可用；实际 smoke call 结果缓存于进程 |
| `vllm-ascend/vllm_ascend/worker/v2/model_runner.py` | MRV2 入口；不能直接假设 MRV1 标记/调用序列适用 |
| `vllm-ascend/vllm_ascend/ops/triton/v2/mamba/precopy.py` | MRV2 state precopy 路径 |

`NPUWorker.profile_prefill_latency` 为动态 chunk sizing 的同步 eager `_dummy_run`
计时，临时设 `max_num_reqs=1`，以毫秒返回。它不是在线 TTFT，也不是一个 HTTP prefill
profiling API。`vllm_ascend/core/scheduler_profiling_chunk.py:ProfilingChunkScheduler`
及 `core/profiling_chunk_predictor.py` 用这些样本做预测；仅当当前部署使用它们时纳入分析。

本地使用说明：`vllm-ascend/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md`。
普通 PD 代理不转发 start/stop 时要直连各实例；EPD 或自定义代理应检查自身路由实现。
MS Service Profiler 可作为框架级补充，先验证安装版本和符号表；不为一次 kernel 排查默认安装整套工具。
