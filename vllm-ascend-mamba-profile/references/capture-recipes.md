# 采集与基准模板

以下为 Bash 模板，运行位置是已配置 NPU 环境的 Linux 主机或容器。
先确认该环境安装的 `vllm bench serve --help` 与本工作树参数一致。
执行 Python 时遵循所在仓库的环境要求；vLLM 仓库使用 uv 管理的 `.venv/bin/python`。
模板不安装依赖、不下载权重、不替代用户已经工作的多卡/投机启动配置。

## 1. 为已工作的服务增加 profiler

在原 `vllm serve` 命令后增加以下配置；输出目录选择本次实验独立、worker 可写的路径：

```bash
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/mamba-profile-run-001","torch_profiler_with_stack":false,"torch_profiler_with_memory":false}'
```

这是一段追加参数，不是单独的 shell 命令。保留原模型、投机、TP/PP、graph、cache 等参数。
启动日志必须确认真实加载的是 vllm-ascend；`torch` 配置名称本身不能证明 NPU worker 已生效。
`/tmp/mamba-profile-run-001` 是示例目录，每组试验使用新名称，分布式时按主机归档。

## 2. 先做 prefill 无采集基准

设置 `PROFILE_MODEL` 为服务接受的模型名；服务地址按实际部署调整：

```bash
PROFILE_MODEL=/models/your-tested-model
PROFILE_BASE_URL=http://127.0.0.1:8080
vllm bench serve \
  --backend vllm --base-url "$PROFILE_BASE_URL" \
  --model "$PROFILE_MODEL" \
  --dataset-name random --seed 0 \
  --random-input-len 4096 --random-output-len 1 --random-range-ratio 0 \
  --num-prompts 32 --num-warmups 4 \
  --request-rate inf --max-concurrency 1 --ignore-eos \
  --percentile-metrics ttft,e2el --metric-percentiles 50,90,99 \
  --save-result --save-detailed --result-dir ./profile-results --result-filename prefill-baseline.json
```

32 个请求是短实验示例，不能支撑稳定的 p99 尾延迟结论。
确定趋势后用足够样本重复无采集基准；比较中位数和波动，同时保留错误数。
`--request-rate inf` 与 `--max-concurrency 1` 在此表示最多一个进行中的请求。
random 数据适合形状隔离；评估投机 acceptance 要增加真实业务文本，随机 token 可能改变其难度。
输出 1 token 只减少 decode 干扰，TTFT 仍不是纯 NPU prefill 耗时。

prefix caching 已开启时，重复 seed 或 warmup 请求可能命中缓存；
比较冷缓存时使用一致的缓存策略或独立服务运行，别把命中提升误当成 kernel 加速。
评估缓存本身时保留它，另报告真实计算 token 数和缓存命中。

## 3. 小窗口采 trace

先在相同服务上完成暖机，再执行：

```bash
vllm bench serve \
  --backend vllm --base-url "$PROFILE_BASE_URL" \
  --model "$PROFILE_MODEL" \
  --dataset-name random --seed 1 \
  --random-input-len 4096 --random-output-len 1 --random-range-ratio 0 \
  --num-prompts 2 --num-warmups 0 \
  --request-rate inf --max-concurrency 1 --ignore-eos \
  --profile \
  --save-result --save-detailed --result-dir ./profile-results --result-filename prefill-trace.json
```

此处用新 seed 减少重复请求的 cache 干扰；它不是与 seed 0 基准的直接性能对照。
若需要严格同 workload 的开/关采集开销比较，固定 seed 并统一缓存初始状态。
当前 CLI 的 `--ready-check-timeout-sec` 默认 0，以上模板跳过 endpoint check。
仅设为正值时，bench 会在 start profiler 前用第一条测试输入检查服务；这次额外请求不在 trace 中，
也可能影响 prefix cache，但第一条输入仍会在正式 benchmark 中执行。
bench 的 profiler start 失败不一定终止整轮，必须查 `Profiler started/stopped`、
HTTP 状态、服务日志和 trace 实际产物，不能只凭 bench 正常结束判断采集成功。

## 4. 手动控制精确业务窗口

自定义 workload 可用显式 API；下例假定服务已有可用模型并已暖机：

```bash
curl --fail-with-body -X POST "$PROFILE_BASE_URL/start_profile"
```

确认启动成功后发出本轮请求。在请求成功、失败或脚本退出时都安排 stop，
例如请求体保存为工作区内的 `profile-request.json`：

```json
{"model":"/models/your-tested-model","prompt":"请解释递归和循环的区别。","max_tokens":128,"temperature":0}
```

```bash
trap 'curl --fail-with-body -X POST "$PROFILE_BASE_URL/stop_profile"' EXIT
curl --fail-with-body "$PROFILE_BASE_URL/v1/completions" \
  -H 'Content-Type: application/json' --data-binary @profile-request.json
```

以上 trap 应放在独立采集脚本中，退出时触发；交互使用则显式执行 stop。
不要同时让另一个 bench `--profile` 控制同一个 profiler。
停止时 flush 可能较慢，等待日志与目录更新；不要为了结束采集直接杀服务。
没有 start 的成功证据时先诊断，而非继续空采集。

## 5. 分析产物

按实际目录枚举每个 worker 的 `*ascend_pt` 子目录，不把通配符字面值传给 analyse。
若输出尚未解析，在已有环境中使用下列 Python API，对一个真实目录运行一次：

```python
from torch_npu.profiler.profiler import analyse

analyse("/tmp/mamba-profile-run-001/actual_worker_ascend_pt")
```

API 与输出说明来自本地 `service_profiling_guide.md`；具体文件是否存在以实际版本为准。
优先查 `ASCEND_PROFILER_OUTPUT/trace_view.json`，辅以 `kernel_details.csv`、
`operator_details.csv`、`op_statistic.csv`、`step_trace_time.csv`。
若没有这些文件，先检查原始目录、解析日志和采集窗口，不能编造固定列名的解析脚本。

## 6. 延伸到投机 decode 或 chunked prefill

- 投机 decode：将基准输出增加到例如 128 token；用已验证的投机启动参数分别跑关闭/开启组，
  使用真实文本与固定采样配置。增加 `--percentile-metrics ttft,tpot,itl,e2el`，保存新的结果文件。
- chunked prefill：固定长输入、并发与输出，服务使用 `--enable-chunked-prefill`，
  改变 `--max-num-batched-tokens` 作单因素对照；实际 chunk 仍需 scheduler/metadata 证据。
- graph 问题：保留当前 graph 模式的 trace，再用 `--enforce-eager` 做诊断对照。
  不在主实验里同时修改投机长度、cache mode、并行度和 graph。
- 并发问题：单请求复现之后恢复目标 `--max-concurrency`/`--request-rate`，
  检查排队与批形状变化；本 skill 不假定所有 Mamba/投机/并行配置组合都支持。

这些 CLI 的本地定义入口见 [源码索引](source-map.md)。
