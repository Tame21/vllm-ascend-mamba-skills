# LoRA 性能与采集模板

面向已配好依赖、有本地 base 和兼容 adapter 的 Linux Ascend 环境；当前仅提供源码核对后的模板。
先核对安装版本的 `vllm serve --help`、`vllm bench serve --help`，保留用户原模型/TP/graph/量化参数。
不要为执行模板下载模型或重装环境。以下示例模型名和目录必须替换成真实本地路径。

## 服务注册与 benchmark 名称不是同一参数语义

```bash
LORA_BASE_MODEL=/models/base
LORA_ADAPTER_A=/models/adapter-a
LORA_ADAPTER_B=/models/adapter-b
LORA_BENCH_URL=http://127.0.0.1:8080
vllm serve "$LORA_BASE_MODEL" --port 8080 \
  --enable-lora --max-loras 2 --max-cpu-loras 2 --max-lora-rank 16 \
  --lora-modules "adapter-a=$LORA_ADAPTER_A" "adapter-b=$LORA_ADAPTER_B"
```

rank=16 只是示例，必须覆盖真实 adapter rank，并保留记录配置门槛的影响。
此服务是两槽比较组，不会触发 `max_loras=1` fast path。测单槽组时只改相应配置并使用独立结果。
服务器 `--lora-modules` 解析 `name=path` 或 JSON；客户端同名选项接收已注册的 **名字**。
验证 `/v1/models` 及请求响应中的服务名称，避免请求一直落到 base。

## 显式暖目标 adapter

短暖机可先对每个 adapter 发请求，随后用目标输入长度/并发跑一轮正式 workload 并丢弃指标。

```bash
curl --fail-with-body "$LORA_BENCH_URL/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"adapter-a","prompt":"Explain matrix multiplication.","max_tokens":16,"temperature":0}'
```

再对 adapter-b 执行相同请求；该短请求只触及部分 shape，不能保证大 batch/长 prefill 都已暖机。
尤其注意：`bench serve` 的 ready check 和 `--num-warmups` 在 adapter 分配迭代器之前构造
`test_input`，使用 `--model`/`--served-model-name` 指定的模型。仅追加 `--lora-modules`
不会让这段暖机自动轮流请求 adapter。
若 adapter 数大于 NPU slots，“逐个暖过”也不保证同时驻留；热组必须核实最后的 active 状态。

## 无 profiler 的混合稳态基准

```bash
vllm bench serve \
  --backend vllm --base-url "$LORA_BENCH_URL" --model "$LORA_BASE_MODEL" \
  --lora-modules adapter-a adapter-b --lora-assignment round-robin \
  --dataset-name random --seed 0 --random-range-ratio 0 \
  --random-input-len 1024 --random-output-len 128 \
  --num-prompts 64 --num-warmups 0 \
  --request-rate inf --max-concurrency 8 --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --result-dir ./lora-results --result-filename mixed-hot.json
```

先以相同参数执行一轮并使用不同结果文件作为暖机；64 只是示例，稳定 p99 需要更多请求。
`round-robin` 是客户端按请求轮转，服务器实际 batching/adapter 混合仍需观察。
示例的无限到达率与并发上限会造成客户端 semaphore 等待；当前 benchmark 单独记录
`client_queue_time`，结果 JSON 为 `queue_times`。HTTP TTFT/latency 不包含这段等待，
分析排队与端到端体验时需一起报告，不能把 TTFT 当作从计划到达起的全部耗时。
纯 base 对照去掉客户端 `--lora-modules`；热单 adapter 只填 `adapter-a`；
服务关闭 LoRA 的 base 组另起相同部署，仅去掉服务 LoRA 配置。不要把这四种变化合在同一组。
prefill 初查用输出 1 token、并发 1；decode 分析保留足够输出并看 TPOT，不能把 ITL chunk 数当 token 数。

random 数据用于固定形状；业务文本、chat template、真实长度分布再单独验证。
`--random-range-ratio 0` 是当前固定范围设置；当前有效区间 `[0,1)`，不要使用旧模板的 `1.0`。
重复 seed、同 adapter 和前缀缓存会改变实际计算量；统一 cache 初始状态或明确测 warm prefix，
不要把 cache hit 造成的 TTFT 下降误称 LoRA kernel 加速。

## 单 adapter 命名与离线多 ID 的边界

只压一个已注册 adapter，也可用 `--model "$LORA_BASE_MODEL" --served-model-name adapter-a`，
这时 benchmark 的 endpoint check/暖机也请求 adapter-a；tokenizer 仍以本地 base 模型解析。
使用客户端 `--lora-modules` 时，每个主请求的 model ID/name 会被 adapter 名覆盖。

`vllm bench throughput --lora-path` 属于离线入口；本地 `throughput.py:assign_loras`
通过 `BenchmarkDataset.get_lora_request` 给请求分配 ID，**多个 ID 指向同一个 lora_path**。
它可用于控制 slot 数的合成压力，不能代表多个不同权重、rank、target modules 的真实分布：

```bash
vllm bench throughput \
  --model "$LORA_BASE_MODEL" --enable-lora --max-loras 2 --max-lora-rank 16 \
  --lora-path "$LORA_ADAPTER_A" --lora-assignment round-robin \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --num-prompts 32 --seed 0
```

离线结果不提供相同的在线排队/网络 TTFT 语义；多个不同 adapter 应使用在线注册的名字或
已核实的显式 `LoRARequest` workload，而不是给 `--lora-path` 拼路径列表。

## CPU 冷加载与 slot churn

使用本地已注册的真实 adapter 集合和固定请求序列，分别设计：

- 所有 adapter ≤max_loras 且已激活：热计算。
- 集合大于 max_loras 但不超过 max_cpu_loras：CPU resident、NPU slot 激活/淘汰。
- 集合大于 max_cpu_loras：同时可能发生 CPU adapter 重新加载；确认 `_load_adapter` 证据。

静态模块在服务初始化时已加载，不能把第一条请求 TTFT 自动称为冷磁盘加载。
本组保存首次/切换请求的单条时延、LRU 状态、activation 和 H2D/repack 时间，
与稳定窗口分开汇报。需要动态 load API 才继续核对对应版本路由与启用条件，
不为普通性能基准擅自打开动态加载开关。

## 加 NPU profiling

在已工作的服务命令后追加下列参数，使用新的、worker 可写的目录：

```bash
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/lora-profile-001","torch_profiler_with_stack":false,"torch_profiler_with_memory":false}'
```

这是追加参数片段。先完成热组暖机，再给对应 bench 增加 `--profile`、减少到 2–4 个请求，
保存新的结果文件；该轮用于归因，不作为无扰动性能结果。
冷组则在待观察的切换/重新加载前开始采集，不能预先消除 load/repack。
也可显式 `POST /start_profile` → 业务请求 → `POST /stop_profile`；异常退出也安排 stop，等待 flush。
API 200 不保证 worker profiler 成功启动，检查日志中的失败 warning 和真实产物。

Ascend wrapper 固定 CPU+NPU、Text/Level1；使用 `torch` 类型而不是 CUDA/Proton。
`torch_profiler_with_stack` 映射 `with_modules`；memory 有透传，shapes/FLOPs/gzip/activities
与 wait/warmup/active schedule 未接入同样行为。delay/max iterations 为 worker step；
与 `additional_config.msmonitor_use_daemon` 互斥。配置缺失时 start/stop 路由可能 404。

逐 rank 找 `*ascend_pt`，优先时间线而非单张算子累计表。必要时在已有 Python 环境执行：

```python
from torch_npu.profiler.profiler import analyse

analyse("/tmp/lora-profile-001/actual_worker_ascend_pt")
```

用真实目录替换示例，不能直接把 `*` 传进去。检查实际生成的 `ASCEND_PROFILER_OUTPUT/trace_view.json`、
kernel/operator 统计与解析日志；不假设不同版本 CSV 都有相同列名。

## 本地参数证据

- `vllm/vllm/engine/arg_utils.py`、`config/lora.py`：enable/rank/slots/CPU slots/fully-sharded。
- `vllm/vllm/entrypoints/launchers/cli_args.py:LoRAParserAction`：服务端 name/path 注册。
- `vllm/vllm/benchmarks/serve.py:add_cli_args/benchmark`：base-url、served-model-name、adapter 名称、assignment、暖机、保存结果及 profile。
- `vllm/vllm/benchmarks/datasets/datasets.py`：random 长度、seed、num-prompts 与离线 adapter ID 分配。
- `vllm/vllm/benchmarks/throughput.py:assign_loras/add_cli_args`：离线 lora-path 与 assignment。
- `vllm-ascend/vllm_ascend/profiler/torch_npu_profiler.py`、`worker/worker.py:profile`：NPU wrapper。
- `vllm/vllm/entrypoints/serve/profile/api_router.py:attach_router`：服务 profile API 注册。
- `vllm-ascend/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md`：analyse 与产物说明。
