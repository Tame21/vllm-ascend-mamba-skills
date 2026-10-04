# 指标口径与性能判读

## 1. 时间口径

客户端 TTFT 为请求到第一段有效输出的时间，包含服务排队、调度、输入处理、
模型 prefill 和首 token 生成等；先确认使用支持流式响应的 endpoint。
`vllm/vllm/benchmarks/serve.py:calculate_metrics` 在输出大于一个 token 时计算：

```text
TPOT = (request_latency - TTFT) / (output_tokens - 1)
output_throughput = total_actual_output_tokens / benchmark_duration
```

ITL 来自流式响应的到达间隔；投机解码可能一次返回多 token，网络分块与服务输出方式
会影响它。结合 TPOT、真实输出 token 数和 NPU 时间线，不把每个流式 chunk 当一个模型步。
输出 1 token 时缺少可用 decode TPOT 样本，不能把零值解释为无限快。

端到端时间、CPU 包裹时间、单 stream kernel 时间、跨 stream/rank 总算子时间是不同指标。
有重叠时总和可以大于请求时延。依据时间线判断谁阻塞关键路径，随后再做算子优化。

## 2. 接受率口径

取相同时间窗口、相同实例/模型的 Prometheus counter 增量：

```text
R = Δvllm:spec_decode_num_accepted_tokens_total
    / Δvllm:spec_decode_num_draft_tokens_total
L = 1 + Δvllm:spec_decode_num_accepted_tokens_total
        / Δvllm:spec_decode_num_drafts_total
P[i] = Δvllm:spec_decode_num_accepted_tokens_per_pos_total{position="i", ...}
       / Δvllm:spec_decode_num_drafts_total
```

分母为零时记为“不适用”。`L` 按上游约定包括一个 bonus token；`R` 不包括该 bonus。
`P[i]` 是按所有 draft 轮次归一化，不是条件概率 `P(接受第 i 位 | 接受前 i-1 位)`；
各轮 draft 长度变化时尤其不能将尾部偏低直接判断为 draft 模型质量差。

当前 `vllm bench serve` 自动抓取 metrics 前后差，JSON 中：

- `spec_decode_acceptance_rate` 使用百分数（0–100）。
- `spec_decode_acceptance_length` 使用上述 `L`。
- `spec_decode_per_position_acceptance_rates` 使用比例（0–1）。
- `spec_decode_num_drafts`、`spec_decode_draft_tokens`、`spec_decode_accepted_tokens` 是增量计数。

bench 会跨找到的相关 label 累加；共享服务其他请求、计数器 reset、metrics 不可达、
抓取延迟都可能破坏解释。独占实验窗口最直接；无法独占时保留原始 metrics 并按 label 筛选。
不要把 MRV1 `num_accepted_tokens.gpu = (output_token_ids != -1).sum(...)`
这一“实际输出 token 数”直接等同于 Prometheus accepted draft counter。

接受率相近而速度变慢，先查 target verification 的 token 量、drafter 成本、
状态维护、host/device 同步、padding 和 graph 覆盖率；不要只继续增大 draft 长度。
`每轮耗时 / 每轮有效输出 token 数` 可辅助分析，但异步重叠场景应以时间线和端到端基准为准。

## 3. 从时间线到源码

| 时间线现象 | 可区分原因的下一项检查 |
| --- | --- |
| NPU 空洞前有 CPU 长段 | 查输入构建、metadata、accepted count 同步、数据准备；验证 CPU 段是否真的阻塞设备 |
| 多个 memcpy/同步围住 state 操作 | 对照 Mamba cache mode、batch 重排、slot 映射和实际 copy 路径 |
| 长 prefill 分成大量短步 | 查实际 scheduled tokens、chunk budget、并发 decode、prefix hit 和模型对齐要求 |
| prefill kernel 名称变化 | 查 GDN fused 探测、硬件能力、PCP 与 fallback；比较实际 shape/初始状态 |
| kernel 时间正常但 TTFT 高 | 分离排队、网络、tokenization、调度和缓存因素 |
| 高 acceptance 仍无加速 | 分解 draft、target、多 token 验证、rejection 和 state copy 成本 |
| eager 正常而 graph 慢 | 查真实 graph dispatch、padding token 数与 replay 形状；不能只改模型逻辑 |
| trace 里只有 CPU 活动 | 确认打开的是 API CPU trace 还是 NPU worker `ascend_pt`，再查采集/解析失败 |

MRV1 的 `prepare input` / `forward` / `post process` / `sample_token` / `draft_token`
等范围由 `vllm/vllm/v1/utils.py:record_function_or_nullcontext` 包裹。
`forward` 覆盖 target prefill、普通 decode 和 speculative verification，内部还可能执行
Mamba state copy；必须结合 scheduler/metadata 和内部事件拆分，不能把整个 scope 记为 verify。
`vllm/vllm/envs.py` 中 `VLLM_CUSTOM_SCOPES_FOR_PROFILING` 默认是 0；
需要这些阶段标记时在启动服务前设置 `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`，
它选择 `record_function`，但仍要确认实际 NPU trace 显示这些事件。
该函数缓存首次选择的实现，所以运行后才改环境变量不能可靠生效。
将它作为诊断配置记录，并保留关闭该开关的最终性能基准。
需要补标记时优先在已有低频阶段边界加临时 annotation，避免逐 token/逐层日志形成热点。

`AscendGatedDeltaNetAttention._forward_core` 的 prefill 路径包含初始 state 获取/清理、
conv 和 gating、chunk 计算以及 final state 写回。比较不同实现时必须涵盖这些成本，
不能只比较内部 chunk kernel。一 token 的有状态 prompt 尾块可被归入 decode 路径。

## 4. 最小实验矩阵

从实际问题选必要行，不强制跑所有组合：

| 组 | 控制变量 | 验证问题 |
| --- | --- | --- |
| A | 当前服务，无采集 | 基准是否稳定、完成率是否正常 |
| B | 同 workload，小窗口采集 | 热点位于 framework、kernel 还是 state copy |
| C | 仅关闭投机 | 投机总成本是否超过有效输出增益 |
| D | 仅改变投机 token 数 | acceptance 和每轮成本的折中 |
| E | 仅改变 chunk token budget | prefill 吞吐与 decode 干扰的折中 |
| F | 仅切换 eager 诊断 | graph/padding 是否相关 |

合适的 workload 应包括真实问题的 input/output 长度、并发和请求混合。
正确性尚未确认或配置不支持时先停止相应对照，不通过吞吐数字掩盖错误、丢请求或提前 EOS。
最后用不带额外同步、dump、profiling 的服务重新验证变更。
