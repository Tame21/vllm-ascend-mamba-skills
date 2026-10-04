---
name: vllm-ascend-mamba-profile
description: 基于本地 vLLM 与 vLLM Ascend 代码，为 Mamba、GDN 等混合模型的投机推理采集和分析 NPU profiling，定位 prefill、chunked prefill、TTFT、decode 与 acceptance 性能问题。适用于“采 profiling/采 prefilling”、trace 分析和性能回归；不把数值精度验证或 CUDA profiler 配置当作 NPU 性能结论。
---

# Mamba 投机推理性能采集与归因

把“哪个请求慢”连接到 scheduler、runner、Mamba 状态管理和实际 NPU 算子。
用户说“采 prefilling”时，结合上下文同时考虑“采 profiling”和“分析 prefill”；
先给出 prefill 采集方案，有明确歧义且影响工作时再询问。

## 工作区与证据

约定工作区根目录下有 `vllm/` 和 `vllm-ascend/`；先定位真实路径及各自 `AGENTS.md`。
本 skill 的源码依据为 vLLM `d61081dc3d3f1740a5d8bf82608b62974393c2de`、
vLLM Ascend `9b8fc5d728e1ea54dc277562b296112d905e15c1` 附近的工作树。
这些是索引基线，不能代替当前 HEAD、未提交 diff 和运行环境的实际版本。
查看 [源码入口与配置差异](references/source-map.md) 以检查版本漂移。

记录模型与 tokenizer、target/draft revision、投机方法和 token 数、dtype/量化、
CANN/torch/torch_npu/fla_npu、NPU 型号、TP/PP/DP/PCP、runner 版本、graph 模式、
async scheduling、prefix caching、Mamba cache mode、chunk token budget。
尽量复用已工作的服务命令，保留用户的模型和部署选择。
本地缺少 NPU、权重或对应环境时，完成命令与证据分析，明确标记“待 NPU 执行”。

## 先决定测量对象

| 症状或问题 | 第一个有用实验 | 需要防止的误读 |
| --- | --- | --- |
| 长输入首 token 慢 | 并发 1、长输入、输出 1 token | TTFT 还含排队、传输、首 token 采样 |
| chunked prefill 回归 | 同一输入，比较实际 chunk 大小与每步耗时 | CLI token budget 不等于实际 chunk |
| 开投机后吞吐下降 | 固定输入输出长度，对比关闭/开启投机 | acceptance 高不代表整体加速 |
| decode 偶发长尾 | 保留线上批形状的小规模复现 | 单请求可能消除 batch 重排与状态搬运 |
| graph/eager 性能差 | 同一 workload 分别采集 | eager 结论不能替代 graph 运行结果 |

不要第一步开启所有 profiling、dump 和同步开关。
做两类独立运行：未开启采集的基准用于量化收益；少量请求的 trace 用于归因。
保持同样的暖机、输入、并发、随机种子和采样参数；每次只改变待验证因素。

## 采集步骤

1. 检查完整启动日志中的实际配置与后端，确认服务已完成模型加载、编译和 graph capture。
2. 用目标输入长度和批形状暖机；`vllm bench serve --num-warmups` 会重复它的测试请求，
   不能保证覆盖所有生产形状。冷启动问题需要单独一组数据。
3. 先运行无 `--profile` 的基准，保存结果 JSON、错误数、实际输入输出 token 数。
4. 为服务设置 `--profiler-config`，选择 `profiler: "torch"` 和独立的本地输出目录。
   低扰动初查使用 `torch_profiler_with_stack: false`、`torch_profiler_with_memory: false`。
5. 选择一种窗口控制：bench 的 `--profile`，或显式 `/start_profile` → 请求 → `/stop_profile`。
   保证异常退出后也尝试停止，等待 flush 完成；查服务日志及实际产物确认成功。
6. 检查每个相关 worker/rank 的 `*ascend_pt` 目录；必要时用 `torch_npu.profiler.profiler.analyse`。
   先看 CPU/NPU 时间线及关键路径，再查算子统计。
7. 提出一个有证据的瓶颈假设，做最小对照；改动后重新跑无采集基准验证收益。

可直接适配的命令见 [采集与基准模板](references/capture-recipes.md)。
这些模板面向已配置好依赖的 Linux Ascend 环境，不是当前 Windows 源码目录的运行证明。

## Ascend profiler 的真实边界

以 `vllm_ascend/profiler/torch_npu_profiler.py:TorchNPUProfilerWrapper` 为准：

- 仅接受 `profiler="torch"`；不要从上游文档移植 `cuda`、`proton` 或 Nsight 配置。
- worker 固定采集 CPU 和 NPU；`torch_profiler_with_stack` 映射到 `with_modules`，
  `torch_npu` 的 `with_stack` 在此包装器内固定为 `False`。
- memory 配置有透传；shapes、FLOPs、gzip、activities 等不能只因上游接受参数就声称生效。
- `delay_iterations`、`max_iterations` 由上游 `WorkerProfiler` 控制；
  `wait_iterations`、`warmup_iterations`、`active_iterations` 未接入此 NPU wrapper 的 schedule。
- worker 的 `execute_model` 在模型执行前调用 `profiler.step()`；迭代数不是请求数。
  采第一段 prefill 时不要凭猜测设置延迟跳过步数。
- 与 `additional_config.msmonitor_use_daemon` 同开会报错。
- 没有配置 profiler 时 `/start_profile`、`/stop_profile` 路由不注册；404 先查配置和端口。

## prefill 与 Mamba 路径归因

先从 scheduler 实际输出的 `num_scheduled_tokens` 和请求已计算 token 数确认 chunk 边界，
再结合 attention metadata 判断 prefill、decode 和 speculative verification。
V1 scheduler 没有一个可靠的全局“纯 prefill 阶段”；同一 batch 可以混合多个阶段。
GDN 一 token 的有状态 prompt 尾 chunk 也可能使用 decode metadata，不能仅按算子名判断。

对 `vllm_ascend/worker/model_runner_v1.py:NPUModelRunner` 依次归因：

- `prepare input`：metadata、H2D、batch 重排和 `preprocess_mamba`。
- `forward`：target 模型计算；按 scheduler/metadata 区分 prefill、普通 decode 和 speculative
  verification，此 scope 内还可能包含 Mamba state copy，不能整段计为 verify。
  在 GDN 中继续区分 conv、gating、recurrent 与 chunk 路径。
- `sample_token`、`draft_token`：rejection/采样和 drafter 成本。
- 状态更新：accepted token 同步、Mamba state copy 与缓存维护；结合实际调用位置判断是否重叠。

这些是 MRV1 的源码标记，是否出现在 trace 中要检查；MRV2 必须重走它自己的调用链。
GDN prefill 当前可选 fla_npu 融合、torch_npu 融合或 Triton chunk 路径，
选择受硬件能力、PCP 和算子探测结果约束。记录实际分支，不能由安装版本猜测。
其他 Mamba 模型先确认其已注册的实现，不把 GDN 的算子名直接套用到 Mamba2/KDA。

更多归因方法见 [指标口径与性能判读](references/interpretation.md)。

## 投机性能判定

同时报告 TTFT、TPOT/ITL、输出吞吐、请求完成率、acceptance、draft 长度和 trace 中的成本。
输出长度为 1 的 prefill 实验不能用于得出 decode 或 acceptance 收益。
当前 `vllm bench serve` 会对 `/metrics` 做前后差，尝试保存 `spec_decode_*` 结果；
没有该字段表示未拿到有效增量，不能自动解释为接受率为零。
关闭投机对照与开启投机组使用同一 workload；必要时逐个改变 `num_speculative_tokens`。
出现输出/状态异常时先处理正确性，再解释性能差异。

## 避免测量改变问题

CPU 墙钟仅包住异步 NPU 调用，通常只测到提交时间。
仅在隔离算子实验的计时边界同步；不要把逐步 `synchronize()`、设备 `.item()`、
`.cpu()` 或逐 token dump 加入正式性能热路径。Python 对设备标量求布尔值也可能触发同步。
`[tensor.item() for tensor in tensors]` 仍可能逐个同步，不是批量搬运。
若为定位需要临时同步或 `--enforce-eager`，把它标为诊断组，恢复后复测。
不要把多个 stream/rank 的算子时间简单相加当作端到端耗时，也不要只看平均算子表。

## 交付

输出可复现命令、实际版本/配置、无采集对照表、trace/结果文件位置、关键时间线证据，
以及“事实 → 假设 → 最小验证 → 结果”的简短链条。
明确哪些命令已执行、哪些仅源码核对、哪些需要 NPU；没有 trace 时不编造热点或加速比。
如仅得到短窗口的诊断证据，保留不确定性并给出下一项可区分假设的实验。
