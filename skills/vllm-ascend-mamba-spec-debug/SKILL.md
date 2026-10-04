---
name: vllm-ascend-mamba-spec-debug
description: 结合本地 vLLM 与 vLLM Ascend 源码排查 Mamba、GDN 混合模型的投机推理异常，包括启动与 patch 未生效、runner 分支、提议与验证、reject 后错误、动态图与 ACL Graph 差异。适用于需要最小复现、跨仓调用链和定向测试的开发排障。
---

# Mamba 类投机推理排障

将故障缩小到一个请求、一个 step 和一个明确的代码边界；输出可重放输入、实际执行路径、首个异常观测及验证结果。
本 skill 中的“Mamba 类”包括状态缓存层面的共性，不表示 Mamba2 与 GDN 使用相同算子。

## 先确定当前代码和运行实例

- 找到项目根目录中的 `vllm/` 与 `vllm-ascend/`，阅读各自适用的 `AGENTS.md`。
- 下文路径都相对于项目根目录；如果只有一个 checkout，通过已安装包的 `__file__` 找另一个来源。
- 本地图对应 vLLM `d61081dc3d3f1740a5d8bf82608b62974393c2de`、Ascend `9b8fc5d728e1ea54dc277562b296112d905e15c1`。
- 每次使用记录两边 HEAD、dirty 状态与 Python 导入路径。版本不同先重新 `rg` 符号和签名，不将这份快照当兼容性承诺。
- 记录模型 architecture、模型与 tokenizer revision、dtype/量化、硬件型号、CANN、torch/torch_npu、Triton Ascend 版本。
- 同时记录最终解析的 speculative config、runner、target/draft 图模式、TP/PP/DP/PCP/DCP、async、chunked prefill、prefix cache、Mamba cache mode 和 block size。
- 运行命令和短复现见 [references/reproduce-and-probe.md](references/reproduce-and-probe.md)。仅源码环境先做静态核对，明确哪些结论需要 NPU。

## 按真实对象选路径

| 边界 | 当前快照中的入口 | 检查目的 |
| --- | --- | --- |
| Runner 选择 | `vllm-ascend/vllm_ascend/worker/worker.py`，`NPUWorker.init_device` 内创建 runner 的分支 | 记录实际 runner 类型；不要把 V1 engine 与 V1 runner 当同一开关 |
| Runner 门控 | `vllm-ascend/vllm_ascend/patch/platform/patch_use_v2_model_runner.py::_patched_use_v2_model_runner` | Ascend 当前快照显式读 `VLLM_USE_V2_MODEL_RUNNER`，未设置时 False |
| V1 runner | `vllm-ascend/vllm_ascend/worker/model_runner_v1.py::NPUModelRunner` | target 执行、`_sample`、`propose_draft_token_ids` 与状态更新 |
| V2 runner | `vllm-ascend/vllm_ascend/worker/v2/model_runner.py::NPUModelRunner` | 同时查看其父类 `vllm/vllm/v1/worker/gpu/model_runner.py::GPUModelRunner` |
| V2 drafter | `vllm-ascend/vllm_ascend/worker/v2/spec_decode/__init__.py::init_speculator` | 实际 method 分派；存在 `NotImplementedError` 的分支不能靠改标签绕过 |
| V2 model state | `vllm-ascend/vllm_ascend/worker/v2/model_states/__init__.py::init_asecnd_model_state` | 原代码拼写如此；模型自定义 state 优先，其次 hybrid/310P 分支 |
| GDN 层替换 | `vllm-ascend/vllm_ascend/utils.py::register_ascend_customop` | 检查 `GatedDeltaNetAttention` 实际注册到标准 NPU 还是 310P 实现 |

不要从文件名推断继承后的运行行为：记录 `type(obj).__mro__` 与关键 bound method 的来源。
例如 `AscendMTPSpeculator(AscendAutoRegressiveSpeculator, MTPSpeculator)` 本身没有实现主体；要沿 MRO 查看 `propose`、模型加载和 draft input 构造。
V2 `AscendMambaHybridModelState` 覆盖了 `prepare_attn`、`preprocess_state` 等；上游新增分支不保证被该 override 包含。
310P 使用独立实现，不能直接套标准 NPU 的 Triton 假设。

## 先排除 patch 和安装来源问题

查 `vllm-ascend/vllm_ascend/patch/worker/__init__.py` 的实际导入条件、平台 patch 导入点，以及目标模块的赋值语句。
在实际 worker 初始化完成后采集符号来源；在另一个 Python 进程手动 import patch 只能证明补丁能加载，不能证明故障进程已加载。

- V1 rejection：`patch/worker/patch_rejection_sampler.py` 改 `vllm.v1.sample.rejection_sampler` 的模块函数。
- V2 rejection：`patch/worker/patch_v2/patch_triton.py` 同时改 `vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils.rejection_sample` 和 importer 的同名 alias。
- V2 precopy：同一个 `patch_v2/patch_triton.py` 赋值 `mamba_utils.precopy_mamba_align_fused_kernel`；它在 `HAS_TRITON` 条件下导入。
- V2 model state：`patch/worker/patch_v2/patch_model_state.py` 改父 runner 模块的 `init_model_state`。
- 比较 patch 目标和调用方 alias。`from module import symbol` 的早绑定可能使只替换原模块无效；不要只看函数注释宣称的替换位置。

## 建立最小对照

保留用户故障命令作为原始样本，先在独立进程运行同权重、同 prompt/token IDs 的无投机对照。
然后保留投机 method/权重，使用 `K=1`、固定 seed、greedy、短输出和尽可能小的 batch 重现；大模型允许保留必要 TP。
target 和 draft 的 eager 配置分别核实，不能只设置 target `enforce_eager` 就宣称所有图已关闭。
每轮只改变一个维度，优先顺序按症状选择：

1. 无投机 vs 投机：区分模型本体和提议/验证/状态提交。
2. eager vs 原图模式：区分算法与 capture/replay、padding、缓存地址更新。
3. 单请求 vs 两个不同 prompt 长度请求：暴露行重排和 mixed prefill/decode。
4. 同步 vs 原 async：暴露 accepted counts、seq lens 或 stream 依赖。
5. prefix cache miss vs hit、跨 block vs 不跨 block：仅在状态恢复相关故障时扩展。

不要一次关闭所有性能功能后宣称找到原因；记录每轮唯一改动及首个异常 step 是否移动。
如果限制条件导致某个对照不受支持，报告分支限制，改用同路径的更小案例。

## 跟踪一轮 target 验证和下一轮提议

| 阶段 | 精确观察点 | 最少证据 |
| --- | --- | --- |
| 调度 | `vllm/vllm/v1/core/sched/scheduler.py::Scheduler.schedule` | request ID、实际 scheduled token 数、`scheduled_spec_decode_tokens` |
| V1 target 输入 | `NPUModelRunner._prepare_inputs`、`_calc_spec_decode_metadata`、`_build_attention_metadata` | positions、query 边界、logits 选取与 request 行映射 |
| V1 验证/提议 | `sample_tokens` → `_sample` → `propose_draft_token_ids` | target logits 对应 token、有效 sampled IDs、下一轮 drafts；查看 `_skip_drafting` 是否触发 |
| V2 target 输入 | Ascend `prepare_inputs` → model state `prepare_attn` | `idx_mapping`、CPU/device seq lens、draft 数和 padding |
| V2 验证/提议 | 父类 `GPUModelRunner.sample_tokens` → Ascend `sample` / `postprocess_sampled` → speculator `propose` | `num_sampled`、`num_rejected`、输出 IDs、draft logits 和下轮 draft IDs |
| 调度回写 | `Scheduler.update_from_output` | generated IDs、accepted/rejected 数、computed tokens 修正 |

V1 `sample_tokens` 在普通/PP、padded/unpadded drafter 下提议和 bookkeeping 顺序不同，应按实际条件走读。
V2 拒绝后 CPU 长度更新还经过 `_copy_num_computed_tokens_to_cpu`、`_update_seq_lens_cpu` 的 event 等待；PP 非末级也可能需要修正 counts。
accepted draft 数、包含 bonus/recovered token 的 sampled 数、Mamba one-based accepted count 是不同语义，先从当前消费者确认再做等式。
greedy 的 draft 与 target argmax 不同会被拒绝；低接受率本身不是最终输出错误。
随机采样还需区分有无 draft logits 的 rejection 分支，不应要求不同随机执行路径逐 token 一致。

## 判断是否进入正确的递归算子

- Mamba2：从 `vllm/vllm/model_executor/layers/mamba/mamba_mixer2.py::MambaMixer2` 和 `vllm/vllm/v1/attention/backends/mamba2_attn.py` 追踪实际 backend/算子支持。
- GDN：从 `vllm/vllm/model_executor/layers/mamba/gdn/base.py::GatedDeltaNetAttention` 到 Ascend `ops/gdn.py::AscendGatedDeltaNetAttention._forward_core`。
- GDN metadata：`vllm-ascend/vllm_ascend/ops/gdn_attn_builder.py::AscendGDNAttentionMetadataBuilder.build` 负责 spec/non-spec 分组及 graph padding。
- spec GDN 同时向 causal conv 和 recurrent kernel 传 accepted counts；只检查其中一个状态不足以证明正确。
- `_forward_core` 将 spec/non-spec token 分开计算再按 index 合并，检查分组顺序、`spec_state_indices_tensor`、`non_spec_token_indx` 与原 token 位置一致。
- 非投机 mixed prefill/decode 的 recurrent decode 与 chunk prefill 也分开计算；不要因 `num_prefills > 0` 就假设整个 batch 使用 prefill 算子。
- builder 的 `_remove_spec_graph_padding_queries` 将 GDN dummy request 设为零长度；共享 FIA 的 padded query span 不能直接当作真实 recurrent 更新长度。
- builder 的 `_treat_single_token_prefills_with_state_as_decodes` 只针对已有 state 的单 token prompt chunk；首 token prefill 语义不同。

## 采集证据而不扰动时序

优先用已有 `VLLM_LOGGING_LEVEL=DEBUG`：V1 `_log_propose_draft_token_ids_entry` 输出 `[spec/dfx]` 和实际 drafter/method/K。
按 request ID + step + rank + phase 关联日志，同时记录 batch row 和持久 request index；连续 step 的行号可能变化。
默认采 CPU metadata、shape/dtype/stride、条件分支和图 key；小窗口设备值集中复制，放在已存在的同步边界。
逐元素 `.item()`、`.tolist()`、`.cpu()` 都可能引入同步；不要写成“列表推导式是单次同步”，也不要把插桩后的时序正常当作竞态已修复。
遇到 NaN 可在受控复现启用已有 `VLLM_RAISE_ON_LOGIT_NANS=1`，核查当前 sampler 是否走这条路径。
设备异常只给出异步报错栈时，在最小复现中临时加 phase 边界同步定位；定位后移除再验证原调度。

## 选择验证路由并交付

先读目标测试的依赖、skip 条件和 fixture，按问题选择，不把上游 CUDA 测试原样作为 NPU 验证命令。
具体命令见 [references/reproduce-and-probe.md](references/reproduce-and-probe.md)。

| 问题 | Ascend 仓库现有测试 |
| --- | --- |
| V2 speculator 分派 | `tests/ut/worker/v2/test_spec_decode.py` |
| V1 rejection 逻辑 | `tests/ut/sample/test_rejection_sampler.py` |
| GDN metadata/混合 batch | `tests/ut/ops/test_gdn_attn_builder.py`、`test_gdn_mixed_batch.py` |
| V2 hybrid model state | `tests/ut/worker/test_model_runner_v2_mamba.py` |
| PP 拒绝后各 rank counts | `tests/ut/worker/test_spec_pp_cpu_counts.py`、`tests/ut/patch/worker/test_patch_spec_pp.py` |
| 两卡 Qwen3.5 + V2 冒烟 | `tests/e2e/pull_request/two_card/model_runner_v2/test_mamba_hybrid.py` |

最后一项当前仅断言输出非空，eager case 还没有开启投机；不能据此宣称跨模式精度相等。
有修改时在最近的现有测试中覆盖真实失效输入，并重跑原始故障配置；需模型精度验证时比较 tokens/logits 或已有评测。
交付应包含：复现命令和版本、已证实调用链、第一处违反契约的值、原因与候选修复位置、已执行测试及结果、剩余未验证项。
将“代码推断”“mock 单测”“NPU 实测”明确区分，未运行的命令标为待运行。
