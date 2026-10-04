---
name: vllm-ascend-lora-dev
description: 开发或适配 vLLM-Ascend 的 LoRA 功能，选择权重加载、层替换、Punica、runner 或 MoE 的修改边界，并建立精度和性能回归。适用于新增模型/target module、TP/EP、fully-sharded、graph、量化或投机推理组合的支持与上游升级。
---

# LoRA 功能开发与回归

把需求落到“哪个 adapter、哪些层、什么执行分支、什么可观察结果”。本技能针对 PEFT LoRA adapter，不把 MLA 模型自身的 `q_lora_rank/kv_lora_rank` 视为 adapter 功能。

## 确定可验证的目标

先定位两个仓库、阅读适用 `AGENTS.md`、检查本地改动和实际导入包。约定上游根为 `V`，插件根为 `A`。源码索引基于 2026-10-04 的 vLLM `d61081dc3d3f1740a5d8bf82608b62974393c2de`、Ascend `9b8fc5d728e1ea54dc277562b296112d905e15c1`；升级后重新查符号与签名。

从现有命令、日志和 adapter 文件收集：base/adapter/tokenizer revision，architecture，dense/MoE/GDN，r/alpha/rsLoRA，实际 target modules 与 A/B shapes，dtype/量化，NPU/CANN/torch/torch-npu/triton-ascend，V1/V2，TP/EP/PP/DP/CP，target/draft graph、spec method、prefix/chunked prefill，max_loras/max_cpu_loras/fully_sharded_loras。

定义一个最小成功用例和要保留的已有行为。例如“某 packed projection 支持两个不同 adapter 混批，base 行无 delta，TP=2 graph 重放结果与各自单独 eager 对齐”，而不只是“支持 LoRA”。

## 按职责选择实现位置

| 问题 | 首选检查入口 |
| --- | --- |
| checkpoint 校验、名称/rank/scale | `V/vllm/lora/peft_helper.py`、`lora_model.py`、`lora_weights.py`、`worker_manager.py` |
| 模型命名/packed projection | 模型的 `packed_modules_mapping`、`hf_to_vllm_mapper`、`lora_skip_prefixes`；`V/vllm/lora/model_manager.py` |
| Ascend layer 替换和 packed A/B | `A/vllm_ascend/lora/utils.py`；对照 `V/vllm/lora/layers/` |
| token mapping 与 adapter slots | 上游 `model_manager.py`、`punica_wrapper/utils.py`、`v1/worker/lora_model_runner_mixin.py` |
| NPU dense 调用/工作区/快路径 | `A/vllm_ascend/lora/punica_npu.py`、`A/vllm_ascend/lora/lora_ops.py`（含专用 Triton 实现） |
| MoE 路由与 delta 注入 | `A/vllm_ascend/lora/fused_moe.py`、`quant_moe.py`；结合 MoERunner 与通信路径 |
| adapter 请求进入 V1/V2 | 各 runner 的 LoRA mixin 调用、V2 `V/vllm/v1/worker/gpu/lora_utils.py:LoraState` |
| graph 分发/捕获 | 上游 `gpu/lora_utils.py`、当前 runner/graph manager 与 Ascend `patch/worker/patch_cudagraph.py` |
| 在线 name 与 reload | `V/vllm/entrypoints/openai/models/serving.py`、worker manager 与 prefix hash |

优先扩展已有 Ascend wrapper/算子和插件入口。仅平台无关契约应放上游；不要为了一个 NPU kernel 差异复制整个 loader/model runner，也不要绕过配置校验来“支持”未知组合。

## 写明四份契约再实现

1. **加载契约**：checkpoint A/B 布局、模型 rename/packing、TP/EP 切片、rank padding、scale 是否已折入 B、遗漏层怎样清零。`PEFTHelper.from_dict` 会过滤未声明字段，配置文件存在某字段不能证明它生效。
2. **计算契约**：对应行的 `base + delta`，add/overwrite 语义、输出 slice、无 LoRA/padding 行、累加 dtype。不要把 wrapper 签名当作底层算子语义，先读实际调用。
3. **路由契约**：request → adapter ID → device slot → token/sampler/expert row，含 LRU 重分配与异构 batch。MoE 的 top-k 权重、local expert IDs、排列与通信必须同 hidden states 一致。
4. **执行契约**：eager/capture/replay、persistent buffer 地址、workspace shape key、stream 依赖、动态加载/重写权重的生效边界。目标 runner 和 drafter 分别核实，不能推断 LoRA 自动传播到 draft model。

用一个小张量或两请求具体实例手算这些契约，再选修改点。性能优化不应靠漏算 LoRA、丢掉 target modules 或改变用户精度阈值获得收益。

## 需特别核对的当前实现

- `A/vllm_ascend/platform.py:get_punica_wrapper` 选 NPU Punica；`PunicaWrapperNPU.__init__` 调用 `refresh_all_lora_classes`，将 Ascend wrappers 放在候选类前面。查实际替换类和 MRO，不只查 import。
- `max_loras=1`、硬件和 `fully_sharded_loras` 会影响单槽快路径；一个活跃 adapter、但容量设为 4 的实验不等同于单槽配置。
- LoRA mixin 当前将 `is_prefill=True` 传入非 CUDA 路径。不能按 SGMV/BGMV 名字判断真实 prefill/decode；使用 scheduler 的真实 token 阶段。
- packed 权重是普通 stacked 权重之外需要维护的状态，`set_lora`、`reset_lora`、槽位复用都需覆盖。
- `LoRAModelManager.set_adapter_mapping` 同时依赖 mapping 和 slot layout，LRU 重排后即使请求不变也要刷新。
- MoE 目前有独立的未量化与 W8A8 动态量化注入路径。量化 delta 要在正确浮点边界加入，并验证 top-k scale 是否恰好应用一次。
- MoE 的 FusedMC2、dynamic EPLB 等限制由具体实现校验；量化路径还有 MC2、antiquant offsets、fused scale-bias 等约束。已有支持文档不是所有量化/通信组合的兼容性保证。
- multimodal tower/connector、classification modules_to_save 和 fully-sharded 是独立契约；只有用户需求触及才展开，不从 dense language 支持推定它们都支持。

## 最小回归策略

读 [references/regression-map.md](references/regression-map.md)，复用邻近测试和 fixtures。优先写能在修改前失败、在修改后通过的可观察断言。

**精度最少分三类对照：** LoRA 完全关闭；LoRA 已启用但本请求为 base；同 base 加指定 adapter。零 delta adapter 还可检测无效行或加法污染，但不能替代真实 adapter 测试。

在此次变更涉及的维度扩展：单独/混批，不同权重 A/B/base，不等长请求与重排，rank 与 projection 边界，TP=1/目标 TP，fully-sharded，冷/热 prefix，slot 复用，eager/多次 graph replay，必要的 spec 和 EP 场景。

**验证层次：** CPU/mock UT 验证路由与布局；真实 NPU 局部 reference 验证输出 delta/未选行；模型同前缀 logits/token 或任务评测验证效果。文件名带 correctness 或特性 marker 不代表具备相应 oracle。

**性能对照：** 相同模型/权重/输入/有效输出/请求到达分布，分别测关闭、enabled-base、热单 adapter、目标混合 adapter；冷加载/CPU cache/device activation 单独测。记录 TTFT/TPOT/吞吐、每个 adapter 的尾延迟和 NPU 内存，短 trace 仅用于归因，收益用无 profiler 的运行确认。

选择能回答本次问题的组合，不做无目的全笛卡尔积。若只是包装/注册改动，已有精准 UT 够用的部分无需新增机械测试；如果改变输出或性能，必须安排对应设备/模型验证并报告未完成项。

## 交付

交付“问题 → 契约 → 修改点 → 精度证据 → 性能证据”，标出源码 SHA、模型/adapter 指纹、实际命中的路径、测试命令/退出结果/skip 和未验证组合。

只有任务要求提交/PR 时才执行贡献流程。本技能不主动升级环境、下载模型或运行大规模性能实验；先复用任务中已有的环境和规模，无设备时仍完成可审阅的实现/回归方案并明确限制。
