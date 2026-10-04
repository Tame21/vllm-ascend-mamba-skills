---
name: vllm-ascend-mamba-dev
description: 开发或移植 vLLM-Ascend Mamba/GDN 类模型的投机推理功能，梳理上下游接口、选择修改层级并设计有证据的回归验证。适用于新增 MTP/EAGLE 支持、runner 迁移、算子或 metadata 改动及上游升级适配。
---

# Mamba 投机推理开发与回归

把需求落实为明确的输入输出契约、最小修改和对应验证。用用户当前代码与模型配置判断能力，不以“同属 Mamba”推断所有 recurrent 模型、draft 方法或硬件支持相同。

## 固定工作基线

定位 vLLM 根目录 `V`、vLLM-Ascend 根目录 `A`，阅读各自及目标目录的 `AGENTS.md`。检查已有改动，记录两边 HEAD、部署解释器和实际导入路径。仓库 checkout 与安装包可能不是同一份代码。

本技能源码锚点来自 2026-10-04：vLLM `d61081dc3d3f1740a5d8bf82608b62974393c2de`，Ascend `9b8fc5d728e1ea54dc277562b296112d905e15c1`。这是可追溯快照，不是兼容性保证；当前上游和插件存在不同版本接口及兼容 patch，升级必须核对签名和调用者。

先根据仓库与用户材料填写以下信息；只有确实阻碍选择实现时才询问缺失项：

- target/draft 模型 architecture、revision、权重、tokenizer，recurrent 层实际类型与 state shapes。
- draft 方法、`num_speculative_tokens`、draft/target 是否共享权重或 hidden states、需要的 draft logits。
- NPU 型号、CANN/driver、torch/torch-npu/triton-ascend 版本，runner V1/V2 的实际类。
- eager/graph、async scheduling、TP/PP/DP/CP、cache mode、prefix cache、chunked prefill、connector。
- 本次新增的行为、一个最小成功例子、需要保留的既有行为。

## 选择真正负责契约的层

| 改动类型 | 优先检查位置（相对各仓库根） |
| --- | --- |
| 平台能力、配置推导 | `A/vllm_ascend/platform.py`、`ascend_config.py`、`patch/platform/`；上游 `V/vllm/config/` |
| Ascend op 替换注册 | `A/vllm_ascend/utils.py:register_ascend_customop` 与 `ops/gdn.py:AscendGatedDeltaNetAttention` |
| GDN 请求划分、索引、padding | `A/vllm_ascend/ops/gdn_attn_builder.py:AscendGDNAttentionMetadataBuilder`；对照 `V/vllm/v1/attention/backends/gdn_attn.py` |
| V1 draft proposer | `A/vllm_ascend/spec_decode/`、`worker/model_runner_v1.py`；对照 `V/vllm/v1/spec_decode/` |
| V2 draft 分发与实现 | `A/vllm_ascend/worker/v2/spec_decode/__init__.py:init_speculator` 及其方法子目录；对照 `V/vllm/v1/worker/gpu/spec_decode/` |
| V2 state 生命周期 | `A/vllm_ascend/worker/v2/model_states/mamba_hybrid.py`；对照上游同名 model state |
| 状态布局/分配 | `A/vllm_ascend/worker/v2/attn_utils.py`、V1 runner cache methods；上游 `V/vllm/v1/kv_cache_interface.py` |
| 平台无关调度/采样语义 | 上游 `V/vllm/v1/core/sched/`、`sample/`、`worker/gpu/sample/`，再查 Ascend patch/override |
| NPU kernel | `A/vllm_ascend/ops/`、`ops/triton/` 或 `csrc/`；查当前算子实际调用及已有 reference |

模型支持通常沿上游模型、Ascend custom op/继承/必要 patch 接入，遵循插件架构。不要因需求涉及模型就直接在插件新增一套模型实现，也不要把纯 kernel 细节塞进 runner。

## 修改前写清四个交界面

1. **Scheduler → runner**：scheduled token 是否含 draft placeholders；prefill/decode 判定依据；query_start_loc、position、slot mapping、request slot 的排序和有效长度。
2. **Target → sampler/rejection → proposer**：哪些 hidden states/logits 对应 base/draft/bonus token；拒绝后下一轮起点；EOS、停止条件和 grammar 对计数的影响。
3. **Recurrent state → 下一次 forward**：accepted-count 的定义、conv 与 temporal 的恢复方式、跨 block checkpoint、抢占/恢复和 batch 重排。更新 `seq_len` 不等于恢复 recurrent state。
4. **Capture → replay / host → device**：哪些 buffer 必须持久，哪些地址捕获进图；真实与 padded 请求怎么区分；异步 CPU mirror 是否可能超前于设备真实计数。

用一个请求、一个迭代的具体 shape/计数示例验证这些交界面，再实现。不要把所有 `K` 都理解成相同含义：draft token 数、verify query length、speculative state blocks 分开命名。

## 上游升级与 patch 生效

- 对每个 override 查看当前父类实现和签名，不只做文本 diff。Ascend V2 的 `prepare_attn` 覆写上游函数，新版上游新增分支不会自动继承。
- `init_speculator` 按实际配置分发，不能因为类型注解写 Eagle 就认为所有方法都走 Eagle；MTP 存在模型特例，hidden-state extraction 也有独立分支。
- patch 检查定义、注册 import、硬件门控、加载顺序、最终被调用的 alias。当前 precopy kernel 的替换在 `patch/worker/patch_v2/patch_triton.py`，不是注释所暗示的所有逻辑都在 `patch_mamba_utils.py`。
- 避免为通过测试悄悄改变用户的默认配置。新增 env 在 `A/vllm_ascend/envs.py` 集中定义；使用现有开关前检查名称、类型、默认值和作用阶段。
- 有 CPU mock 的 UT 通过仍不能证明 import 在真实 NPU worker 顺序下生效。必要时在 worker 初始化后输出一次实际类、模块路径和关键被替换函数的来源。

## 设计和执行回归

读 [references/regression-map.md](references/regression-map.md)，按改动契约选择已有测试，先扩展相邻 suite。若新增测试，说明它验证什么可观察行为、修复前为何失败、为什么选择这个测试层级。

最小回归以风险为导向：

- metadata / dispatch：CPU/mock UT 覆盖 shape、稳定排序、真实/padded、空 batch、关键路由。
- copy / recurrent kernel：NPU 对照 reference 检查输出及最终 state；copy 还检查未选行、页 padding、边界 guards。
- state/拒绝语义：非投机基线与 spec 的相同已提交前缀比对；0/部分/全部 draft 接受，跨块和请求重排。
- graph 改动：多次 replay 改变输入/accepted 数和实际 batch，防止只测 capture shape。
- 性能改动：先证明精度，再比较同一 workload 的无 profiler 基线；性能由 TTFT、TPOT/ITL、吞吐、有效 acceptance 共同解释。

不要求每次穷举所有组合。对宣称支持的关键交界组合建立证据，尚未跑的目标环境和组合明确列为未验证。模型权重、设备、依赖未就绪时仍可完成源码、测试设计及可运行命令，不能伪造运行结果。

## 交付

交付具体行为改变、修改位置和选择该层的原因；附实际运行的命令/退出结果、使用的模型/配置/版本、精度及性能对照。已有测试只断言非空就标注为 smoke，不把接受率达标解释成目标分布正确。

若任务包含提交/PR，再按两个仓库的贡献规则处理签名、重复工作检查和 PR 模板；本技能本身不要求创建 PR 或扩大开发范围。
