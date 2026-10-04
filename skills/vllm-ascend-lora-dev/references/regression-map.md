# 回归测试与支持边界

`V` 为 vLLM 根，`A` 为 vLLM-Ascend 根。以下是本地源码索引，不是本次已执行的设备测试报告。

## 单元/算子入口

| 变更 | 已有入口 | 验证范围 |
| --- | --- | --- |
| PEFT 配置/shape/scale | `V/tests/lora/test_lora_weights.py`、`test_lora_checkpoints.py`、`test_lora_huggingface.py` | 按实际 fixtures 选择，HF/PEFT 和设备依赖不可省略 |
| load/activate/slot/LRU | `V/tests/lora/test_lora_manager.py` | model/worker managers 与槽位重分配；上游设备类型不等同 NPU |
| NPU wrapper 注册 | `A/tests/ut/lora/test_utils.py` | 当前仅检查 wrapper 注册优先级/类名；未覆盖 packing/set/reset |
| NPU Punica | `A/tests/ut/lora/test_punica_npu.py` | dtype/硬件分支、padding、logits 的 sampler mapping、MoE combined index |
| dense/no-op/MoE 映射 | `A/tests/ut/lora/test_lora.py` | 单槽 mask、no_lora、packed matmul 和 MoE routing contracts |
| NPU wrapper ABI | `A/tests/ut/lora/test_lora_ops.py` | 调用参数、expand wrapper 语义；mock 不证明 NPU 数值 |
| 量化 MoE delta | `A/tests/ut/lora/test_quant_moe.py` | 浮点边界、路由、top-k scale、空 EP rank、unsupported 路径 |
| 特定单槽 Triton | `A/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_lora_linear.py` | 读当前参数组合和实际 dtype；不代表所有 NPU LoRA kernels |
| prefix identity | `V/tests/v1/core/test_kv_cache_utils.py` 中 LoRA 相关函数 | name/path 参与 hash；不对同路径文件内容做版本化 |

量化 MoE 的源码是 `A/vllm_ascend/lora/quant_moe.py`，使用注册的 `QuantType.W8A8` 实现并校验输入/通信边界。它不能自动推广到其他量化类型；`test_unregistered_quantized_moe_lora_fails_fast` 是保留明确失败的回归。

packed 权重生命周期尚需补定向用例：实际经过 `set_lora/reset_lora`，验证普通/packed buffers 同步更新、缺失模块归零与 slot 重用后的结果。`test_lora.py::test_packed_single_lora_matmul` 只对预先构造的 packed tensors 验证运算，不证明动态装载和清零正确。

## 模型测试真实断言

| 路径 | 当前内容 | 不应据此宣称 |
| --- | --- | --- |
| `A/tests/e2e/pull_request/one_card/lora/test_qwen35_densemodel_lora.py` | Qwen3.5-4B 三条 SQL 文本与预期列表精确相等；采样 temperature=0.01 | 不是 PEFT logits oracle，也不是任意 adapter 的测试 |
| `A/tests/e2e/pull_request/two_card/lora/test_qwen35_densemodel_lora_tp.py` | 复用上述断言，TP=2，fully_sharded=False/True | 不覆盖不同权重混批、spec 或所有 graph 分支 |
| `A/tests/e2e/pull_request/one_card/lora/test_lora_with_spec_decode.py::test_batch_inference_correctness` | 同 adapter 的 spec off/on，100 个重复 prompt，文本匹配数严格大于 int(0.90*N) | 不是所有 token 一致，也不覆盖任意 GDN+MTP 组合 |
| `A/tests/e2e/pull_request/two_card/lora/test_qwen35_4b_multi_lora_mtp_aclgraph.py` | 两个服务名字、顺序请求非空且输出不同，FULL_DECODE_ONLY、TP2、fully-sharded、MTP | 不证明两份权重混批正确，不证明 PEFT/无投机精度或所有 runner |
| `A/tests/e2e/pull_request/two_card/lora/test_qwen3moe_lora.py` | eager TP/EP SQL 前缀；multi-ID case 混入 base | 两个 LoRA ID 指向同一文件，base 是负例检查，非严格 base oracle |
| `A/tests/e2e/pull_request/one_card/lora/test_qwen3_multi_loras.py` | CPU/device cache 不同容量、add/list/remove、命名 adapter 的生成 | 函数名带 tp，但实际 TP=2 参数被注释；不能报两卡验证 |
| `A/tests/e2e/pull_request/one_card/lora/test_qwenvl_lora.py` | 多种视觉/language adapter、greedy/beam 与切换入口 | 依赖模型/adapter/MM 预处理约束，不泛化到全部视觉模型 |

当前 `A/docs/source/user_guide/feature_guide/lora.md` 描述 dense/MoE 与部分量化/通信支持。结合实现 gate、选中 kernel 和模型测试判断本次功能；release note、测试 marker 或模型列表都不能替代对具体配置的验证。

## Qwen3.5 GDN 与 MTP 的开发例子

`V/vllm/model_executor/models/qwen3_5.py` 的 LoRA `packed_modules_mapping` 包含
`in_proj_qkvz=[in_proj_qkv,in_proj_z]`、`in_proj_ba=[in_proj_b,in_proj_a]`。
新增 target 时从这份映射追到 packed layer 的实际展开和 output slices；不能直接照抄内层
base-weight mapper 的 q/k/v/z 四段，也不能照抄 Qwen3-Next 的单项映射。

上述 Qwen3.5+TP+MTP graph smoke 当前标记为 BF16，默认创建 rank-8 `down_proj` adapters。
它不能证明量化组合或 GDN projections 均支持 LoRA；需要按真实 adapter target 建立独立
delta oracle，再检查 projection 后的 conv/gating/state 和投机提交/回滚边界。
LoRA loader 的 rename-only mapper 与 base loader 的删除/stacking 规则分开核实。

## 命令示例

从 `A` 根目录，用已配置好的虚拟环境选相关 UT，下面是一组示例而非每次必须全跑：

```bash
.venv/bin/python -m pytest -q tests/ut/lora/test_utils.py tests/ut/lora/test_punica_npu.py
.venv/bin/python -m pytest -q tests/ut/lora/test_lora.py tests/ut/lora/test_lora_ops.py
.venv/bin/python -m pytest -q tests/ut/lora/test_quant_moe.py
```

有真实 NPU、匹配依赖及模型文件后，按修改范围选取：

```bash
.venv/bin/python -m pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_lora_linear.py
.venv/bin/python -m pytest -sv tests/e2e/pull_request/two_card/lora/test_qwen35_densemodel_lora_tp.py
```

这些命令可能使用 fixture 中固定的模型名和下载逻辑；先读 fixture、准备匹配的本地文件/缓存。尤其 Qwen3.5 单卡测试在模块级创建 tokenizer，收集测试也可能访问模型资源。上游测试按 `V/AGENTS.md` 使用 uv 管理的 `.venv/bin/python`，不要把 CUDA 命令简单换成 NPU 标签。

## 每次改动保存的最小证据

```text
基线: 两repo SHA/dirty diff；导入来源；NPU/runtime与解释器
模型: base/adapter/tokenizer revision；r/alpha/scale；实际目标模块
配置: max_loras/max_cpu_loras/fully_sharded；TP/EP；dtype/quant
执行: runner、target/draft graph、prefix/chunk；wrapper/kernel分支
workload: prompt IDs、adapter分配/比例、长度、到达时间、采样与输出长度
精度: 首差位置；同前缀logits/delta；无效行未污染；reference与容差依据
性能: 无profiler结果；每adapter TTFT/TPOT/吞吐/错误；load vs hot分开
验证: 命令、实际执行/skip、退出状态、产物；明确未跑的设备/组合
```

验证 layer delta 时，base 主干 FP32/BF16 和量化舍入可能影响结果，先统一同输入与权重语义。模型输出首差后以相同前缀重放定位，不能把后续文本差异累计当作多个独立 kernel 错误。
