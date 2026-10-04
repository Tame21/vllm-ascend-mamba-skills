# 变更到回归的映射

`V` 是 vLLM 根，`A` 是 vLLM-Ascend 根。以下清单是测试入口，不是“已通过”记录；运行前阅读 fixture/参数/skip，确认模型与卡数。

## 现有测试能证明什么

| 变更 | 优先测试路径 | 判断边界 |
| --- | --- | --- |
| V2 spec 方法 dispatch | `A/tests/ut/worker/v2/test_spec_decode.py` | MTP/Eagle/DFlash/DSpark/extract 分发；不证明模型精度 |
| V1 proposer / 元数据 | `A/tests/ut/spec_decode/test_mtp.py`、`test_eagle_proposer.py`、`test_multi_kv_cache_group_proposer.py` | mock 的接口与分组，不是端到端设备结果 |
| GDN builder 分类与 graph | `A/tests/ut/ops/test_gdn_attn_builder.py` | mixed、stateful 单 token、spec-width prompt、实际/padded、重放 buffer |
| GDN core / 混合非投机路径 | `A/tests/ut/ops/test_gdn.py`、`test_gdn_mixed_batch.py`、`test_gdn_chunk_meta.py` | UT 的调用与分支；需另选真实 kernel 数值测试 |
| 按层 KV 时序 | `A/tests/ut/ops/test_gdn_layerwise_kv.py`、`tests/ut/worker/test_model_runner_v2_mamba.py` | connector 更新 state 的调用边界、延迟 copy；仍需真实异步环境 |
| Mamba 分配/页布局 | `A/tests/ut/worker/test_model_runner_v2_mamba.py` | homogeneous wrapper、page stride、hybrid/draft 隔离 |
| align 调度与 config | `A/tests/ut/patch/platform/test_patch_mamba_block_aligned_split.py`、`test_patch_mamba_config.py` | PD role 与 verifier window、resolved config |
| V1 copy 与 wrapped groups | `A/tests/ut/patch/worker/test_patch_mamba_utils.py`、`test_patch_mamba_utils_uniform_groups.py` | 状态决策与缓冲区 |
| NPU state copy | `A/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_precopy.py`、`test_postprocess_mamba.py`、`test_postprocess_temporal_copy.py`、`test_mamba_state_index.py` | 真实数据和 guards；仍不能替代整个模型评估 |
| V2 hybrid 模型 smoke | `A/tests/e2e/pull_request/two_card/model_runner_v2/test_mamba_hybrid.py` | 当前 Qwen3.5 TP=2，eager/graph；只断言非空 |
| 投机性能/接受指标 | `A/tests/e2e/pull_request/two_card/spec_decode/test_spec_decode.py` | 读取当前模型/数据集/阈值；不直接复用为任意 Mamba 的精度基准 |
| 上游 checkpoint 契约 | `V/tests/v1/core/prefix_cache/test_mamba_eagle_resume_checkpoint.py`、`tests/v1/core/test_mamba_align_chunk_split.py` | 作为平台无关逻辑参照 |
| 上游 recurrent state | `V/tests/v1/worker/test_mamba_hybrid_model_state.py`、`tests/v1/worker/test_mamba_utils.py`、`tests/kernels/mamba/` | GPU/依赖要求以实际测试为准，不直接在 NPU 上照跑 CUDA 测试 |

`A/tests/ut/conftest.py` 在无 NPU 时安装 torch_npu/torch.npu mocks，UT 仍需要匹配版本的 Python/torch/vLLM 等依赖。不要为让 UT 导入而把真实缺失能力全部 mock 掉，再宣布设备支持。

## 命令模板

以下是 Linux Ascend 开发环境模板，先准备仓库要求的环境。上游要求 uv 管理的 `.venv/bin/python`；若 Ascend 使用另一套已配置虚拟环境，应明确替换解释器路径。

```bash
# 在包含两个 checkout 的工作区保存版本证据
git -C vllm rev-parse HEAD
git -C vllm status --short
git -C vllm-ascend rev-parse HEAD
git -C vllm-ascend status --short

# A 根目录，举例：只修改 GDN builder 或 V2 model state
.venv/bin/python -m pytest --collect-only -q tests/ut/ops/test_gdn_attn_builder.py tests/ut/worker/test_model_runner_v2_mamba.py
.venv/bin/python -m pytest -q tests/ut/ops/test_gdn_attn_builder.py tests/ut/worker/test_model_runner_v2_mamba.py

# A 根目录，真实两卡 NPU smoke；先确认本地权重、显存和可用设备
QWEN35_DENSE_MODEL=/models/Qwen3.5-27B \
  .venv/bin/python -m pytest -sv tests/e2e/pull_request/two_card/model_runner_v2/test_mamba_hybrid.py
```

不要把示例模型换成没有 MTP 权重/对应架构的任意模型。该 smoke 的 graph 测例 target 使用 graph，`speculative_config.enforce_eager=True`；不能据此声称 draft graph 通过。

## 模型精度实验记录

复现时记录以下项目到用户指定输出目录，默认保留最少的可重现样例：

```text
case_id / 日期 / 两个 repo commit 与 dirty diff 标识
Python 和实际包路径 / torch / torch-npu / CANN / driver / NPU
target + draft + tokenizer revision / dtype / quantization
resolved config: runner, spec method/K, target & draft graph,
                 scheduler, parallelism, cache mode, prefix, chunk, connector
输入 token IDs / chat template / seed / sampling / EOS-stop 策略
对照 A / 变体 B / 唯一变化变量
输出 token IDs / 首差位置 / 对齐前缀下 logits 与 recurrent state
已运行命令 / 退出状态 / 实际执行与 skip 数 / 日志与 trace 路径
结论 / 未验证项
```

精度比较先固定 greedy，检查非投机、投机是否稳定重现首差；FP16/BF16 微小 logits 扰动可以放大成后续文本差异，应检查首差前同一前缀及 top-1/top-2 margin。随机采样不要求同 seed 下逐 token 相同；在确认 greedy 和局部 reference 后，用统计/任务指标判断分布或质量。

## 小范围实验矩阵

每行只扩展当前改动触及的维度：

| 风险 | 最小对照 |
| --- | --- |
| speculative metadata | spec off → K=1 → 目标 K；用实际受支持方法 |
| state rollback | 全拒绝/部分/全接受；比相同已提交前缀后的 state |
| batch mapping | 单请求 → 两请求长度不同 → 结束/加入造成 reorder |
| prefill 分类 | 无初态单 token / 有初态单 token / prompt chunk 长度恰为 K+1 |
| cache/checkpoint | prefix off/cold/warm；解析后 B 边界两侧 |
| graph | eager → target graph → 所需 draft graph；同一捕获多次 replay |
| topology | 支持的最小卡数基线 → 需求中的 TP/PP/DP/CP 组合 |

若当前模型或平台不支持某组合，记录源码限制，选择仍保持待测语义的最小组合；不要为了填满矩阵主动修改配置约束。
