# 回归入口与证据边界

路径相对双仓工作区。先看实际断言与 fixtures，再决定该测试能否回答当前问题。
下面列的是源码核对结论，不表示这些测试已在当前电脑运行。

## 精度验证的三层 oracle

1. **配置与装载。** base、adapter revision、name/id/path、模块名、A/B shape、scale、
   mapper、device slot 都能追踪；装载成功只能证明通过该 loader 的条件。
2. **算子。** 固定 X、A、B、base 输出，对单请求与混合 adapter 行计算显式低秩增量，
   验证 shrink/expand、inactive 行、packed slices 及 TP 局部/归约结果。
3. **模型。** 相同 base 与 adapter 的 PEFT/oracle、vLLM eager、Ascend graph/spec 对照，
   在同一 token 前缀比较 logits/首差，随后用目标任务数据做质量评测。

PEFT 对照须固定 eval 模式、关闭训练 dropout、tokenizer/chat template、dtype、base revision，
禁止重复 merge+load。先独立验证 PEFT adapter A 有正确非零增量，
不要用一个错误的 oracle 来判定 NPU 正确。仅将经过确认支持的模型/adapter 组合用于对照；
无法获得 PEFT 环境时，明确缺少此证据并完成同 NPU 的 base/no-op/混批和算子层比较。

## 上游测试

| 入口 | 实际覆盖与边界 |
| --- | --- |
| `vllm/tests/lora/test_peft_helper.py`：`test_peft_helper_pass` / `test_peft_helper_error` / `test_peft_helper_invalid_rank_direct` | 配置校验与错误路径；不证明模型输出正确 |
| `vllm/tests/lora/test_lora_weights.py`：`test_moe_packing_uses_stored_scaling`、`test_non_gated_moe_keeps_w3_unscaled` | 检查 rsLoRA scale 的 packing 与共享 w3 避免重复缩放；不是 NPU GEMM 测试 |
| `vllm/tests/lora/test_lora_checkpoints.py`：`test_load_checkpoints` | 某些 Baichuan 格式装载成功、错误架构 adapter 报错；不比较 logits |
| `vllm/tests/lora/test_lora_huggingface.py`：`test_load_checkpoints_from_huggingface` | 本地路径/Hub ID 均能装载，最终断言对象非 None；不是 HF/PEFT 数值对照 |
| `vllm/tests/lora/test_layers.py`：`test_linear_parallel`、`test_column_parallel_packed`、`test_merged_column_parallel_variable_slice` | 选择实际包装层/TP/packed 算子回归前先读参数化设备与 reference；上游测试不能自动等价为 Ascend 硬件验证 |
| `vllm/tests/lora/test_lora_manager.py`：`test_set_adapter_mapping_refreshes_after_slot_reassignment` | 模拟 adapter 被 LRU 替换后 slot `[1,2]→[2,1]`，相同 mapping 的 token indices 必须 `[0,1]→[1,0]`；不含真实服务请求输出 |
| `vllm/tests/v1/core/test_kv_cache_utils.py`：`test_lora_path_change_changes_block_hashes` | 相同名字不同路径产生不同 hash；未证明同路径内容更新自动失效 |

## Ascend UT

| 入口 | 实际覆盖与边界 |
| --- | --- |
| `vllm-ascend/tests/ut/lora/test_utils.py`：`test_refresh_all_lora_classes_prepends_ascend_wrappers` | class 注册顺序，不能证明 runtime 每一层选择了它 |
| `vllm-ascend/tests/ut/lora/test_lora_ops.py`：`test_bgmv_shrink_forwards_scale_in_kernel_order` / `test_bgmv_expand_slice_forwards_offset_and_size` | mocked `_C_ascend` 参数次序、scale、offset；没有执行真实 NPU kernel |
| `vllm-ascend/tests/ut/lora/test_punica_npu.py`：`test_prefill_shrink_and_expand_skip_when_no_lora`、`test_add_lora_logits_uses_sampler_indices` | no-op 分支与 sampler mapping；不能替代 graph 后 base 行不污染的端到端测试 |
| 同文件 `test_add_lora_fused_moe_masks_disabled_adapter_rows`、`test_add_lora_fused_moe_scales_shrink_buffer_by_routed_weight` | adapter 禁用行、router weight 对增量的作用；参数化范围之外需补验证 |
| `vllm-ascend/tests/ut/lora/test_lora.py`：`test_allgather_routing_preserves_multi_adapter_and_base_mapping`、`test_all2all_routing_uses_local_experts_and_exchanged_adapters` | MoE token/expert/adapter 映射变换；本身不证明所有通信/算子精度 |

Linux 已配置测试环境中，从 `vllm-ascend/` 执行：

```bash
"$PYTHON" -m pytest -q tests/ut/lora/test_punica_npu.py -k 'no_lora or sampler_indices or disabled_adapter'
"$PYTHON" -m pytest -q tests/ut/lora/test_lora_ops.py
```

`$PYTHON` 应指向已核实的环境解释器；若执行上游仓库测试，遵守其 uv 与 `.venv/bin/python` 要求。
这些命令只选择与疑点有关的用例，不意味着它们不依赖 torch/Ascend 测试 fixtures。

## NPU kernel 与模型 E2E

| 入口 | 真正断言了什么 |
| --- | --- |
| `vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_lora_linear.py::test_lora_linear_triton_accuracy` | rank 8/16/32/64、若干 token 和 Qwen3.5 TP-local projection shape，以 FP32 显式 matmul 增量为 reference，整体 relative L2 <0.005；inactive 行要求与原 base 输出完全相等 |
| `vllm-ascend/tests/e2e/pull_request/one_card/lora/test_qwen35_densemodel_lora.py::test_qwen35_text_lora` | 三个 SQL prompts 生成文本 `.strip()` 后等于固定预期；使用 temperature=0.01，不是所有输入或同前缀 logits oracle |
| `vllm-ascend/tests/e2e/pull_request/two_card/lora/test_qwen35_densemodel_lora_tp.py::test_qwen35_text_lora` | TP=2，fully-sharded False/True，共用上述三条 SQL 固定输出断言 |
| `vllm-ascend/tests/e2e/pull_request/one_card/lora/test_lora_with_spec_decode.py::test_batch_inference_correctness` | Qwen3-1.7B+Eagle3、同一 LoRA、重复 100 个相同 prompt，spec off/on 文本相等数要求 `> int(0.90*N)`；不是 100% token 等价，也不是 Qwen3.5 GDN/MTP 的证明 |
| `vllm-ascend/tests/e2e/pull_request/two_card/lora/test_qwen35_4b_multi_lora_mtp_aclgraph.py::test_qwen35_4b_named_multi_lora_mtp_aclgraph` | Qwen3.5-4B、TP=2、fully-sharded、MTP K=3、FULL_DECODE_ONLY；确认两个 API 模型名存在，顺序请求两 adapter 输出非空且文本不同 |

最后一条默认生成两个 rank-8 `down_proj` adapter：A 为零 A/B，B 为随机权重；可由
`MIX_LORA_A`/`MIX_LORA_B` 改用已有权重。它没有并发混批、base/no-op 等值、spec-off、PEFT 对照，
也没有断言每个请求的 adapter 数值正确，不能以该用例通过宣称完整多 LoRA 精度已通过。

算子用例的 relative L2 阈值只适用于其中的 reference 和分布；
总输出的 base 部分可能掩盖小 LoRA 增量错误，所以诊断中还要单独比较 ΔY。

NPU 选择性回归示例，从 `vllm-ascend/` 执行：

```bash
"$PYTHON" -m pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_lora_linear.py
"$PYTHON" -m pytest -q tests/e2e/pull_request/two_card/lora/test_qwen35_densemodel_lora_tp.py
```

先确认卡数、可用模型/adapter fixture 与资源成本；只在当前问题涉及这些组合时运行。
模型影响修复还需保留：独立 A/B/base oracle、同批乱序/退出/slot 复用、当前 graph 重放、
cache 冷热隔离以及目标 adapter 的任务评测。报告每一层证据实际验证了什么、哪些仍未执行。
