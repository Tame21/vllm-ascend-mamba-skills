# 源码边界与测试选择

以下路径相对双仓工作区根目录。适用基线见主技能；版本变化后按符号重定位。
这是阅读导航，不是所有候选都必经的调用图。

## 模型与元数据

| 源文件与符号 | 精度取证重点 |
| --- | --- |
| `vllm/vllm/model_executor/layers/mamba/gdn/base.py`：`GatedDeltaNetAttention` | 上游 GDN 合约和状态布局，结合具体 qwen/kimi/olmo 派生类 |
| `vllm-ascend/vllm_ascend/ops/gdn.py`：`AscendGatedDeltaNetAttention._forward_core` | conv cache `kv_cache[0]` 与 recurrent state `kv_cache[1]`；spec/non-spec 分组；prefill 与 recurrent 分支；merge 写回 |
| 同文件 `_probe_fused_chunk` / `_chunk_gated_delta_rule_fused` | fused prefill 是否命中；对照 FLA/CANN/fallback 的初始状态 dtype 与方向 |
| `vllm-ascend/vllm_ascend/ops/gdn_attn_builder.py`：`AscendGDNAttentionMetadataBuilder` | 真实请求与 graph padding、spec query 划分、conv 与 recurrence 各自消费的元数据 |
| 同文件 `_stable_argsort_for_npu` | stable partition 必须保留分组内 token 顺序 |
| 同文件 `_remove_spec_graph_padding_queries` | FIA 的 K+1 padding query 在 GDN 局部视图变为零长度；不修改共享 MLA 视图 |
| 同文件 `_treat_single_token_prefills_with_state_as_decodes` | `query_len=1` 且 `seq_len>1` 的有状态 prompt chunk 分类 |
| 同文件 `_materialize_graph_request_tensor` / `_materialize_graph_query_start_loc` | stable graph buffer 的实时值与 dummy 值是否每轮刷新 |
| `vllm/vllm/v1/attention/backends/gdn_attn.py`：`GDNAttentionMetadataBuilder` | 上游 metadata 合约、block table 与 state indices 的生成 |
| `vllm/vllm/model_executor/layers/mamba/mamba_mixer2.py`：`MambaMixer2.conv_ssm_forward` | 真正 Mamba2 的 conv/SSD/SSU，以及可选 ReplaySSM 状态；不要套 GDN 的转置和公式 |
| `vllm/vllm/v1/attention/backends/mamba2_attn.py`：`Mamba2AttentionMetadataBuilder.build` | Mamba2 专用 metadata 路径 |
| `vllm-ascend/vllm_ascend/_310p/ops/fla/gdn_310.py` | 310P 实际适配分支，先核对硬件和注册实现，不能只看通用 GDN |

`gdn.py` 当前 prefill 顺序是：可用 FLA 分支 → 可用 CANN fused chunk 分支 → fallback。
fallback 显式执行 `ssm_state[prefill_state_indices].transpose(-1, -2).contiguous()`，
写回也转置；fused 分支使用 gather/scatter。分支 oracle 比较前先规范化布局和 dtype。
检查 `prefill_has_initial_state=False` 的请求是否清除初始状态，避免读取缓存遗留内容。

## 投机采样、state copy 和请求生命周期

| 源文件与符号 | 要关联的证据 |
| --- | --- |
| `vllm-ascend/vllm_ascend/worker/model_runner_v1.py`：`NPUModelRunner._calc_spec_decode_metadata` / `_sample` | flatten 后 target/bonus logits 索引与请求 token 对齐 |
| 同文件 `_update_states_after_model_execute` / `_sync_num_accepted_tokens` | `-1` placeholder 排除后的 accepted counts；上一轮行号映射到当前请求顺序；新请求填 1 |
| 同文件 `_prepare_inputs` / `_build_attention_metadata` / `execute_model` | 事件同步、状态 preprocess、元数据建立与模型 forward 的先后 |
| `vllm-ascend/vllm_ascend/sample/rejection_sampler.py`：`AscendRejectionSampler.forward` / `apply_logits_processors` | target 与 bonus 采样、原地 logits 修改、penalties、allowed tokens、bad words |
| `vllm/vllm/v1/sample/rejection_sampler.py`：`RejectionSampler` | 上游采样语义、PLACEHOLDER_TOKEN_ID、有效输出格式 |
| `vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`：`RejectionSampler.__call__` | V2 采样器返回的 `num_sampled` / `num_rejected`，不要套用 V1 sampler 的调用签名 |
| `vllm-ascend/vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py`：`rejection_sample` | V2 NPU 验证实现；其调用 alias 由 `patch/worker/patch_v2/patch_triton.py` 同时替换 utils 与 sampler 模块 |
| `vllm/vllm/v1/worker/gpu/model_runner.py`：`GPUModelRunner.sample_tokens` / `postprocess_sampled` | sampled 路径把 `num_sampled` 交给 model state；未采样的 prefill 分支显式传 0 |
| `vllm/vllm/v1/worker/gpu/model_states/mamba_hybrid.py`：`MambaHybridModelState.postprocess_state` / `_scatter_num_accepted_kernel` | V2 将 `num_sampled` 按 `idx_mapping` scatter 到持久槽位并钳制为至少 1；负映射 sentinel 跳过 |
| `vllm-ascend/vllm_ascend/worker/v2/model_runner.py`：`NPUModelRunner.postprocess_sampled` / `_copy_num_computed_tokens_to_cpu` / `_update_seq_lens_cpu` | 调父类完成 reject 修正和 state 更新后同步 CPU counts；下一轮 CPU seq lens 构造前等待 D2H event |
| `vllm/vllm/v1/worker/mamba_utils.py`：`preprocess_mamba` / `postprocess_mamba_align_gpu` / `MambaSpecDecodeGPUContext` | 前缀状态复制、accepted 偏移、block 跨越及状态提交 |
| `vllm-ascend/vllm_ascend/patch/worker/patch_mamba_utils.py`：`_collect_mamba_copy_meta_torch` / `_tensor_view_from_data_ptr` | `accept_token_bias + 1` 的状态选择与 page padding/非连续 view，不可假定整张 state tensor 连续 |
| 同文件 `_stage_mamba_copy_metadata` / `_do_mamba_copy_block_npu` | 元数据准备与真正 state copy 的延后执行 |
| `vllm-ascend/vllm_ascend/ops/triton/mamba/postprocess.py`：`postprocess_mamba_fused_kernel` | 接受后实际写回/状态位置更新的 NPU kernel |
| `vllm-ascend/vllm_ascend/worker/v2/model_states/mamba_hybrid.py`：`AscendMambaHybridModelState.preprocess_state` / `do_mamba_copy_for_layer` | V2 的 state 管理继承上游，按层 load 后再 copy 的覆盖逻辑 |
| `vllm-ascend/vllm_ascend/ops/triton/mamba/state_index.py`：`gather_ssm_states` / `scatter_ssm_states_` | 只读写选中 state slot，非连续 state cache 的地址与 row stride |

记录 accepted/draft/bonus 的语义时附一条真实样例：本轮 draft IDs、验证输出含 placeholder 的完整行、
有效输出个数、下一轮读取的 state slot 和偏移。不要凭变量名把“接受 draft 数”和“有效输出 token 数”混为一谈。
保持 request ID → 当前 batch row → 前一轮 row → block/state slot 的映射证据。
V1 的 Ascend `_update_states_after_model_execute` 中显式计数适用于异步 hybrid；同步路径调用父类。
V2 的未采样 prefill 使用 `postprocess_state(..., 0)`，内部 scalar 与 tensor 分支都将 accepted count 保持为至少 1。
这份中性 state 计数与请求实际输出长度不同；第二个 chunk 首差时同时保存两者，不能看到 1 就认定采样过 token。

## 按疑点选择回归入口

下面是定位入口，执行前检查 fixtures、依赖、设备与模型条件。Ascend UT 可能 mock 掉 NPU 算子，
只能证明 Python 合约；上游 GPU/Triton kernel 测试不能直接当作 Ascend 已通过的证据。

| 疑点 | 测试文件与代表用例 |
| --- | --- |
| GDN graph request/token 粒度 | `vllm-ascend/tests/ut/ops/test_gdn_attn_builder.py::test_full_graph_spec_conv1d_args_keep_request_granularity` |
| graph padding 刷新 | 同文件 `test_spec_graph_fia_padding_refreshes_captured_buffers`、`test_full_graph_idle_dummy_uses_zero_length_recurrent_metadata` |
| 一 token prompt chunk | 同文件 `test_one_token_prefill_selection_respects_recurrent_state`、`test_spec_width_prompt_chunk_retains_prefill_metadata` |
| device state 索引 | 同文件 `test_causal_conv1d_cache_indices_use_device_block_table`、`test_mamba_align_cache_indices_follow_device_seq_lens` |
| 混合 non-spec prefill/decode | `vllm-ascend/tests/ut/ops/test_gdn_mixed_batch.py::test_mixed_non_spec_reuses_rearranged_qkv` |
| state copy 时序 | `vllm-ascend/tests/ut/patch/worker/test_patch_mamba_utils.py::test_preprocess_stages_metadata_but_defers_state_copy` |
| V2 state/layout | `vllm-ascend/tests/ut/worker/test_model_runner_v2_mamba.py` 中 `test_mamba_cache_reshape_uses_page_strided_state_tensors`、`test_prepare_attn_keeps_actual_counts_separate_from_padding` |
| state 提交 NPU kernel | `vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_postprocess_mamba.py::test_matches_python_postprocess_mamba` |
| NPU state gather/scatter | `vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mamba_state_index.py` |
| stable token partition | `vllm-ascend/tests/e2e/pull_request/one_card/test_gdn_stable_argsort.py::test_boolean_token_partition` |
| rejection sampler | `vllm-ascend/tests/ut/sample/test_rejection_sampler.py` |
| 上游 GDN 合约 | `vllm/tests/v1/attention/test_gdn_metadata_builder.py` |
| 上游 cache 边界 | `vllm/tests/v1/core/prefix_cache/test_mamba_eagle_resume_checkpoint.py`、`vllm/tests/v1/e2e/general/test_mamba_prefix_cache.py` |
| 真 Mamba2 recurrence oracle | `vllm/tests/kernels/mamba/test_mamba_mixer2.py`、`vllm/tests/kernels/mamba/test_replayssm_prefill_decode_equivalence_mamba2.py` |

Linux/NPU 主机上的示例，从 `vllm-ascend/` 仓库目录执行，`$PYTHON` 指向已确认的测试环境解释器：

```bash
"$PYTHON" -m pytest -q tests/ut/ops/test_gdn_attn_builder.py -k 'spec_conv1d or one_token or padding'
"$PYTHON" -m pytest -q tests/ut/sample/test_rejection_sampler.py
"$PYTHON" -m pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_postprocess_mamba.py
```

只有疑点覆盖到时才扩展用例，不必每次跑整张测试表。修改模型行为还需用目标模型做端到端对照和评测。

## 防止误读已有测试

- `vllm-ascend/tests/e2e/pull_request/four_card/spec_decode/test_mtp_qwen3_next.py` 中
  `test_qwen3_next_mtp_acceptance_tp4` 读取 per-position acceptance 并与阈值比较；
  它不是 spec off/on 的 token 或任务精度比较。
- `vllm-ascend/tests/e2e/pull_request/one_card/spec_decode/test_mtp_eagle_correctness.py` 的
  `test_deepseek_mtp` 在此版本只调用 speculative `generate`，没有 off/on 输出断言。
  通过说明执行成功，不能单独证明正确性。
- `vllm/tests/v1/e2e/spec_decode/mtp/_correctness.py::check_mtp_correctness` 实际运行 off/on
  和 GSM8K，并采用允许部分差异的请求匹配阈值；诊断首差时仍需逐 token 证据，
  不应把测试注释中的说明直接当成当前断言阈值。
