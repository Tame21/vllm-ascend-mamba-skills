# LoRA 精度源码图

路径相对双仓工作区根目录，版本基线见主技能。先按符号确认当前实现，行号不是稳定合约。

## 1. 配置、装载与权重变换

| 路径与符号 | 取证内容 |
| --- | --- |
| `vllm/vllm/config/lora.py`：`LoRAConfig` | 服务容量、dtype、fully-sharded 与 target 过滤配置 |
| `vllm/vllm/lora/request.py`：`LoRARequest` | 请求 name/id/path、正 ID 校验、load_inplace；ID 唯一性声明不等于运行时全局强制 |
| `vllm/vllm/lora/worker_manager.py`：`WorkerLoRAManager._load_adapter` | supported modules 展开、PEFT 合法性校验、rename mapper、CPU 权重加载、dtype 转换 |
| `vllm/vllm/lora/peft_helper.py`：`PEFTHelper.from_dict` / `__post_init__` / `validate_legal` | 忽略未声明字段；标准/rsLoRA scale；DoRA、bias、rank 与 modules_to_save 限制 |
| `vllm/vllm/lora/lora_model.py`：`LoRAModel.from_local_checkpoint` / `from_lora_tensors` | safetensors/bin/pt 路径、名称/shape、embedding 词表、skip_prefixes、MoE EP 加载范围 |
| `vllm/vllm/lora/utils.py`：`parse_fine_tuned_lora_name` / `get_supported_lora_modules` | PEFT key 到模型模块名称、可替换层集合 |
| `vllm/vllm/lora/lora_weights.py`：`LoRALayerWeights.optimize` / `PackedLoRALayerWeights.pack` | scale 折入 B，packed 子模块缺失时允许 None |
| 同文件 `PackedLoRALayerWeights.pack_moe` / `pack_moe_stacked` | MoE scale 使用存储的 adapter scaling；非 gated w3 复用 w1 时避免重复缩放 |
| `vllm/vllm/lora/model_manager.py`：`LoRAModelManager.activate_adapter` / `_create_merged_loras_inplace` | 实际应用层数、packed 变换、set/reset 与 slot 生命周期 |

验证 PEFT 配置时不要只读 `target_modules`：真正决定装载的是 checkpoint keys、模型支持层、
服务端过滤和 mapper 的交集。此版本 `rank_pattern`/`alpha_pattern` 不在 `PEFTHelper` 字段内，
它们会在 `from_dict` 被过滤；不能把 PEFT 所有训练特性自动视为该 loader 的支持项。

`WorkerLoRAManager._load_adapter` 调用 `hf_to_vllm_mapper.get_rename_mapper()`，
只保留重命名，丢弃 stacking/deletion；`lora_skip_prefixes` 是单独传入的路径。
因此看到 base 权重 mapper 里 `"mtp.": None`，还需要继续核对 LoRA checkpoint 如何处理 MTP key。

取证保存原始 B、缩放前后 B 与最终 device slice 的独立快照。
`optimize()` 原地修改 B；`LoRAModel.clone()` 共享底层 tensors，不能把浅 clone 当权重快照。
读取未知来源的 adapter 时优先安全的配置/张量格式与既有 loader，不为检查 shape 自行执行 pickle。

## 2. 模型 packed mapping 的具体差异

| 模型文件 | 此版本映射 |
| --- | --- |
| `vllm/vllm/model_executor/models/qwen3_5.py` | `qkv_proj=[q_proj,k_proj,v_proj]`、`gate_up_proj=[gate_proj,up_proj]`；GDN `in_proj_qkvz=[in_proj_qkv,in_proj_z]`、`in_proj_ba=[in_proj_b,in_proj_a]` |
| `vllm/vllm/model_executor/models/qwen3_next.py` | 标准 attention/MLP 同前；GDN `in_proj_qkvz=[in_proj_qkvz]`、`in_proj_ba=[in_proj_ba]` |

从 PEFT key 到 packed module 再到 kernel output slices 可能经历多步拆分；
`MergedColumnParallelLinearWithLoRA.expand_packed_lora` 会处理部分可展开情形。
不要从 checkpoint target 数量推断最终 slice 数量，也不要把 Qwen3.5 映射拷给 Qwen3-Next。
不同模型的 Q/K/V heads 与 GDN dimensions 不同，B 的输出区段不必等长。

## 3. Ascend 运算分支与并行

| 路径与符号 | 首差候选 |
| --- | --- |
| `vllm-ascend/vllm_ascend/lora/utils.py`：`refresh_all_lora_classes` | Ascend wrappers 排在上游候选前，但 `can_replace_layer` 仍有条件 |
| 同文件 `_PackedLoRAAWeightsMixin.set_lora` / `reset_lora` | A 拼接、B 的转置块对角 packing、slot 清零、各 output slice 偏移 |
| 同文件 `_TransposedLoRABMixin` | contiguous `[rank,output]` B 转置副本与原 `[output,rank]` 的对应；独立缓冲由 `copy_` 更新，不会随原 B 自动变化 |
| `vllm-ascend/vllm_ascend/lora/punica_npu.py`：`PunicaWrapperNPU._update_base_metadata` / `update_metadata` | token slot 与 single-LoRA mask；decode 也刷新 no_lora |
| 同文件 `add_lora_linear` / `_lora_linear_kernel` / `_lora_linear_matmul` | 无 adapter fast return、预分配 shrink buffer、kernel 与 matmul 各自布局 |
| 同文件 `add_lora_logits` | 使用 sampler indices，不应套用全部 forward token 的 indices |
| `vllm-ascend/vllm_ascend/lora/lora_ops.py`：`lora_linear` / `_try_lora_linear_triton` | 实际 custom-op 分派、捕获/设备/shape 约束、Triton split-K 数值路径 |
| `vllm/vllm/lora/layers/column_parallel_linear.py`：`_mcp_apply`、`MergedColumnParallelLinearWithLoRA`、QKV 相关派生类 | TP 的 B 输出切分、fully-sharded A rank 切分与 shrink all-gather、QKV 副本与 slice |
| `vllm/vllm/lora/layers/row_parallel_linear.py` | row-parallel A/B 切分、增量与 base 的归约位置；按实际 class 阅读 |
| `vllm-ascend/vllm_ascend/lora/fused_moe.py`、`quant_moe.py` | token/expert/adapter 对应关系、EP 交换、router weight、base-only 与有 LoRA 的量化 MoE 路径 |

当前 Ascend single-linear/packed wrappers 的替换条件通常包含 `max_loras==1` 和非 fully-sharded；
单 slot mask 优化还限制设备类型。`PunicaWrapperNPU.__init__` 的某些 kernel 选择受
hardware capability 及 `max_lora_rank>=128` 影响。因此容量调整可能改变数值路径，
但不能只凭 rank 阈值宣称整个 LoRA forward 一定走某一后端；继续跟踪 dispatch。

## 4. 请求、缓存与投机

| 路径与符号 | 要保存的身份或状态 |
| --- | --- |
| `vllm/vllm/lora/model_manager.py`：`set_adapter_mapping` / `_set_adapter_mapping` | mapping 与 `_last_slot_layout`；LRU 后相同请求 mapping 可能对应不同 device slot |
| `vllm/vllm/lora/punica_wrapper/punica_base.py`：`PunicaWrapperBase` | token/sampler metadata 与 prefill 分段，禁用 adapter 的行 |
| `vllm/vllm/v1/core/kv_cache_utils.py`：`_gen_lora_extra_hash_keys` | block hash 含 `("lora",name,path)`，不是文件内容版本 |
| `vllm-ascend/vllm_ascend/worker/model_runner_v1.py`：`_prepare_inputs` / `set_active_loras` | 按 scheduled/sample token 数设置实际 LoRA mapping |
| 同文件 `_determine_batch_execution_and_padding` / `_dummy_run` | graph batch descriptor 的 has_lora/num_active_loras 与 dummy capture，实际 replay 还需检查 |
| `vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py` | drafter graph dispatch 使用 has_lora 的位置；这本身不是 draft 模型已装 adapter 的证明 |
| `vllm-ascend/vllm_ascend/ops/gdn.py`：`AscendGatedDeltaNetAttention._forward_core` | projection 后的 conv、gating、recurrent/chunk、SSM state 读写；把 LoRA 输入错误与 state 错误分开 |

对 spec 每轮额外保存 draft IDs、target 验证输入位置、实际有效输出和 adapter slot。
若关闭 spec 也错，先追 LoRA/base/缓存首差；若仅 spec-on 错，按相同 adapted target 前缀
检查 verify 映射与状态回滚，不拿 draft 与 target logits 直接做等值断言。
