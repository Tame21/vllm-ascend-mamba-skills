# 状态契约、源码与测试

`V` 表示 vLLM 仓库根，`A` 表示 vLLM-Ascend 仓库根。表中是 2026-10-04 本地快照的相对路径；优先按符号查找，不依赖行号。

## 所有路径都必须识别的契约

| 源码 | 关键符号与用途 |
| --- | --- |
| `V/vllm/v1/kv_cache_interface.py` | `MambaSpec` 的 shapes/dtypes/mamba_type、page size、`num_speculative_blocks`；`UniformTypeKVCacheSpecs`；checkpoint helpers |
| `V/vllm/v1/core/single_type_kv_cache_manager.py` | `MambaManager` 分配、copy-on-write、prefix hit；和 scheduler 的 split 联合阅读 |
| `V/vllm/v1/core/sched/scheduler.py` | `Scheduler._mamba_block_aligned_split`、scheduled/spec token 计数 |
| `V/vllm/model_executor/layers/mamba/mamba_utils.py` | 模型提供的 state copy specs；不要从名称猜测 SD/DS 布局 |
| `V/vllm/v1/worker/mamba_utils.py` | `MambaCopyBuffers`、`MambaSpecDecodeGPUContext`、`preprocess_mamba_align_fused_kernel`、`run_fused_postprocess_align` |
| `V/vllm/v1/worker/gpu/model_states/mamba_hybrid.py` | `MambaHybridModelState.add_request/preprocess_state/postprocess_state` 和 accepted scatter/fill kernels |
| `A/vllm_ascend/patch/platform/patch_mamba_config.py` | `verify_and_update_config`：KV store / prefix / sparse page 参数解析 |
| `A/vllm_ascend/patch/platform/patch_mamba_block_aligned_split.py` | `_mamba_block_aligned_split`：PD verifier window 与 producer/standalone checkpoint 分支 |
| `A/vllm_ascend/patch/platform/patch_mamba_manager.py` | `AscendMambaManager`：Mamba DCP lookup 与外部同步加载容量 |
| `A/vllm_ascend/patch/worker/patch_mamba_utils.py` | V1 preprocess、NPU copy、310P fallback、按层 copy staging；实际 patch 赋值在文件末尾附近 |
| `A/vllm_ascend/worker/v2/model_states/mamba_hybrid.py` | `AscendMambaHybridModelState`：覆写 preprocess/prepare_attn，继承其它生命周期方法 |
| `A/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py` | `AscendStoreConnector.prepare_mamba_state_copy/wait_for_layer_load/finish_mamba_state_copy`：接管、load 后执行 copy、清理 |
| `A/vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py` | `AscendMultiConnector` 同名方法：所有参与的 child connector 完成本层 load 后执行 copy |
| `A/vllm_ascend/worker/v2/attn_utils.py` | `unwrap_mamba_kv_cache_groups`、`_reshape_mamba_kv_cache`；page-strided state views |
| `A/vllm_ascend/ops/triton/mamba/postprocess.py` | `postprocess_mamba_fused_kernel`：独立 group block tables、计数、copy、SD/DS、64-bit 地址 |
| `A/vllm_ascend/ops/triton/v2/mamba/precopy.py` | `precopy_mamba_align_fused_kernel`；先核实运行时实际绑定，文件存在不等于已使用 |
| `A/vllm_ascend/ops/triton/mamba/state_index.py` | `gather_ssm_states`、`scatter_ssm_states_`，GDN fused prefill 的物理状态行存取 |

注册链还要查 `A/vllm_ascend/patch/platform/__init__.py`、`A/vllm_ascend/patch/worker/__init__.py`。当前 precopy 的实际替换在 `A/vllm_ascend/patch/worker/patch_v2/patch_triton.py`，由 worker 的 `HAS_TRITON` 分支导入。模块属性替换与其它模块的 `from ... import ...` 早期绑定不是同一件事。对版本注释和真实赋值不一致的地方，应查真实调用者或记录运行时绑定。

## 计数账本

### V2 forward 前的 align 决策

来自 `V/vllm/v1/worker/mamba_utils.py:preprocess_mamba_align_fused_kernel`。
只对真实活动 batch 行 `r < num_reqs` 应用；`s = idx_mapping[r]` 必须是合法、互异的非负请求槽位，
`B > 0`，设备 computed 为当前已计算 token 数，`q = query_start_loc[r+1] - query_start_loc[r]`
为本行实际 scheduled query 长度。真实 forward 行应有 `q > 0`；padding 不放入这些活动行。

```text
src_col[s] = old_state_idx[s]
src_off[s] = max(old_accepted_for_state[s] - 1, 0)
new_state_idx[s] = (computed[s] + q + B - 1) // B - 1
state_idx[s] = new_state_idx[s]
若 old_state_idx[s] >= 0 且 old_state_idx[s] != new_state_idx[s]:
    accepted_for_state[s] = 1
否则保留旧 accepted_for_state[s]
```

这里的 active mask 仅为 `r < num_reqs`，没有再屏蔽 `s < 0`。
postprocess 的 `-1` filtered-row sentinel 不能泛化到 preprocess 或 `prepare_attn` 的直接 gather；
先查调用者在哪个阶段引入 sentinel。新请求的 accepted 初始化为 1；继承的 `add_request`
按 `(new_req_data.num_computed_tokens - 1) // cache_config.block_size` 播种 state index，
因此 fresh prefix 为 0 时源列为 -1。核对该 block size 与实际 Mamba spec 的解析结果。

后续 precopy 遇到 `src_col < 0` 或 `src_col == dst_col` 均不拷贝；同列时保留 accepted 偏移供
forward 的恢复路径使用。跨列 copy 使用重置前已保存的 `src_off`，再让 forward 读取中性计数 1。
这与下面 postprocess 的“同列但 bias 非零仍需复制”不同，不可合并两套条件。
按层模式中这些决策先执行，数据拷贝由 connector 延迟至该层 load 后；下一次非空 preprocess
的 `_finish_previous_layerwise_mamba_copy` 检查 pending layers，不代表一次 Python hook 返回就证明 NPU 操作已完成。

### 采样后的 checkpoint 决策

下面公式来自当前 Ascend `postprocess_mamba_fused_kernel`，不是所有 sampler 的通用公式：

```text
V1: running = computed_before + scheduled - draft
    computed_after = running + accepted_for_state - 1
V2: computed_after 由调用者传入已经推进的设备计数
    running = computed_after - accepted_for_state + 1
aligned = floor(computed_after / B) * B
needs_copy = aligned >= running
若 needs_copy:
    bias = aligned - running
    dst_col = aligned / B - 1
```

`accepted_for_state` 应追溯到本轮 sampler 和 state postprocess 的转换。V2 `_scatter_num_accepted_kernel` 使用 `max(num_sampled, 1)`；pre-copy/post-copy 又可能把恢复偏移归一为 1。它不能直接拿去统计 draft acceptance。

若 `src_col == dst_col`，kernel 有 accepted 输出重置逻辑；仅在同时 `bias==0` 时无需复制数据。当前上游 `run_fused_postprocess_align` 先把 accepted 拷到独立 snapshot 再作为 kernel 输入，输出写原 buffer，避免多 state CTA 读写竞争。移植旧版本时必须核实这份输入/输出契约。

## 寻址与 copy

- Mamba group 分别分配 physical blocks；同一个逻辑列在不同 group 的物理 ID 不要求相同。
- V2 `HAS_IDX_MAPPING=True` 时，计数/运行 state index 按 `req_idx`，block table 行按 `batch_idx`；只对 batch 有效行做边界检查。
- conv 复制的是带 `bias` 的窗口，SD 与 DS 的时间维不同；DS 要尊重逐 dim row 的 stride。
- temporal 的源列是 `src_col+bias`，复制有效数据 `state_inner_size * element_size`。物理 page stride 可能包含 padding，不能当作 copy size。
- block ID 参与字节地址乘法前转换为 int64；不能先 int32 乘完溢出再转换。
- `gather_ssm_states` 在 `has_initial_state=False` 时生成零状态；scatter 的目标物理行需唯一，否则存在并行写冲突。验证未选择行和 page guards 不变。
- 对纯数据搬运按相同 dtype conversion 后逐元素/字节相等检查；不要用宽松浮点误差掩盖地址错位。

## 回归选择

以下命令从相应仓库根执行；使用准备好的依赖环境。vLLM 按仓库要求使用 uv 管理的 `.venv/bin/python`；Ascend 示例同样以该虚拟环境入口表示。

```bash
# A：CPU/mock 的状态、布局、调度分支回归
.venv/bin/python -m pytest -q tests/ut/worker/test_model_runner_v2_mamba.py
.venv/bin/python -m pytest -q tests/ut/distributed/ascend_store/test_ascend_store_connector.py
.venv/bin/python -m pytest -q tests/ut/patch/worker/test_patch_mamba_utils.py tests/ut/patch/worker/test_patch_mamba_utils_uniform_groups.py
.venv/bin/python -m pytest -q tests/ut/patch/platform/test_patch_mamba_block_aligned_split.py tests/ut/patch/platform/test_patch_mamba_config.py

# A：真实 NPU 的拷贝/地址/graph replay 结果
.venv/bin/python -m pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_postprocess_mamba.py
.venv/bin/python -m pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_precopy.py
.venv/bin/python -m pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_postprocess_temporal_copy.py
.venv/bin/python -m pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mamba_state_index.py

# V：上游 checkpoint/split 契约；不能据此声称 NPU 兼容性
.venv/bin/python -m pytest -q tests/v1/core/test_mamba_align_chunk_split.py tests/v1/core/prefix_cache/test_mamba_eagle_resume_checkpoint.py
```

按层场景优先查看 `TestAscendStoreConnectorLayerwise` 下的
`test_v2_mamba_state_copy_runs_after_layer_load`、`test_mamba_state_copy_not_deferred_after_finish`、
`test_non_layerwise_connector_keeps_batched_mamba_copy`。这些验证 hook 顺序/清理和分支，
需要真实异步 load 实验才能证明 stream/event 顺序正确。

当前 `test_precopy_matches_reference` 虽参数化 `HAS_IDX_MAPPING`，实际 mapping 是 identity，
并且只使用一个 Mamba group；不能据此声称请求重排、稀疏持久槽位、多 group 独立物理 ID、
padded page 或连续 graph replay 已覆盖。涉及这些契约时在该 suite 补对应输入和未选区域 guards。
postprocess 可参考 `test_mixed_postprocess_replay_uses_updated_metadata`，其已有重排、`-1`、
padded page 和重复 replay 的变化输入；它不替代 preprocess、按层 load 或模型级恢复验证。

先阅读对应测试实际 import、fixture、skip 条件，再决定本机可运行部分。`A/tests/ut/conftest.py` 的 CPU mocks 只说明 UT 的设计，不保证任意 Windows Python 环境都可直接运行。`A/tests/e2e/pull_request/two_card/model_runner_v2/test_mamba_hybrid.py` 是可参考的模型启动例子，但当前只断言输出非空，不是严格的精度回归。
