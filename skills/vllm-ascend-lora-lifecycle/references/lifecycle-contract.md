# 源码链与切换实验

`V` 表示 vLLM 根，`A` 表示 vLLM-Ascend 根。符号适用于技能记录的本地快照，使用前按当前源码重定位。

## 生产者与消费者

| 路径 | 关键符号/职责 |
| --- | --- |
| `V/vllm/lora/request.py` | `LoRARequest`：正整数 ID、name equality/hash、load_inplace |
| `V/vllm/config/lora.py` | `LoRAConfig`：max_loras/max_cpu_loras、dtype、rank、target_modules、图相关配置 |
| `V/vllm/entrypoints/serve/lora/api_router.py` | `attach_router`：动态路由受 `VLLM_ALLOW_RUNTIME_LORA_UPDATING` 控制 |
| `V/vllm/entrypoints/openai/models/serving.py` | `OpenAIServingModels`：按 name 加锁的 load/unload/resolve 和 registry |
| `V/vllm/lora/worker_manager.py` | `WorkerLoRAManager._load_adapter`、`LRUCacheWorkerLoRAManager.add_adapter`：校验/CPU 加载/激活 |
| `V/vllm/lora/model_manager.py` | `activate_adapter`、`set_adapter_mapping`、`remove_adapter`、LRU/pin；权重与 slot layout |
| `V/vllm/lora/punica_wrapper/utils.py` | `convert_mapping`：整数 ID → 设备 slot，base 及 sampler indices |
| `V/vllm/v1/worker/lora_model_runner_mixin.py` | `load_lora_model`、`_set_active_loras`、`set_active_loras`；不要按函数注释猜真实 prefill |
| `V/vllm/v1/worker/gpu/lora_utils.py` | `LoraState`：V2 持久槽位与本轮请求映射；capture/dispatch helpers |
| `V/vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule`：scheduled_loras 与 max_loras admission |
| `V/vllm/v1/core/kv_cache_utils.py` | `_gen_lora_extra_hash_keys`：name+path cache identity |
| `A/vllm_ascend/platform.py` | `get_punica_wrapper`：选 `PunicaWrapperNPU` |
| `A/vllm_ascend/worker/worker.py` | `NPUWorker.add_lora/remove_lora/list_loras/pin_lora`：转交 runner |
| `A/vllm_ascend/lora/punica_npu.py` | `update_metadata`、`_update_base_metadata`、单槽 mask、no_lora |
| `A/vllm_ascend/lora/utils.py` | `refresh_all_lora_classes`、packed/transposed weights 的 `set_lora/reset_lora` |
| `A/vllm_ascend/lora/fused_moe.py` | `prepare_lora_indices/preprocess_lora_indices/all2all_lora_indices/postprocess_lora_indices` |

## 可以手算的重排例子

```text
请求: rA -> adapter ID 11，rBase -> None，rB -> adapter ID 22
初始 lora_index_to_id: [11,22]
本轮 batch 顺序: [rB,rBase,rA]；scheduled tokens: [2,1,3]
token adapter IDs: [22,22,0,11,11,11]  # 0 仅表示内部 no-adapter mapping
kernel slot indices: [1,1,-1,0,0,0]

LRU 后 lora_index_to_id 变为 [22,11]，即使 token adapter IDs 不变：
kernel slot indices 必须变为 [0,0,-1,1,1,1]
```

base 的内部 ID sentinel 在不同生产者可能是 0 或负值；当前 `convert_mapping` 把非正值映射为 -1。别把这个规则扩展到 `LoRARequest` 的合法 ID。

更新 API 登记表、加载 CPU 权重、激活设备槽位和刷新 mapping 是四个不同的成功条件。snapshot 记录应含 step、rank、request ID、name、整数 ID、path/version、slot layout、token/sample mapping、真实/padded 行数。

## 在隔离开发服务复现加载/替换

复用用户已经工作的 base 模型和 LoRA 启动命令。静态注册示例追加参数：

```bash
--enable-lora --max-loras 2 --max-cpu-loras 3 \
--lora-modules '{"name":"adapter-a","path":"/models/adapter-a-v1"}' '{"name":"adapter-b","path":"/models/adapter-b-v1"}'
```

这是启动参数片段，不是独立 shell 命令；rank 上限、dtype、并行配置按实际 adapter 设置。客户端 `model` 用 `adapter-a` 或 `adapter-b`，路径必须由服务端 worker 访问。

只有任务涉及动态更新且目标是选定的开发实例时，才在该实例启动前启用 `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`；当前路由源码明确将其用于本地开发。普通精度/性能实验优先用静态注册。

请求体保存为文件，先核对路径/name，再发给隔离服务：

```json
{"lora_name":"adapter-a","lora_path":"/models/adapter-a-v2","load_inplace":true}
```

```bash
LORA_TEST_URL=http://127.0.0.1:8080
curl --fail-with-body "$LORA_TEST_URL/v1/models"
curl --fail-with-body "$LORA_TEST_URL/v1/load_lora_adapter" \
  -H 'Content-Type: application/json' --data-binary @reload-adapter.json
```

确认请求已完成，再比对该 adapter 新请求与新权重单独启动的基线。同路径覆盖只适合专门验证缓存失效语义；普通换版本对照使用新路径，不能仅通过改变请求 seed 掩盖旧 KV 复用。

卸载的 body 格式是 `{"lora_name":"adapter-a"}`，路由为 `/v1/unload_lora_adapter`。当前 handler 只删 API 映射，worker cache 的移除与内存回收需要另查，不能用这个接口成功作为内存释放证明。多 API server 时 name lock 是各进程内对象，不应宣称提供跨进程权重版本切换屏障。

## 现有回归入口

以下路径真实存在，但实际运行仍需要当前仓库依赖与设备；上游的 `DEVICES`、dist fixture 和 CUDA kernels 不能直接当作 NPU 验证。

| 测试 | 实际用途与限制 |
| --- | --- |
| `V/tests/lora/test_lora_manager.py::test_set_adapter_mapping_refreshes_after_slot_reassignment` | mapping 相同、物理槽位互换后重建 Punica metadata |
| `V/tests/lora/test_lora_manager.py::test_lru_cache_worker_adapter_manager` | worker LRU/加载/替换生命周期；按真实断言扩展失败不破坏旧 adapter |
| `V/tests/lora/test_lora_manager.py::test_lora_lru_cache_model_manager` | device slot 激活/置换/pin |
| `V/tests/v1/core/test_kv_cache_utils.py::test_lora_path_change_changes_block_hashes` | 同名不同路径 prefix hash 不同，不证明同路径内容变更能失效 |
| `V/tests/entrypoints/serve/lora/test_lora_adapters.py` | registry、load_inplace、动态并发 API；测试依赖服务和模型 |
| `A/tests/ut/lora/test_utils.py` | 仅验证 Ascend wrapper 注册优先级/类名；不覆盖 packed 权重 set/reset |
| `A/tests/ut/lora/test_lora.py` | no_lora、单槽 mask、MoE 多 adapter 与 base mapping |
| `A/tests/ut/lora/test_punica_npu.py` | wrapper 路由及 sampler indices，不等于真实 NPU 数值 |
| `A/tests/e2e/pull_request/one_card/lora/test_qwen3_multi_loras.py` | 查看实际 LoRA 文件和 ID 分配，再判断多权重隔离覆盖 |
| `A/tests/e2e/pull_request/two_card/lora/test_qwen3moe_lora.py::test_qwen3moe_lora_multi_id_ep` | 多 ID 与 base 混批；两个 ID 指向同一份文件，base 用负例断言 |
| `A/tests/e2e/pull_request/two_card/lora/test_qwen35_4b_multi_lora_mtp_aclgraph.py` | adapter 名字存在、顺序生成非空且文本不同；不是混批或严格精度 oracle |

在 Ascend 仓库根、使用已经配置好的虚拟环境执行所选 UT，例如：

```bash
.venv/bin/python -m pytest -q tests/ut/lora/test_utils.py tests/ut/lora/test_lora.py tests/ut/lora/test_punica_npu.py
```

针对本次问题增加的模型验证应是：每个 adapter 单独运行 vs 同 adapter 在混批/置换后的同前缀结果；不要仅断言 A/B 文本不同。权重、tokenizer、fixtures 未就绪时先静态读测试，`--collect-only` 也可能触发模块级模型/tokenizer 加载。

packed 权重的热更新/清零需要新增定向回归：实际调用 wrapper `set_lora/reset_lora`，检查普通与 packed A/B 同时更新、缺失子模块归零，再验证复用 slot 的输出。现有 `test_lora.py::test_packed_single_lora_matmul` 使用已构造好的 packed tensors，不能替代此生命周期检查。
