---
name: vllm-ascend-lora-lifecycle
description: 排查或开发 vLLM-Ascend 多 LoRA 的加载、热切换、CPU/设备缓存、adapter 路由和前缀缓存隔离。适用于 adapter 不生效、换权重后仍输出旧结果、混批串 adapter、LRU 置换后精度或尾延迟异常。
---

# LoRA 生命周期、路由与缓存

证明一个请求实际使用了哪份 adapter 权重，以及从名称解析到 kernel 槽位的映射是否一致。不要以 HTTP 成功、模型列表存在或 adapter 文本不同作为完整证明。

## 固定版本和身份

定位 vLLM 根目录 `V`、vLLM-Ascend 根目录 `A`，读取待修改目录适用的 `AGENTS.md`。本技能的源码快照为 2026-10-04：vLLM `d61081dc3d3f1740a5d8bf82608b62974393c2de`，Ascend `9b8fc5d728e1ea54dc277562b296112d905e15c1`。使用时重新核对 HEAD、dirty 状态、实际导入包位置和运行的 V1/V2 runner。

本技能的 LoRA 是 PEFT adapter；模型里的 MLA `q_lora_rank`、`kv_lora_rank` 是另一类低秩结构，不能当作已启用 adapter。

给每份权重记录 base 模型/revision、adapter 路径/revision或内容指纹、name、正整数 ID、rank、dtype、target_modules。保持 name/ID 与权重版本关系明确；base 请求使用 `lora_request=None`，不要构造 ID=0 的 `LoRARequest`。

## 先画四级映射

| 层次 | 观察项 | 不能混淆的含义 |
| --- | --- | --- |
| API | 请求 `model`、`lora_name`、`lora_path` | 注册名称与实际服务端路径；`/v1/models` 只反映 API 登记 |
| 请求/调度 | request ID、`lora_int_id`、本轮 scheduled 请求 | adapter ID 与 batch row 不相同 |
| worker manager | registered adapters、active adapters、`lora_index_to_id` | CPU 缓存 ID 与设备物理槽位不相同 |
| Punica/kernel | token/sampler indices、base mask、真实/padded tokens | kernel 消费槽位索引；内部 `-1` 表示无 LoRA，不是可加载 ID |

`LoRARequest` 当前按 name 实现相等和 hash，worker cache 通常按 ID；不能用同一个 name 配多份不同 ID/权重来构造独立 adapter 的 set。
`max_loras` 限制一个 batch 可用的 adapter 数并决定设备槽位数；`max_cpu_loras` 是 worker CPU 注册缓存容量，必须不小于前者；API 注册总数和活跃数不能互换。

源码入口、HTTP 与离线复现示例见 [references/lifecycle-contract.md](references/lifecycle-contract.md)，仅按当前故障读取对应部分。

## 按症状追踪

### 加载成功但行为像 base

1. 检查请求 `model` 是否使用 adapter 的服务名称；离线生成是否传了对应 `LoRARequest`。
2. 沿 `OpenAIServingModels` → engine add → `LRUCacheWorkerLoRAManager` → `LoRAModelManager.activate_adapter`，查看真实 ID、slot 和应用模块数。
3. 对齐模型 `supported_lora_modules`、packed mapping、HF rename mapper、skip prefixes、部署 `target_modules` 和 checkpoint keys。
4. `activate_adapter` 没有找到某层权重时会 `reset_lora(slot)`；所有层都没应用时也不必然抛错。PP/EP 某个 rank 没有本地目标模块可以合理，需结合其它 rank 判断。
5. 选择一个应非零的层确认 A/B 和输出 delta；adapter 存在但零权重、被 mask、未覆盖目标层，都可能产生 base 结果。

### 换路径/换权重后仍是旧输出

- 离线相同 ID 默认命中已加载 adapter，不会因换了 path 自动重载；读取 `load_inplace` 和当前 worker 实现。
- 在线同 name 默认重复加载会报错；显式 `load_inplace=True` 才替换，并保留原整数 ID。这和离线 cache 命中返回不是同一种行为。
- 当前 loader 先加载验证新 adapter，成功后才移除旧项并按容量置换；验证失败后检查旧 adapter 仍可用。
- 当前 prefix block hash 额外包含 `("lora", name, path)`，不是文件内容 hash。改变同一路径中的权重不改变这个 cache identity。
- serving 对同路径 reload 会提示旧 prefix KV 未失效。优先用不同版本路径做隔离复现；若任务确需原地更新，按当前已支持的缓存重置/请求生命周期机制验证，不凭 reload 成功推定 KV 已清空。
- 不把正在生成的请求混入版本替换精度对照；先在隔离测试中固定请求完成边界，再另测用户确需的并发更新语义。

### 只在混批、切换或 LRU 后串位

- 为每个 step 保留 request ID → adapter ID → slot → token 范围，按真实 batch 重排更新。
- V1 的 token mapping 与 sample/prompt mapping 长度不同，投机验证还可能有多 logits 行；V2 通过 `LoraState.make_lora_inputs(req_ids, idx_mapping, num_scheduled_tokens)` 构造映射。按当前调用者核对，不能把 V1 的重复次数照抄到 V2。
- 即使 batch 的 `LoRAMapping` 完全没变，旁路 `add_lora` 也可能置换设备槽位。当前 `set_adapter_mapping` 同时比较 mapping 与 slot layout，必须保留这个刷新条件。
- slot 复用时查普通 A/B、packed A/B、遗漏模块是否一起清零/重写；新 adapter 不覆盖的模块不能残留上一份 delta。
- base 行、padding 行和 no-LoRA 整批分别检查。当前 `PunicaWrapperNPU.update_metadata` 会刷新 `no_lora`，单槽 mask 依赖实际 token slot。
- LoRA + MoE 的 token ID 随 dispatch/repeat/permutation/AlltoAll 一起移动；不能仅验证 dense 部分的 mapping。

### 卸载后内存没降或首请求很慢

- 当前 `OpenAIServingModels.unload_lora_adapter` 删除 API name 登记，未直接调用 engine `remove_lora`；不能据此证明 worker 权重已立即释放。
- `list_loras()` 的 worker 注册集合、active LRU、API models 列表分别观察。设备预分配的 stacked buffers 即使槽位释放也可能保留。
- 区分从文件加载、CPU cache 命中后的设备激活、热 adapter 计算和调度等待。`max_loras` 满时 scheduler 会暂跳不能加入的新 adapter 请求，低设备利用率不必然是 kernel 慢。
- pin 会影响 CPU/设备两级缓存置换，不是无条件的性能修复；用真实 working set 和各 adapter 延迟验证。

## 图与算子分支的生命周期约束

观察实际 wrapper 类、Punica 对象和被调用 kernel。`AscendPlatform.get_punica_wrapper` 选择 NPU wrapper，其构造会调用 `refresh_all_lora_classes` 注册 Ascend 层替换。

`max_loras=1` 是单槽优化的配置门槛，不等于当前只有一个 active adapter；改变它可能同时改变 packed weights、kernel 和 graph。容量实验不得当成同 kernel 的纯缓存实验。

上游 `_set_active_loras` 当前固定 `LoRAMapping.is_prefill=True` 以使用非 CUDA 的 SGMV 路径；该标志和算子名都不等同于请求真正处于 prompt prefill。

图重放检查 metadata、mask 与权重 buffer 是否更新到捕获地址；比较相同图下 A → base → B → A，而不是只跑首次捕获。dummy adapter/capture key 中的无 LoRA 情况不能解释为真实多加载了一份权重。

## 回归与交付

最小样例用两份已知不同的 adapter A/B、base，以及必要的第三份 C 触发 LRU。先建立各自单独运行基线，再检查不等长混批、重排、结束/加入、置换后重激活和图重放。单独一组不同 ID 指向同一份文件只能验证部分路由，不能证明不同权重的隔离。

定向扩展现有 mapping/LRU/packed reset UT；真实 NPU 上比较同请求同前缀下的 tokens/logits/delta，再测缓存切换延迟。参考中的模型测试有不同强度的断言，按实际覆盖补缺口。

交付应包含：错误的映射或生命周期边界、身份/版本账本、复现序列、最小改动、实际测试和精度/延迟结果。无 NPU 时区分静态结论、CPU/mock 验证和待执行命令。
