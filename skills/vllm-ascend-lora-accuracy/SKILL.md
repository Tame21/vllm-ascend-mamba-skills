---
name: vllm-ascend-lora-accuracy
description: 基于 vLLM 与 vLLM Ascend 源码定位 PEFT LoRA 推理精度问题，覆盖 adapter 装载、缩放和 packed 权重、TP/fully-sharded、多 adapter 串位、ACL Graph、prefix cache 及 LoRA 与投机/GDN 的组合；不把 DeepSeek MLA 内部低秩维度当作可加载 LoRA adapter。
---

# LoRA 精度验证与首差定位

将“LoRA 没生效、回答变差或混批串 adapter”定位到请求身份、装载映射、低秩增量、
状态复用或采样边界。先证明当前请求用了正确的 adapter，再比较模型数值。
不要把 API 装载成功、输出非空、两个 adapter 回答不同或 acceptance 高当作完整精度证明。

## 范围与运行事实

1. 在包含 `vllm/`、`vllm-ascend/` 的根目录工作，读取适用 `AGENTS.md`，
   保存 HEAD、工作区修改、实际 import 路径。源码导航基线为 vLLM
   `d61081dc3d3f1740a5d8bf82608b62974393c2de` 和 vLLM Ascend
   `9b8fc5d728e1ea54dc277562b296112d905e15c1`；变化后按符号重新定位。
2. 记录 base 权重/tokenizer revision、adapter 文件版本与哈希、模型架构、dtype/量化，
   CANN/torch/torch_npu、设备、TP/EP/PP、runner、graph、cache、spec method/K。
   另记 `max_loras`、`max_cpu_loras`、`max_lora_rank`、`lora_dtype`、
   `fully_sharded_loras`、`lora_target_modules` 与请求 LoRA name/id/path。
3. 本技能的 LoRA 是 `adapter_config.json` 与 A/B 权重所表示的适配器。
   DeepSeek MLA 的 `q_lora_rank` / `kv_lora_rank` 是 base 架构内部维度；
   修改它们既不是选择 adapter，也不是设置 `max_lora_rank`。
4. 从 [装载、计算与缓存源码图](references/code-map.md) 选择相关路径阅读；
   先确认实际替换的 wrapper/backend，不能由“安装了 Ascend”推定用了某个算子。

## 先排除 adapter 根本没有正确生效

- 对齐 base 模型、词表、模块维度和权重版本；`base_model_name_or_path` 是线索，
  不是 loader 已证明 base 与 adapter 匹配的证据。核对请求名是否真路由到该 adapter。
- 检查 checkpoint 的 A/B 成对、实际 shape/dtype、非零分布和完整模块名；
  不仅检查 `target_modules` 文本。区分 checkpoint targets 与服务端 `lora_target_modules` 过滤。
- 跟踪 `WorkerLoRAManager._load_adapter` → 模型 mapper → packed mapping →
  `LoRAModelManager.activate_adapter` → 各层 `set_lora`，记录应用到哪些层、TP/EP rank 与 slot。
  单个 rank 没有应用权重可能符合 PP/EP 分工，应检查全体相关 rank，而非立即下结论。
- `PEFTHelper` 此版本支持标准 `alpha/r` 和 rsLoRA `alpha/sqrt(r)`；
  `LoRALayerWeights.optimize()` 会原地将 scale 折入 B，再把 scaling 设为 1。
  因而 kernel 参数为 1 不代表没缩放；用折叠后的 B 再乘 alpha/r 会双重缩放。
- `PEFTHelper.from_dict` 会忽略未定义字段；存在 `rank_pattern` / `alpha_pattern`
  不等于这条路径支持逐层配置。检查实际消费链，遇到静默忽略先报告，不擅自改权重“兼容”。
  `use_dora`、非 none bias、普通生成模型的非受支持 `modules_to_save` 也需按当前校验确认。
- 区分磁盘 adapter rank 与按 `max_lora_rank` 分配的 buffer 容量；检查未使用 rank 行及
  未提供的 packed 子模块是否为零，尤其是 slot 复用后。不要将容量当作真实 rank。

## 建立 oracle 与控制组

固定相同 token IDs、chat template/BOS/EOS、停止条件和采样配置；先用 greedy 短输出。
跨后端非零 temperature 即使同 seed 也不能要求逐 token 相同，首差比较优先使用相同前缀 logits。
在已验证可用的 Transformers/PEFT 环境，以相同 base revision、adapter、dtype 和 eval 模式
建立 base-only 与 base+adapter oracle；先确认 PEFT 自身装载的是预期配置且 dropout 已关闭。
量化、融合和设备不同会改变数值，先用同一 NPU 的控制组隔离，再解释跨实现差异。

| 组 | 需要证明的行为 |
| --- | --- |
| base-only，未启用 LoRA / 启用 LoRA 但请求不带 adapter | wrapper 与调度本身没有错误增量 |
| base-only / 专门的零 B adapter | 无增量适配器不污染 base；不要修改用户原权重构造此组 |
| PEFT A / vLLM Ascend A，spec off、eager、单请求 | adapter A 的独立数值语义正确 |
| A、B、base 分别独立 / 同 batch 混排 | 每个请求等价于它自己的独立 oracle |
| A→B→base→A，含不同 rank、请求结束和 slot 复用 | 权重 reset、slot 与 token mapping 没有串位 |
| 相同 adapter，TP=1 / 故障 TP，fully-sharded 开/关 | 分片、gather/reduce、rank 对齐正确 |
| 相同组合，eager / 当前 ACL Graph | replay 使用实时 slot/mask/权重，而非捕获旧值 |
| 相同 adapter 冷/热前缀，再换 adapter | cache 复用保留身份隔离 |
| 相同 adapter spec off/on，再恢复 GDN 混合负载 | target 精度不被 draft/verify 或 state 回滚破坏 |

只运行能区分当前假设的组合，不需要整张矩阵全排列。
LoRA 增量可能不足以改变 argmax，因此“A 与 base 文本一样”不能证明没生效；
“A 与 B 文本不同”也不能证明每个请求的 adapter 都正确。
合并权重可作额外 oracle，但不能在已经 merge 的 base 上再次加载同一 adapter，
且 merge/未 merge 的舍入、量化路径差异须记录。

## 从首个 token 差异追到低秩增量

1. 按 request ID、逻辑 token 位置与相同前缀比较；不要按混批行号或 engine step 直接相减。
   保存首差 token IDs、target logits、top-k/margin、停止原因；前缀已不同后的误差不用于归首因。
2. 在相同层输入 X 上独立验证普通线性层：`ΔY = scale * (X @ A.T) @ B.T`。
   这里 A 为 `[rank,input]`，B 为 `[output,rank]`；已折叠 B 时 scale=1。
   FP32 显式计算可作算子参考，但阈值要考虑实际输入、累加与输出 dtype。
3. 同时比较 base 输出、shrink 中间结果、expand 增量和残差相加后输出；
   只比较总输出的相对误差会被大幅 base 分量掩盖错误 LoRA 增量。
4. QKV、gate/up、GDN projection 按真实 `packed_modules_mapping` 与 output slices 分别验证。
   检查 A/B 的维度、B 转置/块对角布局、slice offset 和缺失子模块；
   不把 GDN 的 fused qkvz 一概当作标准三段 QKV，也不盲目等分输出维度。
5. TP 下先对照每个 rank 的局部输入、A/B slice、局部增量，再跟 gather/reduce 后输出比较；
   fully-sharded 会改变 rank 维切分及通信，不能把局部 shrink 当完整 rank 中间结果。
   MoE 还需同时跟踪 token/expert/adapter 的 permutation 与 routing weight。
6. 报告 shape/stride/dtype、首差位置、最大绝对误差及增量误差；检查 NaN/Inf。
   不套统一容差，也不因为 cosine 很高就忽略少数异常 token 或被误修改的 base 行。

## 多 adapter、graph 与 prefix cache

核对 request name/id/path → `lora_index_to_id` → token/sampler indices → device slot；
LoRA ID 必须为正且稳定，同一 batch 含 base 时检查它对应的禁用行不读取其它 adapter 权重。
当前 manager 会同时比较 mapping 与 slot layout；排查 LRU 后相同 mapping 的 metadata 是否刷新。
单 slot 优化有 `_single_lora_mask` 和 packed 权重缓冲，检查 real/padded token 行、
base→adapter 切换和 adapter 重新装载后 buffer 是否更新，不能只证明首次 graph capture 成功。

当前 `_gen_lora_extra_hash_keys` 使用 LoRA name 与 path，不是 adapter 文件内容哈希。
同名同路径原地替换文件不能推定 prefix cache 失效；取证采用带版本的新路径或受控冷缓存，
并记录 prefix hit 与实际计算 token。涉及共享服务的 reset/reload 按已有授权和运行窗口执行。
混合模型还需跟踪 conv/SSM state 是否属于相同 adapter 与已验证前缀；
改变 adapter 后复用旧 recurrent state，即使 token prefix 一致也不满足数值等价。

## LoRA 与 spec/GDN 联合定位

先使相同 adapter 的 spec-off 基线通过，再打开 spec，分别记录 target 与 drafter 的
LoRA 配置/manager/模块；不能从 target 启用 LoRA 推定 drafter 自动适配。
上游 base 的 `hf_to_vllm_mapper` 删除 MTP 权重不等于 LoRA loader 也删除：
后者使用 rename-only mapper，并单独看 `lora_skip_prefixes`。
GDN 首差先比较 LoRA projection 输出，再比较 conv、gating、recurrent state；
投机失败还检查验证 token 的 adapter mapping、accepted counts、state 提交/回滚及下一轮恢复。
spec 一轮多个 token 需同 request/adapter 身份扩展；采样的 indices 与全部 forward token indices 不同。
acceptance 降低是性能线索，不能单独诊断 LoRA 精度；target 正确也不要求 draft logits 等于 target。

## 验证与交付

按 [实际测试断言与执行入口](references/tests-and-oracles.md) 选择最便宜的回归层级，
在实际 NPU 上复测故障形状、混批、graph 和必要组合；mock UT 只验证 Python 合约。
模型行为改动补真实 adapter 任务评测，不用合成随机 adapter 的路由测试代替质量验证。
输出版本/配置、adapter 身份链、最小复现、oracle、首差边界、修复与实测结果。
无 NPU/权重/PEFT 环境时完成静态核对和可执行步骤，明确待验证项，不声称已证明端到端精度。
