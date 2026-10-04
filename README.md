# vLLM-Ascend 开发 Skills

`Tame21/vllm-ascend-skills` 收录十个中文技能，结合 vLLM 与 vLLM-Ascend 源码，支持 Mamba/GDN 投机推理和 PEFT LoRA adapter 的开发、排障、精度定位与性能分析。每个技能可单独使用，详细命令、源码索引和测试覆盖说明放在对应的 `references/` 中。

## Mamba/GDN 投机推理

| 技能 | 用途 |
| --- | --- |
| [mamba-spec-debug](skills/vllm-ascend-mamba-spec-debug/SKILL.md) | 定位启动、patch、runner/drafter 与 reject 后异常，建立最小复现和实际调用链。 |
| [mamba-spec-profile](skills/vllm-ascend-mamba-spec-profile/SKILL.md) | 采集 prefill / 投机推理的 NPU profiling，分析 TTFT、吞吐和时间线。 |
| [mamba-spec-accuracy](skills/vllm-ascend-mamba-spec-accuracy/SKILL.md) | 对齐 token、target logits 与 recurrent state，定位 spec/graph/chunk/混批引起的首差。 |
| [mamba-spec-state-cache](skills/vllm-ascend-mamba-spec-state-cache/SKILL.md) | 排查跨块状态拷贝、前缀命中、拒绝回滚、请求重排与 KV 加载时序。 |
| [mamba-spec-dev](skills/vllm-ascend-mamba-spec-dev/SKILL.md) | 为功能开发、runner 迁移和上游升级选择实现层级、接口检查与回归测试。 |

## LoRA adapter

| 技能 | 用途 |
| --- | --- |
| [lora-accuracy](skills/vllm-ascend-lora-accuracy/SKILL.md) | 核对 adapter 身份、A/B/scale 与低秩增量，定位 TP/混批/graph/spec 后精度异常。 |
| [lora-profile](skills/vllm-ascend-lora-profile/SKILL.md) | 区分冷加载、槽位切换和稳态计算，分析 prefill/decode 与多 adapter 性能。 |
| [lora-kernel](skills/vllm-ascend-lora-kernel/SKILL.md) | 联合验证 BGMV/SGMV、packed matmul、Triton 路径的布局、数值契约与性能。 |
| [lora-lifecycle](skills/vllm-ascend-lora-lifecycle/SKILL.md) | 跟踪 name → ID → slot → token 映射，排查热更新、LRU、混批串扰与缓存隔离。 |
| [lora-dev](skills/vllm-ascend-lora-dev/SKILL.md) | 为新增模型、target module、TP/EP、量化、MTP 和 graph 组合设计实现与回归。 |

## 目录与使用方式

十个完整技能位于本仓库的 `skills/` 下：

```text
vllm-ascend-skills/
├── README.md
└── skills/
    ├── vllm-ascend-mamba-spec-debug/
    ├── vllm-ascend-mamba-spec-profile/
    ├── vllm-ascend-mamba-spec-accuracy/
    ├── vllm-ascend-mamba-spec-state-cache/
    ├── vllm-ascend-mamba-spec-dev/
    ├── vllm-ascend-lora-accuracy/
    ├── vllm-ascend-lora-profile/
    ├── vllm-ascend-lora-kernel/
    ├── vllm-ascend-lora-lifecycle/
    └── vllm-ascend-lora-dev/
```

克隆仓库不会自动安装技能。可以让助手直接读取所需 `SKILL.md`，也可以按所用助手的安装方式安装完整技能目录，保留 `references/`、`agents/` 和 `scripts/`（如有）。

源码检查和 NPU 测试需要另行准备源码工作区，并向助手提供实际路径：

```text
<源码工作区>/
├── vllm/
└── vllm-ascend/
```

技能中的 `vllm/...`、`vllm-ascend/...` 和源码命令所称的“工作区根目录”指这个双仓工作区；`V`、`A` 分别指两个源码仓根目录。若源码分开存放，先定位各自路径再调整命令。本技能仓可以放在其它位置，下列调用示例及比较器命令以本技能仓根目录为起点。

```text
请读取 skills/vllm-ascend-mamba-spec-debug/SKILL.md，结合启动命令和报错，
确认 V2 runner、MTP speculator 和 NPU patch 是否实际生效，再建立最小复现。

请读取 skills/vllm-ascend-mamba-spec-accuracy/SKILL.md，定位 spec off 正常、
spec on 在第二个 chunk 后漂移的问题，先找同一前缀下的 target logits/state 首差。

请读取 skills/vllm-ascend-lora-profile/SKILL.md，分析 max_loras 从 1 改成 4
后 decode 变慢的问题，区分冷加载、暖机、graph 与稳态计算，并给出 NPU 采集方案。
```

## 离线 dump 比较器

[compare_dumps.py](skills/vllm-ascend-mamba-spec-accuracy/scripts/compare_dumps.py) 仅依赖 Python 标准库，按逻辑 key 比较 JSON/JSONL 小样本，支持整数精确比较、浮点容差、shape/dtype 检查、缺失记录和 NaN/Inf 检查，并报告首差与误差统计。

在 Linux 环境将 `$PYTHON` 设为已确认可用的 Python 3 解释器，从本技能仓根目录运行，替换两个 dump 路径：

```bash
"$PYTHON" skills/vllm-ascend-mamba-spec-accuracy/scripts/compare_dumps.py ref.jsonl candidate.jsonl --atol 0.001 --rtol 0.001
```

`0.001` 仅演示参数语法，不是推荐容差。记录格式、采集要求和退出码见 [dump-format.md](skills/vllm-ascend-mamba-spec-accuracy/references/dump-format.md)。比较器不自动添加模型 hook，也不证明未采样张量或端到端推理正确。

## 代码快照与验证边界

编写基线日期：2026-10-04。

- vLLM：`d61081dc3d3f1740a5d8bf82608b62974393c2de`
- vLLM-Ascend：`9b8fc5d728e1ea54dc277562b296112d905e15c1`

使用时核对实际 HEAD、未提交修改、包导入路径与设备环境；版本变化后按符号重新定位实现。源码快照不表示所有模型、并行、量化和 graph 组合均已验证兼容。

已进行技能格式校验、引用路径检查、独立故障场景审阅及比较器的本地行为验证。未运行模型推理、NPU 算子测试或 profiler 采集，也未实测加速比。文档中的 Linux NPU 命令是待执行模板，需要按部署环境调整模型路径、解释器、设备规模与输出目录。
