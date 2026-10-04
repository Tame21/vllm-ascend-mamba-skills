# vLLM-Ascend Mamba 投机推理 Skills

`Tame21/vllm-ascend-mamba-skills` 是一组结合 vLLM 和 vLLM-Ascend 实现编写的中文技能，面向 Mamba/GDN 等 recurrent 或 hybrid 模型的投机推理开发。五个技能可单独使用，主文件保留排查决策，命令、源码索引和细节放在各自 `references/`。

## 选哪个

| Skill | 适用问题 | 主要产出 |
| --- | --- | --- |
| [vllm-ascend-mamba-debug](vllm-ascend-mamba-debug/SKILL.md) | 启动失败、patch 未生效、不清楚实际 runner/drafter、reject 后异常 | 最小复现、真实调用链、首个失效边界 |
| [vllm-ascend-mamba-profile](vllm-ascend-mamba-profile/SKILL.md) | 采 profiling / prefill、TTFT 或投机吞吐回归 | NPU 采集命令、无采集基线、指标与时间线归因 |
| [vllm-ascend-mamba-accuracy](vllm-ascend-mamba-accuracy/SKILL.md) | spec/graph/chunk/cache/混批导致输出偏差 | token → target logits → recurrent state 首差与数值证据 |
| [vllm-ascend-mamba-state-cache](vllm-ascend-mamba-state-cache/SKILL.md) | 跨 block、前缀命中、拒绝回滚、重排或按层 KV 加载后出错 | 请求状态账本、copy/恢复契约和边界回归 |
| [vllm-ascend-mamba-dev](vllm-ascend-mamba-dev/SKILL.md) | 新功能、runner 迁移、上游升级或 kernel/metadata 修改 | 修改层级、接口契约、已有测试映射和验证计划 |

## 目录与使用方式

本仓库根目录直接包含五个技能目录。克隆仓库不会自动安装技能；可以让助手读取所需 `SKILL.md`，也可以按所用助手的安装方式安装完整技能目录，保留 `references/`、`scripts/`（如有）和 `agents/`。

源码检查和 NPU 测试还需要单独准备源码工作区：

```text
<源码工作区>/
├── vllm/
└── vllm-ascend/
```

向助手提供该工作区的实际路径。技能中的 `vllm/...`、`vllm-ascend/...` 是相对于源码工作区的路径；源码和 NPU 命令所说的“工作区根目录”也指此处。技能仓可以放在其它位置；读取技能文件的下列示例，以及离线比较器的调用，则以本技能仓根目录为起点。

## 可以直接交给助手的请求

从本技能仓根目录明确指定文件，例如：

> 请读取 `vllm-ascend-mamba-debug/SKILL.md`，结合这条启动命令和报错，确认 V2 runner、MTP speculator 和 NPU patch 是否实际生效，再做最小复现。

> 请读取 `vllm-ascend-mamba-profile/SKILL.md`，为 Qwen GDN+MTP 设计 prefill 的 NPU profiling，分别保留无采集基线和短窗口 trace，核实当前参数能否透传。

> 请读取 `vllm-ascend-mamba-accuracy/SKILL.md`，定位 spec off 正常、spec on 在第二个 chunk 后漂移的问题，先找同一前缀下的 target logits/state 首差。

> 请读取 `vllm-ascend-mamba-state-cache/SKILL.md`，检查 V2 align 模式 warm prefix hit 后请求重排和跨块 copy，给出状态索引与 connector 时序证据。

> 请读取 `vllm-ascend-mamba-dev/SKILL.md`，为本次 Mamba 投机功能改动选择实现位置，并扩展最小的 metadata、NPU state 和模型回归。

## 离线 dump 比较工具

精度技能包含 [compare_dumps.py](vllm-ascend-mamba-accuracy/scripts/compare_dumps.py)，用 Python 标准库读取 JSON/JSONL，按逻辑 key 比较：

- token/indices/counts 精确值；tensor 的显式绝对和相对容差。
- shape/dtype、缺失或多余记录、重复 key、NaN/Inf。
- 首差坐标、异常元素数和最大误差。

格式和采集边界见 [dump-format.md](vllm-ascend-mamba-accuracy/references/dump-format.md)。它是已采集小样本的比较器，不自动给模型加 hook，也不代表未采样张量通过验证。

## 来源与验证边界

编写日期：2026-10-04。代码快照：

- vLLM：`d61081dc3d3f1740a5d8bf82608b62974393c2de`
- vLLM-Ascend：`9b8fc5d728e1ea54dc277562b296112d905e15c1`

技能按实际源码符号、patch 注册链和测试断言编写，使用时仍需记录部署环境与当前 HEAD。快照不表示这两个版本或所有模型/并行/graph 组合已完成兼容性验证。

已进行 skill 格式校验、引用路径检查、独立故障场景试读，以及比较脚本的本地行为验证。文档中的 Linux NPU 命令是待执行模板；这些验证不包含模型推理、NPU 算子测试或 profiler 采集。
