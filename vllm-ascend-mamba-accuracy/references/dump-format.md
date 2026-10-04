# 离线首差比较

`scripts/compare_dumps.py` 面向已按逻辑位置对齐的小规模取证快照，只用 Python 标准库。
它不采集设备张量、不推断模型布局，也不自动决定精度阈值。
默认浮点零容差，整数 token/count/index 用 `kind=exact`，不会被浮点容差掩盖。

## 记录格式

`.json` 为单个对象或对象数组；`.jsonl` 每个非空行一个对象。
每条记录包含 `key`、`kind`、`shape`、`dtype`、`data`：

```json
[
  {"key":"r0/pos12/token","kind":"exact","shape":[1],"dtype":"int64","data":[42]},
  {"key":"r0/pos12/target/layer8/ssm-before","kind":"tensor","shape":[2,2],"dtype":"bfloat16","data":[1.0,2.0,3.0,4.0]}
]
```

- `key` 在每个文件中唯一；两侧按 key 配对，所以调度批次行顺序不同仍能比较。
  建议编码 request ID、逻辑 token 位置、target/draft、层和更新阶段；缺失/额外 key 都报告差异。
  相同逻辑位置可能被重算，若需要保存多次调用，用显式 occurrence 标记并确认两侧语义对应。
- 记录按参考运行的逻辑先后导出；脚本按参考文件记录顺序报告首差，不做 key 字符串排序。
  两个 engine step 不对应时，由采集者先关联已验证前缀；不要把 wall-clock 时间当 key。
- `data` 是 row-major 展平数值；`shape=[]` 表示含一个值的标量；零维度表示空张量。
  shape 与数据元素数不一致属于输入错误；维度/布局不同先转换到相同逻辑语义。
- `dtype` 默认必须一致；需要 FP32 oracle 对 BF16 candidate 时显式使用 `--ignore-dtype`，
  并在报告保留两侧实际 dtype。额外的 stride/device/配置等字段可保存，但脚本不比较这些字段。
- NaN/Inf 用字符串 `"NaN"`、`"Inf"`、`"+Inf"`、`"-Inf"`；拒绝非标准 JSON 裸 NaN/Infinity。
  所有 NaN 都算差异；默认任何 Inf 都算差异；`--allow-matching-inf` 允许同位同号 Inf，
  仅在确认是预期 logits mask 等场景使用。没有自动把 raw state Inf 当合法值的逻辑。

## 采集方式

由已有复现 harness 在指定边界对小片段做独立 snapshot；在设备端先 clone，
再在受控、正确同步的边界转 CPU 并 row-major 展平。避免存引用后统一 dump：
conv/SSM cache 会在下一次 forward 原地更新。不要把新增 D2H/打印引起的时序变化误当 bug 修复。
不在 ACL Graph 被捕获区域执行 `.cpu()`、逐元素 `.item()` 或文件 IO。

同时保存复现 manifest：两个仓库 SHA/修改、模型/tokenizer revision、设备/runtime、运行配置、
prompt IDs、请求 ID 与 batch/state slot 映射、接受/拒绝结果。脚本只比较数值记录，不核验 manifest。
更大张量应先按请求/层/位置缩小；该脚本会把文件读入内存，不适合全模型海量 dump。

## 调用与解释

在 Linux 调试环境中，`$PYTHON` 指向确认可用的 Python 3 解释器；从 `vllm-ascend-mamba-skills` 技能仓根目录运行，`ref.jsonl` 和 `candidate.jsonl` 替换为实际 dump 路径：

```bash
"$PYTHON" vllm-ascend-mamba-accuracy/scripts/compare_dumps.py ref.jsonl candidate.jsonl --atol 0.001 --rtol 0.001
```

这里的 `0.001` 仅演示参数语法，不是 GDN/Mamba 的推荐容差。
Windows PowerShell 用 `& $PythonPath <script> <ref> <candidate>` 调用已确认的解释器。
报告为 JSON，退出码 0=满足本次比较条件，1=差异，2=输入格式/参数/读文件失败。
有限输入的差值或容差计算若溢出，也会以可比较数值范围超限退出 2；不会将 `Inf <= Inf` 判为匹配。
这只影响浮点容差比较；`kind=exact` 的大整数仍保持整数精确比较。
`--max-records 20` 只限制展示条数，`mismatched_records` 仍是所有记录的差异数。
形状/dtype/kind 先比较；不同则报告该结构问题，不继续把不同布局的数据逐元素相减。
数值报告包含首差坐标、异常元素数、非有限值对数和有限值对的最大绝对差。
`--allow-matching-inf` 只改变预期 Inf 的判定，NaN 永不通过。
脚本 match 仅表示所提供切片通过指定条件，不代表未采样位置或模型端到端正确。
