# Linux NPU 复现与定点观测

以下命令从同时含 `vllm/`、`vllm-ascend/` 的项目根目录执行，使用已准备好的 NPU 环境。
这里不提供通用的 CUDA 安装步骤。先确认 torch、torch_npu、CANN 和两仓代码构成用户实际失败环境。
`TASK_PY` 指向 vLLM 的现有虚拟环境；遵循本仓 `AGENTS.md`，需要管理依赖时用 `uv`。

## 1. 来源与源码路由

```bash
git -C vllm rev-parse HEAD
git -C vllm status --short
git -C vllm-ascend rev-parse HEAD
git -C vllm-ascend status --short
export TASK_PY="$PWD/vllm/.venv/bin/python"
test -x "$TASK_PY"
"$TASK_PY" - <<'PY'
import os
import importlib.metadata as m
import vllm, vllm_ascend, torch, torch_npu
for name in ("vllm", "vllm-ascend", "torch", "torch-npu", "triton-ascend"):
    try:
        print(name, m.version(name))
    except m.PackageNotFoundError:
        print(name, "distribution metadata unavailable")
print("vllm", vllm.__file__)
print("vllm_ascend", vllm_ascend.__file__)
for name in ("VLLM_USE_V2_MODEL_RUNNER", "VLLM_LOGGING_LEVEL", "ASCEND_RT_VISIBLE_DEVICES"):
    print(name, os.getenv(name))
print("npu available", torch.npu.is_available())
PY
npu-smi info
rg -n 'def schedule|def update_from_output|scheduled_spec_decode_tokens' vllm/vllm/v1/core/sched/scheduler.py
rg -n 'init_speculator|def sample|def postprocess_sampled|def prepare_inputs' vllm-ascend/vllm_ascend/worker/v2
rg -n 'rejection_sample =|precopy_mamba_align_fused_kernel =|init_model_state =' vllm-ascend/vllm_ascend/patch
```

版本输出不包含 CANN 的可靠版本信息；另从当前容器镜像标签/安装 manifest 收集 CANN，并说明来源。
导入失败时优先解决导入路径和依赖冲突，不为了运行诊断就批量升级环境。

## 2. 短输出 eager 对照

先设置 `TARGET_MODEL` 为现有模型路径，`SPEC_CONFIG` 为原任务已支持的 JSON 配置。
例如**模型已包含并支持 MTP 权重时**才可用 `{"method":"mtp","num_speculative_tokens":1,"enforce_eager":true}`。
EAGLE 或独立 draft model 应保留真实 draft 模型与 revision，不能替换成这个 MTP 示例。
`TP_SIZE` 可保留使模型装得下的并行度；两个运行会顺序重复加载模型。

```bash
: "${TARGET_MODEL:?set TARGET_MODEL to the existing model path}"
: "${SPEC_CONFIG:?set SPEC_CONFIG to the verified speculative JSON config}"
: "${TP_SIZE:=1}"
: "${MAX_MODEL_LEN:=4096}"
export TARGET_MODEL SPEC_CONFIG TP_SIZE MAX_MODEL_LEN
mkdir -p debug-artifacts
for mode in baseline speculative; do
  RUN_MODE="$mode" VLLM_LOGGING_LEVEL=DEBUG "$TASK_PY" - <<'PY' >"debug-artifacts/${mode}.log" 2>&1
import json
import os
from vllm import LLM, SamplingParams

spec = None
if os.environ["RUN_MODE"] == "speculative":
    spec = json.loads(os.environ["SPEC_CONFIG"])
llm = LLM(
    model=os.environ["TARGET_MODEL"],
    tensor_parallel_size=int(os.environ.get("TP_SIZE", "1")),
    max_model_len=int(os.environ.get("MAX_MODEL_LEN", "4096")),
    max_num_seqs=1,
    enable_prefix_caching=False,
    enforce_eager=True,
    seed=0,
    speculative_config=spec,
)
outputs = llm.generate(
    ["Explain why the sky appears blue in three sentences."],
    SamplingParams(temperature=0, max_tokens=64, seed=0),
)
for output in outputs:
    print(json.dumps({
        "mode": os.environ["RUN_MODE"],
        "prompt_token_ids": output.prompt_token_ids,
        "output_token_ids": output.outputs[0].token_ids,
        "text": output.outputs[0].text,
    }, ensure_ascii=False))
PY
done
rg -n '\[spec/dfx\]|Traceback|ERROR|output_token_ids' debug-artifacts
```

这是起点，不会保留所有故障条件。若原配置需要量化、远端模型代码、async、chunked prefill 或 cache mode，应从已验证的原命令补入；每个试验记录差异。
不要比较两份整个日志是否相同；比较末尾 JSON 的 prompt IDs、输出 IDs 以及第一处分叉位置。
同样的 seed 不保证跨 kernel 或投机开关的随机采样路径逐 token 相同，所以第一轮使用 greedy。

## 3. 在故障 worker 中核实实际 callable

以下片段适合临时插入现有、已完成初始化的 runner 调试位置；`self` 指 runner。
只在一个选定 rank 记录一次，排查完成后去掉；这不是独立运行的环境探针。

```python
import inspect

def describe(label, obj):
    fn = getattr(obj, "__func__", obj)
    fn = getattr(fn, "fn", fn)  # Triton JIT wrapper may carry a Python fn.
    try:
        source = inspect.getsourcefile(fn)
    except (TypeError, OSError):
        source = None
    logger.info("DFX %s module=%s source=%s", label,
                getattr(fn, "__module__", None), source)

logger.info("DFX runner=%s MRO=%s", type(self), type(self).__mro__)
for name in ("execute_model", "sample_tokens", "sample", "_sample"):
    if hasattr(self, name):
        describe(name, getattr(self, name))
for name in ("speculator", "drafter", "model_state", "rejection_sampler"):
    obj = getattr(self, name, None)
    if obj is not None:
        logger.info("DFX %s=%s MRO=%s", name, type(obj), type(obj).__mro__)
        if hasattr(obj, "propose"):
            describe(name + ".propose", obj.propose)
```

按当前 runner 额外记录 rejection 模块的 `rejection_sample`，检查 V2 importer 和 utils 两个 alias 是否为同一 NPU callable。
记录 `model_state.prepare_attn` 和 `mamba_utils.precopy_mamba_align_fused_kernel` 时也使用同样方法。
仅有 `__module__` 可能不足以排除包装器；结合 source、MRO、当前函数体中的实际被调函数。

## 4. 测试与统计

从 Ascend 仓库执行最相关的现有单测；UT 也可能需要测试 fixture 提供 torch_npu/mock，不能假定任何 CPU 环境都能运行。

```bash
cd vllm-ascend
"$TASK_PY" -m pytest -q tests/ut/worker/v2/test_spec_decode.py
"$TASK_PY" -m pytest -q tests/ut/ops/test_gdn_attn_builder.py tests/ut/ops/test_gdn_mixed_batch.py
"$TASK_PY" -m pytest -q tests/ut/worker/test_model_runner_v2_mamba.py
```

两卡 hybrid 冒烟需兼容设备、可用内存和既有模型；先用 `--collect-only` 看参数化，再定向运行：

```bash
QWEN35_DENSE_MODEL="$TARGET_MODEL" "$TASK_PY" -m pytest --collect-only -q tests/e2e/pull_request/two_card/model_runner_v2/test_mamba_hybrid.py
QWEN35_DENSE_MODEL="$TARGET_MODEL" "$TASK_PY" -m pytest -sv tests/e2e/pull_request/two_card/model_runner_v2/test_mamba_hybrid.py::test_qwen35_27b_acl_graph
```

这里的 `TARGET_MODEL` 必须与测试假定的 Qwen3.5 架构匹配。测试用例内 `num_speculative_tokens=3`，draft eager、target 可用图；不能当作“双侧图模式”验证。
接受率可复用 `vllm-ascend/tools/spec_decode_metrics.py` 的 `capture_baseline` 和 `measure_acceptance_rate`，在同一请求窗口前后取差分。
其 per-position 比率分母是 draft 次数；这些累计接受率不是条件接受率，也不是 target 最终 token 精度。
对共享服务还应按模型/worker labels 过滤或隔离测试流量；现有 helper 会累计匹配 metric family 的 samples。
