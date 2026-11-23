# 题面格式规范

一题一个目录：`tasks/<family>-<op>-<tier>/`

## 一、三件套（kernel 类）

### `definition.json` —— 不变的部分

```json
{
  "id": "gemm-bf16-l1",
  "kind": "kernel",
  "family": "gemm",
  "tier": "L1",
  "precision": ["bfloat16"],
  "axes": {"M": {"type": "var"}, "N": {"type": "const", "value": 4096}},
  "inputs": {"A": {"shape": ["M","N"], "dtype": "bfloat16"}},
  "outputs": {"C": {"shape": ["M","N"], "dtype": "bfloat16"}},
  "tolerance": {"rtol": 0.02, "atol": 0.01},
  "reference": "reference.py",
  "dcu": {"peak_constant": "fp8_peak_nominal", "min_l2_flush_mb": 16}
}
```

规则：
- `axes` 只能是 `const`（给 value）或 `var`（取值域由 workload 给）
- `tolerance` 必填，**评测时不许临时放宽**
- `dcu.peak_constant` 只写 constants 里的 id，**不写字面量**
- `reference` 指向同目录的参考实现，必须暴露 `arun(...)`

### `workload.jsonl` —— 具体形状

```json
{"axes": {"M": 4096}, "inputs": {"A": {"type": "random"}}, "uuid": "pub-1", "split": "public"}
{"axes": {"M": 13312}, "inputs": {"A": {"type": "random"}}, "uuid": "hold-1", "split": "holdout"}
```

规则：
- `split: holdout` 的实例**不出现在题面里**，只在评分时用
- public / holdout 至少各一条，否则 `workload.split()` 抛错
- 输入生成必须可复现：seed 由 `uuid` 派生
- **形状要有不整齐的**（如 13312 而非 16384），破对齐假设——整齐形状容易被「刚好对齐」的写法骗到

### `solution.json` —— 提交

```json
{
  "task_id": "gemm-bf16-l1",
  "kind": "kernel",
  "entrypoint": "solution.py:run",
  "claims": {"impl": "aiter.gemm_a16w16", "ratio": 0.84}
}
```

`claims` 是幻觉检测的输入：它声称的每个函数、库名、数字都会被逐条验。

## 二、judgment 类（L3 / L5）

没有 `reference.py`，没有 `tolerance`，`kind: judgment`：

```json
{
  "id": "moe-asm-fmoe-l5",
  "kind": "judgment",
  "family": "moe",
  "tier": "L5",
  "question": "aiter.asm_fmoe_a8 能否用于 w8a8 MoE 融合",
  "expected_verdict": "unavailable",
  "verdict_keywords": ["unsupport", "dtype", "mode", "namespace"],
  "forbidden_claims": ["可以融合", "预期提升", "建议上线"],
  "evidence_required": ["error_output", "command"]
}
```

判分不看数值，看「结论对不对 + 有没有证据 + 有没有编」。

## 三、六件套目录

```
tasks/<id>/
├── task.md          人读的题面（三臂逐字相同的 prompt）
├── definition.json  机器读
├── workload.jsonl   形状实例（kernel 类）
├── reference.py     参考实现（kernel 类）
├── expect.md        参考答案（不给 agent）
└── check.py         题专属机检（可选）
```

## 四、`task.md` 的写法

必须包含：背景（真实场景）、要回答的（一句话 + 交付项）、约束（预算 / 没验证的标未验证）、评判标准。

**不能包含**：答案、我们已有的结论、任何 `expect.md` 的内容。
