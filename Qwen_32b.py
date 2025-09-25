#!/usr/bin/env python3
# AIME 2024 evaluator — Boosted CoT accuracy (code-only path unchanged)
# - Keeps your Code-only logic and prompts exactly the same.
# - Improves CoT by: (1) stronger guardrails in the CoT prompt,
#   (2) self-consistency (multi-sample majority vote),
#   (3) an answer-only fallback if the vote is weak or missing.
#
# Usage: run as-is. Tune COT_N / COT_TEMP / COT_TOKENS below if desired.

import re
import io
import math
import traceback
from typing import Dict, Optional, List, Tuple
from contextlib import redirect_stdout
from collections import Counter

from vllm import LLM, SamplingParams

# -------- Load dataset (AIME 2024 only) --------
try:
    from aime_problems import AIME_2024_PROBLEMS  # list of {"question": str, "answer": str}
    print(f"✓ Loaded {len(AIME_2024_PROBLEMS)} real AIME 2024 problems")
except Exception as e:
    print(f"⚠️  WARNING: Could not load real AIME problems ({e}), using fallback dummy problems")
    AIME_2024_PROBLEMS = [
        {"question": "Compute 1+2+...+10.", "answer": "55"},
        {"question": "If 2x+3=17, find x.", "answer": "7"},
        {"question": "Find the remainder when 7^5 is divided by 100.", "answer": "7"},
    ]

# ========================= Utilities =========================

def _coerce_int(x) -> Optional[int]:
    try:
        if isinstance(x, bool): return None
        if isinstance(x, int): return x
        if isinstance(x, float):
            r = round(x);  return r if abs(x - r) < 1e-9 else None
        return int(str(x).strip())
    except Exception:
        return None

def execute_python_code_return_int(code_text: str, timeout: int = 12) -> int:
    """
    Execute a Python code block (from model) in a restricted environment and recover an integer.
    NOTE: This is the same logic you had; unchanged to honor your request.
    """
    m = re.search(r"```python\s*\n(.*?)```", code_text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        code = m.group(1).strip()
    else:
        lines = code_text.splitlines()
        code_lines, in_code = [], False
        for line in lines:
            if line.strip().startswith("#") or any(k in line for k in ("import", "def ", "for ", "while ", "if ", "=", "print")):
                in_code = True
            if in_code: code_lines.append(line)
        code = "\n".join(code_lines).strip()
    if not code or len(code) < 4: return 0

    out_buf = io.StringIO()
    try:
        safe_globals = {
            "__builtins__": {
                "range": range, "len": len, "int": int, "float": float,
                "str": str, "list": list, "dict": dict, "set": set,
                "tuple": tuple, "sum": sum, "max": max, "min": min,
                "abs": abs, "round": round, "pow": pow, "print": print,
                "enumerate": enumerate, "zip": zip, "sorted": sorted,
                "reversed": reversed, "all": all, "any": any,
                "__import__": __import__,
            },
            "math": math,
            "__name__": "__main__",
        }
        safe_locals = {}
        with redirect_stdout(out_buf):
            exec(code, safe_globals, safe_locals)

        def _extract_from_printed(printed: str) -> Optional[int]:
            if not printed: return None
            m2 = re.findall(r"__ANS__\s*=\s*(-?\d+)", printed)
            if m2: return int(m2[-1])
            m3 = re.findall(r"-?\d+", printed)
            return int(m3[-1]) if m3 else None

        printed = out_buf.getvalue().strip()
        v = _extract_from_printed(printed)
        if v is not None: return v

        for key in ("answer", "result", "final_answer", "ans", "solution", "output"):
            if key in safe_locals:
                v = _coerce_int(safe_locals[key])
                if v is not None: return v

        for fn_name in ("solve", "main"):
            fn = safe_locals.get(fn_name)
            if callable(fn):
                out2 = io.StringIO()
                with redirect_stdout(out2): ret = fn()
                v = _extract_from_printed(out2.getvalue().strip())
                if v is not None: return v
                v2 = _coerce_int(ret)
                if v2 is not None: return v2
                for key in ("answer", "result", "final_answer", "ans", "solution", "output"):
                    if key in safe_locals:
                        v3 = _coerce_int(safe_locals[key])
                        if v3 is not None: return v3

        if printed:
            m4 = re.findall(r"-?\d+", printed)
            if m4: return int(m4[-1])

        m5 = re.findall(r"answer\s*=\s*(-?\d+)", code_text)
        if m5: return int(m5[-1])
        m6 = re.findall(r"-?\d+", code_text)
        if m6: return int(m6[-1])

        return 0
    except Exception:
        return 0

def extract_final_answer(text: str) -> Optional[int]:
    # Slightly stricter: only accept 0..999
    patt = [
        r"\*\*Final Answer[:\s]*([0-9]{1,3})\*\*",
        r"Final Answer[:\s]*([0-9]{1,3})\b",
        r"The final answer is[:\s]*([0-9]{1,3})\b",
        r"Answer[:\s]*([0-9]{1,3})\b",
        r"\b=\s*([0-9]{1,3})\s*$",
    ]
    for p in patt:
        m = re.search(p, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            try:
                v = int(m.group(1))
                if 0 <= v <= 999: return v
            except Exception:
                pass
    tail = "\n".join((text or "").strip().splitlines()[-20:])
    nums = re.findall(r"\b([0-9]{1,3})\b", tail)
    if nums:
        v = int(nums[-1])
        return v if 0 <= v <= 999 else None
    return None

# ========================= Model Wrapper =========================
class QwenRunner:
    def __init__(self, model_path: str = "Qwen/Qwen3-32B"):
        print(f"Loading model: {model_path}")
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=16384,
            dtype="bfloat16",
            enforce_eager=False,
            disable_log_stats=True,
            enable_prefix_caching=True,
            max_num_seqs=16,
        )
        print("✓ Model loaded\n")

    def generate_once(self, prompt: str, *, temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 2048) -> str:
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.0,
            n=1,
            max_tokens=max_tokens,
        )
        out = self.llm.generate([prompt], params)[0]
        return (out.outputs[0].text or "").strip() if out.outputs else ""

    def generate_many(self, prompt: str, *, n: int, temperature: float, top_p: float, max_tokens: int) -> List[str]:
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.0,
            n=n,
            max_tokens=max_tokens,
        )
        out = self.llm.generate([prompt], params)[0]
        return [ (o.text or "").strip() for o in out.outputs ]

# ========================= Prompts =========================
def cot_prompt(problem: str) -> str:
    # Strong guardrails, built-in sanity/verification, strict final line.
    return f"""You are solving an AIME problem. Work carefully and check your result.

Problem:
{problem}

Checklist before answering:
- Keep reasoning concise and logically valid.
- Use exact arithmetic when possible; verify key steps (parity, mod arithmetic, bounds, plugging into conditions).
- The AIME answer is an integer in [0, 999].

Output format:
- End with **exactly one final line**: **Final Answer: [number]**
- Do not add text after that final line.
"""

def cot_answer_only_prompt(problem: str) -> str:
    return f"""Solve the AIME problem below **privately** and output only the final line.

Problem:
{problem}

Output exactly one line:
**Final Answer: [number]**
"""

def code_only_prompt(problem: str) -> str:
    # UNCHANGED (you asked not to change code-only logic)
    return f"""You are a Python code generator. Generate ONLY Python code. No explanations.

```python
import math

def solve():
    # Problem: {problem}
    
    # TODO: Solve the problem above and set answer to the integer result
    answer = 1  # Replace this with your solution
    
    print("__ANS__=" + str(answer))

if __name__ == "__main__":
    solve()
```"""

# ---------- CoT booster: self-consistency + fallback ----------
# Tune these to trade speed for accuracy:
COT_N = 5          # how many CoT samples to draw
COT_TEMP = 0.25    # small temperature for diversity
COT_TOP_P = 0.95
COT_TOKENS = 2048

def cot_ensemble_predict(runner: QwenRunner, q: str) -> Tuple[Optional[int], List[str]]:
    texts = runner.generate_many(
        cot_prompt(q),
        n=COT_N,
        temperature=COT_TEMP,
        top_p=COT_TOP_P,
        max_tokens=COT_TOKENS,
    )
    preds = [extract_final_answer(t) for t in texts if t]
    preds = [p for p in preds if p is not None and 0 <= p <= 999]

    if preds:
        counts = Counter(preds)
        best, freq = counts.most_common(1)[0]
        # Weak vote (all different)? add an answer-only fallback burst
        if freq == 1:
            ao_texts = runner.generate_many(
                cot_answer_only_prompt(q),
                n=3,
                temperature=0.15,
                top_p=0.95,
                max_tokens=160,
            )
            ao_preds = [extract_final_answer(t) for t in ao_texts if t]
            all_preds = [p for p in preds + ao_preds if p is not None]
            if all_preds:
                best = Counter(all_preds).most_common(1)[0][0]
        return best, texts

    # No usable CoT prediction → try a compact answer-only fallback
    t = runner.generate_once(cot_answer_only_prompt(q), temperature=0.1, top_p=0.95, max_tokens=160)
    return extract_final_answer(t), texts + ([t] if t else [])

# ========================= Evaluation =========================
def evaluate_problem(runner: QwenRunner, prob: Dict) -> Dict:
    q = prob["question"]
    true_ans = int(prob["answer"])

    # ---- Improved CoT (self-consistency + fallback) ----
    cot_pred, cot_texts = cot_ensemble_predict(runner, q)

    # ---- Code-only (unchanged) ----
    code_text = runner.generate_once(code_only_prompt(q))
    print(f"\n=== DEBUG: Model response for CODE prompt ===")
    print(code_text)
    print("=== END DEBUG ===\n")

    if "```python" not in code_text:
        print(f"  WARNING: No python code block found in response for problem")
    elif "__ANS__" not in code_text and "answer" not in code_text.lower():
        print(f"  WARNING: No answer extraction mechanism found in code")

    code_pred = execute_python_code_return_int(code_text)

    return {
        "true": true_ans,
        "cot_pred": cot_pred,
        "code_pred": code_pred,
        "cot_ok": (cot_pred == true_ans),
        "code_ok": (code_pred == true_ans),
    }

def main():
    print("🔥 AIME 2024 — CoT (ensemble) + Code (original logic)")
    print("=" * 64)
    print(f"CoT: n={COT_N}, temp={COT_TEMP}, top_p={COT_TOP_P}, max_tokens={COT_TOKENS}")

    runner = QwenRunner("Qwen/Qwen3-32B")
    problems = AIME_2024_PROBLEMS

    cot_correct = 0
    code_correct = 0
    total = len(problems)

    for i, prob in enumerate(problems, 1):
        res = evaluate_problem(runner, prob)
        cot_correct += int(res["cot_ok"])
        code_correct += int(res["code_ok"])

        print(f"Problem {i:02d}: "
              f"COT={'✓' if res['cot_ok'] else '✗'}(pred={res['cot_pred']}, true={res['true']})  |  "
              f"CODE={'✓' if res['code_ok'] else '✗'}(pred={res['code_pred']}, true={res['true']})")

    cot_acc = 100.0 * cot_correct / total if total else 0.0
    code_acc = 100.0 * code_correct / total if total else 0.0

    print("\n=== FINAL ACCURACY ===")
    print(f"COT : {cot_correct}/{total}  = {cot_acc:.2f}%")
    print(f"CODE: {code_correct}/{total}  = {code_acc:.2f}%")
    print("======================")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
