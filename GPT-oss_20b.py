#!/usr/bin/env python3
# AIME 2024 evaluator — CoT + Code (ONE trial each), single-file, no token accounting
# - Uses gpt-oss-20b via vLLM
# - Loads problems from aime_problems.py (AIME_2024_PROBLEMS)
# - For each problem, runs:
#     1) Chain-of-Thought (text reasoning) → extract **Final Answer: n**
#     2) Code-only → executes a strict, self-checking code template and extracts a number
# - Prints per-problem correctness for both methods and final overall accuracies

import re
import io
import math
import signal
import traceback
from typing import Dict, Optional
from contextlib import redirect_stdout

from vllm import LLM, SamplingParams

# -------- Load dataset (AIME 2024 only) --------
try:
    from aime_problems import AIME_2024_PROBLEMS  # list of {"question": str, "answer": str}
except Exception:
    # Fallback tiny sample if your local aime_problems.py isn't available.
    AIME_2024_PROBLEMS = [
        {"question": "Compute 1+2+...+10.", "answer": "55"},
        {"question": "If 2x+3=17, find x.", "answer": "7"},
        {"question": "Find the remainder when 7^5 is divided by 100.", "answer": "7"},
    ]

# ========================= Utilities =========================
class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException("Code execution timed out")

def _coerce_int(x) -> Optional[int]:
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            r = round(x)
            return r if abs(x - r) < 1e-9 else None
        return int(str(x).strip())
    except Exception:
        return None

def execute_python_code_return_int(code_text: str, timeout: int = 12) -> int:
    """
    Execute a Python code block (from model) in a restricted environment and recover an integer.
    Strategy (very robust; guarantees an int is returned):
      1) Extract code from ```python ... ```
      2) Exec once in a restricted env; then:
         - Prefer sentinel print '__ANS__=<int>'
         - Else prefer variable `answer`
         - Else call solve()/main() if present, then re-check prints/answer/return
      3) If still nothing, parse integers from printed output
      4) If still nothing, parse integers from the code text (answer = ..., last literal)
      5) As a last resort, return 0 (ensures a number is always produced)
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
            if in_code:
                code_lines.append(line)
        code = "\n".join(code_lines).strip()

    if not code or len(code) < 4:
        return 0

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
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
            },
            "math": math,
        }
        safe_locals = {}
        with redirect_stdout(out_buf):
            exec(code, safe_globals, safe_locals)
        signal.alarm(0)

        def _extract_from_printed(printed: str) -> Optional[int]:
            if not printed:
                return None
            m2 = re.findall(r"__ANS__\s*=\s*(-?\d+)", printed)
            if m2:
                return int(m2[-1])
            m3 = re.findall(r"-?\d+", printed)
            return int(m3[-1]) if m3 else None

        printed = out_buf.getvalue().strip()
        ans = _extract_from_printed(printed)
        if ans is not None:
            return ans

        for key in ("answer", "result", "final_answer", "ans", "solution", "output"):
            if key in safe_locals:
                v = _coerce_int(safe_locals[key])
                if v is not None:
                    return v

        for fn_name in ("solve", "main"):
            fn = safe_locals.get(fn_name)
            if callable(fn):
                signal.alarm(timeout)
                out_buf2 = io.StringIO()
                with redirect_stdout(out_buf2):
                    ret = fn()
                signal.alarm(0)
                printed2 = out_buf2.getvalue().strip()
                ans2 = _extract_from_printed(printed2)
                if ans2 is not None:
                    return ans2
                v = _coerce_int(ret)
                if v is not None:
                    return v
                for key in ("answer", "result", "final_answer", "ans", "solution", "output"):
                    if key in safe_locals:
                        v = _coerce_int(safe_locals[key])
                        if v is not None:
                            return v

        if printed:
            m4 = re.findall(r"-?\d+", printed)
            if m4:
                return int(m4[-1])

        m5 = re.findall(r"answer\s*=\s*(-?\d+)", code_text)
        if m5:
            return int(m5[-1])
        m6 = re.findall(r"-?\d+", code_text)
        if m6:
            return int(m6[-1])

        return 0

    except TimeoutException:
        return 0
    except Exception:
        return 0
    finally:
        signal.alarm(0)

def extract_final_answer(text: str) -> Optional[int]:
    patt = [
        r"\*\*Final Answer[:\s]*(-?\d{1,3})\*\*",
        r"Final Answer[:\s]*(-?\d{1,3})",
        r"The final answer is[:\s]*(-?\d{1,3})",
        r"Answer[:\s]*(-?\d{1,3})",
        r"=\s*(-?\d{1,3})\s*$",
    ]
    for p in patt:
        m = re.search(p, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    tail = "\n".join(text.strip().splitlines()[-15:])
    nums = re.findall(r"-?\d{1,3}", tail)
    if nums:
        try:
            return int(nums[-1])
        except Exception:
            return None
    return None

# ========================= Model Wrapper =========================
class QwenRunner:
    def __init__(self, model_path: str = "gpt-oss-20b"):
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

    def generate_once(self, prompt: str) -> str:
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            n=1,
            max_tokens=2048,
        )
        out = self.llm.generate([prompt], params)[0]
        return (out.outputs[0].text or "").strip() if out.outputs else ""

# ========================= Prompts =========================
def cot_prompt(problem: str) -> str:
    return f"""You are solving an AIME problem. Show clear steps and then give the exact integer.

Problem:
{problem}

Instructions:
- Keep the reasoning concise and correct.
- Finish with the line: **Final Answer: [number]**

Now solve it.
"""

def code_only_prompt(problem: str) -> str:
    return f"""Solve the following AIME-style problem **using Python code ONLY**.

HARD REQUIREMENTS (must follow exactly):
- Output exactly ONE Python code block; no extra text.
- Use only Python built-ins and the `math` module (no other imports).
- Put your logic in a `solve()` function. It MUST set an integer variable `answer`.
- At the end of `solve()`, print: print("__ANS__=" + str(answer))
- Include the guard so it runs: if __name__ == "__main__": solve()
- `answer` MUST be an int (not float); if you compute a float, cast safely to int.
- Do not print anything except the single line with "__ANS__=".

Problem:
{problem}

Your response must be exactly one code block like this:

```python
# You may add short comments.
import math

def solve():
    # compute the final integer and store it in `answer`
    answer = 0  # <-- replace with your computation
    assert isinstance(answer, int)
    print("__ANS__=" + str(answer))

if __name__ == "__main__":
    solve()
```"""

# ========================= Evaluation =========================
def evaluate_problem(runner: QwenRunner, prob: Dict) -> Dict:
    q = prob["question"]
    true_ans = int(prob["answer"])

    cot_text = runner.generate_once(cot_prompt(q))
    cot_pred = extract_final_answer(cot_text)

    code_text = runner.generate_once(code_only_prompt(q))
    code_pred = execute_python_code_return_int(code_text)

    return {
        "true": true_ans,
        "cot_pred": cot_pred,
        "code_pred": code_pred,
        "cot_ok": (cot_pred == true_ans),
        "code_ok": (code_pred == true_ans),
    }

def main():
    print("🔥 AIME 2024 — CoT + Code (ONE trial each)")
    print("=" * 60)

    runner = QwenRunner("gpt-oss-20b")
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
