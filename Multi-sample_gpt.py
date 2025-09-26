#!/usr/bin/env python3
# AIME 2024 evaluator — Multi-sample consensus (vote) per method
# - Uses vLLM to sample N generations for each method (CoT and Code)
# - Majority vote within each method to pick its final prediction
# - Prints per-problem correctness for both methods and final accuracies
#
# Usage:
#   python GPT-oss_120b.py --n 5         # number of samples per method (default 5)

import re
import io
import math
import traceback
import sys
import argparse
from typing import Dict, Optional, List
from contextlib import redirect_stdout
from collections import Counter

from vllm import LLM, SamplingParams

# -------- Load dataset (AIME 2024 only) --------
try:
    from aime_problems import AIME_2024_PROBLEMS  # list of {"question": str, "answer": str}
    print(f"✓ Loaded {len(AIME_2024_PROBLEMS)} real AIME 2024 problems")
except Exception as e:
    # Fallback tiny sample if your local aime_problems.py isn't available.
    print(f"WARNING: Could not load real AIME problems ({e}), using fallback dummy problems")
    AIME_2024_PROBLEMS = [
        {"question": "Compute 1+2+...+10.", "answer": "55"},
        {"question": "If 2x+3=17, find x.", "answer": "7"},
        {"question": "Find the remainder when 7^5 is divided by 100.", "answer": "7"},
    ]


# ========================= Utilities =========================

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
                "__import__": __import__,  # allow imports in code
            },
            "math": math,
            "__name__": "__main__",  # allow if __name__ == "__main__"
        }
        safe_locals = {}
        with redirect_stdout(out_buf):
            exec(code, safe_globals, safe_locals)

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
                out_buf2 = io.StringIO()
                with redirect_stdout(out_buf2):
                    ret = fn()
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

    except Exception:
        return 0


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


def select_best_answer_from_paths(predictions: List[Optional[int]]) -> Optional[int]:
    """Majority vote over predictions; tie-breaker = first seen among ties."""
    valid_predictions = [p for p in predictions if p is not None]
    if not valid_predictions:
        return None
    counter = Counter(valid_predictions)
    max_count = max(counter.values())
    # deterministic tie-breaker: first value in generation order that hits max_count
    seen = set()
    for p in valid_predictions:
        if p in seen:
            continue
        seen.add(p)
        if counter[p] == max_count:
            print(f"    📊 Vote counts: {dict(counter)}")
            print(f"    Winner: {p} (appeared {counter[p]} times; tie-broken by first occurrence)")
            return p
    return valid_predictions[0]


# ========================= Model Wrapper =========================
class QwenRunner:
    def __init__(self, model_path: str = "openai/gpt-oss-120b"):
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

    def generate_n_paths(self, prompt: str, n: int) -> List[str]:
        """Generate exactly n diverse samples (no heuristic ranking)."""
        params = SamplingParams(
            temperature=0.8,      # diversity for voting
            top_p=0.9,
            repetition_penalty=1.1,
            n=n,
            max_tokens=4000,
        )
        out = self.llm.generate([prompt], params)[0]
        responses = [(o.text or "").strip() for o in out.outputs]
        # Keep exactly n non-empty if possible; if some are empty, return what we have.
        responses = [r for r in responses if r]
        print(f"    Generated {len(responses)} non-empty responses (requested {n})")
        return responses


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
    return f"""Do not use chain of thought. Use python code to solve the problem.
Generate ONLY Python code. No explanations or comments.

```python
import math

def solve():
    # Problem: {problem}
    # TODO: Solve the problem above and set 'answer' to the integer result
    answer = 1  # Replace this with your solution
    print("__ANS__=" + str(answer))

if __name__ == "__main__":
    solve()
```"""


# ========================= Evaluation (consensus per method) =========================
def evaluate_cot_paths(runner: QwenRunner, question: str, n_paths: int) -> Optional[int]:
    print(f"  Generating {n_paths} CoT samples...")
    responses = runner.generate_n_paths(cot_prompt(question), n=n_paths)

    print(f"  Extracting predictions from CoT samples...")
    predictions: List[Optional[int]] = []
    for i, resp in enumerate(responses):
        pred = extract_final_answer(resp)
        predictions.append(pred)
        print(f"    CoT Path {i+1}: {pred} (len={len(resp)} chars)")
    final_answer = select_best_answer_from_paths(predictions)
    print(f"  CoT Final Answer (vote): {final_answer}")
    print(f"    All CoT predictions: {predictions}")
    return final_answer


def evaluate_code_paths(runner: QwenRunner, question: str, n_paths: int) -> Optional[int]:
    print(f"  Generating {n_paths} Code samples...")
    responses = runner.generate_n_paths(code_only_prompt(question), n=n_paths)

    print(f"  Executing code samples...")
    predictions: List[int] = []
    for i, resp in enumerate(responses):
        pred = execute_python_code_return_int(resp)
        predictions.append(pred)
        has_code = "```python" in resp
        print(f"    Code Path {i+1}: {pred} (code block: {'✓' if has_code else '✗'}, len={len(resp)} chars)")

    # Optional: filter out zeros (often failure fallback)
    valid_predictions = [p for p in predictions if p != 0] or predictions
    if valid_predictions is predictions:
        print("  Note: No non-zero predictions; including zeros in the vote.")
    else:
        print(f"  Filtered out {len(predictions) - len(valid_predictions)} zero predictions")

    final_answer = select_best_answer_from_paths(valid_predictions)  # type: ignore[arg-type]
    print(f"  Code Final Answer (vote): {final_answer}")
    print(f"    All Code predictions: {predictions}")
    print(f"    Valid Code predictions: {valid_predictions}")
    return final_answer


def evaluate_problem(runner: QwenRunner, prob: Dict, n_paths: int) -> Dict:
    q = prob["question"]
    true_ans = int(prob["answer"])

    print(f"\n{'='*80}")
    print(f"PROBLEM: {q[:100]}{'...' if len(q) > 100 else ''}")
    print(f"TRUE ANSWER: {true_ans}")
    print(f"{'='*80}")

    # Multi-sample consensus (vote) for both methods
    print("\nCHAIN-OF-THOUGHT (consensus):")
    cot_pred = evaluate_cot_paths(runner, q, n_paths)

    print("\nCODE (consensus):")
    code_pred = evaluate_code_paths(runner, q, n_paths)

    print(f"\nRESULTS SUMMARY:")
    print(f"  CoT Prediction:  {cot_pred} {'✅' if cot_pred == true_ans else '❌'}")
    print(f"  Code Prediction: {code_pred} {'✅' if code_pred == true_ans else '❌'}")
    print(f"  True Answer:     {true_ans}")

    return {
        "true": true_ans,
        "cot_pred": cot_pred,
        "code_pred": code_pred,
        "cot_ok": (cot_pred == true_ans),
        "code_ok": (code_pred == true_ans),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="number of samples per method for voting")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b", help="vLLM model id/path")
    args = parser.parse_args()

    print(f"🔥 AIME 2024 — Multi-sample consensus voting per method (n={args.n})")
    print("=" * 70)

    runner = QwenRunner(args.model)
    problems = AIME_2024_PROBLEMS

    cot_correct = 0
    code_correct = 0
    total = len(problems)

    for i, prob in enumerate(problems, 1):
        res = evaluate_problem(runner, prob, args.n)
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
