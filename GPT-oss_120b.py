#!/usr/bin/env python3
# AIME 2024 evaluator CoT + Code with N-paths and Top-5 selection
# - Uses gpt-oss-120b via vLLM
# - Loads problems from aime_problems.py (AIME_2024_PROBLEMS)
# - For each problem, runs:
#     1) Chain-of-Thought (text reasoning) with N-paths → majority voting selection
#     2) Code-only with N-paths → majority voting selection  
# - Prints per-problem correctness for both methods and final overall accuracies
#
# Usage:
#   python GPT-oss_120b.py          # Use N-paths with Top-5 selection (default)
#   python GPT-oss_120b.py --single # Use single-path evaluation (original behavior)

import re
import io
import math
import signal
import traceback
import sys
from typing import Dict, Optional
from contextlib import redirect_stdout

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

    # Remove signal handling as it can cause issues in containerized environments
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
                "__import__": __import__,  # Add __import__ to allow import statements
            },
            "math": math,
            "__name__": "__main__",  # Add __name__ to allow if __name__ == "__main__" checks
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

def select_best_answer_from_paths(predictions: list[Optional[int]]) -> Optional[int]:
    """Select the best answer from multiple paths using majority voting"""
    # Filter out None values
    valid_predictions = [p for p in predictions if p is not None]
    
    if not valid_predictions:
        return None
    
    # Use majority voting - count frequency of each answer
    from collections import Counter
    counter = Counter(valid_predictions)
    
    print(f"    📊 Vote counts: {dict(counter)}")
    
    # Return the most common answer (in case of tie, returns first one)
    most_common = counter.most_common(1)
    if most_common:
        winner = most_common[0]
        print(f"    Winner: {winner[0]} (appeared {winner[1]} times)")
        return winner[0]
    else:
        return valid_predictions[0]

def evaluate_cot_paths(runner, question: str) -> Optional[int]:
    """Evaluate CoT using N paths and select best answer"""
    print(f"  Generating {10} CoT responses...")
    responses = runner.generate_n_paths(cot_prompt(question), n=10)  # Generate more, select top 5
    
    print(f"  Generated {len(responses)} responses, extracting predictions...")
    predictions = []
    for i, resp in enumerate(responses):
        pred = extract_final_answer(resp)
        predictions.append(pred)
        print(f"    Path {i+1}: {pred} (response length: {len(resp)} chars)")
    
    final_answer = select_best_answer_from_paths(predictions)
    print(f"  CoT Final Answer via majority vote: {final_answer}")
    print(f"    All predictions: {predictions}")
    
    return final_answer

def evaluate_code_paths(runner, question: str) -> Optional[int]:
    """Evaluate Code using N paths and select best answer"""
    print(f"  Generating {10} Code responses...")
    responses = runner.generate_n_paths(code_only_prompt(question), n=10)  # Generate more, select top 5
    
    print(f"  Generated {len(responses)} responses, executing code...")
    predictions = []
    for i, resp in enumerate(responses):
        pred = execute_python_code_return_int(resp)
        predictions.append(pred)
        # Show if code block was found
        has_code = "```python" in resp
        print(f"    Path {i+1}: {pred} (code block: {'✓' if has_code else '✗'}, length: {len(resp)} chars)")
    
    # Filter out 0 values which are often fallback values from failed execution
    valid_predictions = [p for p in predictions if p != 0]
    if not valid_predictions:
        valid_predictions = predictions  # Fall back to including zeros if that's all we have
        print(f"  No non-zero predictions found, using all predictions including zeros")
    else:
        print(f"  Filtered out {len(predictions) - len(valid_predictions)} zero predictions")
    
    final_answer = select_best_answer_from_paths(valid_predictions)
    print(f"  Code Final Answer via majority vote: {final_answer}")
    print(f"  All predictions: {predictions}")
    print(f"  Valid predictions: {valid_predictions}")
    
    return final_answer

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

    def generate_once(self, prompt: str) -> str:
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            n=1,
            max_tokens=4000,
        )
        out = self.llm.generate([prompt], params)[0]
        return (out.outputs[0].text or "").strip() if out.outputs else ""
    
    def generate_n_paths(self, prompt: str, n: int = 5) -> list[str]:
        """Generate N diverse paths using higher temperature and return top 5 by length/quality"""
        params = SamplingParams(
            temperature=0.8,  # Higher temperature for diversity
            top_p=0.9,
            repetition_penalty=1.1,
            n=n,
            max_tokens=4000,
        )
        out = self.llm.generate([prompt], params)[0]
        responses = [(output.text or "").strip() for output in out.outputs]
        
        # Filter out empty responses
        responses = [r for r in responses if r]
        print(f"    Generated {len(responses)} non-empty responses from {n} requested")
        
        # Sort by quality metrics (length as a proxy for completeness)
        # You could add more sophisticated scoring here
        scored_responses = []
        for i, resp in enumerate(responses):
            score = len(resp)  # Basic scoring by length
            bonus_points = 0
            # Add bonus for having key indicators of good responses
            if "Final Answer:" in resp or "__ANS__" in resp or "answer" in resp.lower():
                bonus_points += 1000
            if "```python" in resp:
                bonus_points += 500
            
            total_score = score + bonus_points
            scored_responses.append((total_score, resp))
            print(f"      Response {i+1}: score={total_score} (length={score}, bonus={bonus_points})")
        
        # Sort by score (descending) and take top 5
        scored_responses.sort(key=lambda x: x[0], reverse=True)
        top_responses = [resp for _, resp in scored_responses[:5]]
        
        print(f"    ⭐ Selected top {len(top_responses)} responses for evaluation")
        return top_responses if top_responses else [""]  # Return at least one response

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
    Read the problem and then solve the problem within the code block. Output everything after the ```python tag. 
    Generate ONLY Python code. No explanations or comments.

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

# ========================= Evaluation =========================
def evaluate_problem(runner: QwenRunner, prob: Dict) -> Dict:
    q = prob["question"]
    true_ans = int(prob["answer"])

    print(f"\n{'='*80}")
    print(f"PROBLEM: {q[:100]}{'...' if len(q) > 100 else ''}")
    print(f"TRUE ANSWER: {true_ans}")
    print(f"{'='*80}")
    
    # Use N-paths approach for both CoT and Code
    print("\nCHAIN-OF-THOUGHT EVALUATION:")
    cot_pred = evaluate_cot_paths(runner, q)
    
    print("\nCODE EVALUATION:")
    code_pred = evaluate_code_paths(runner, q)

    print(f"\nRESULTS SUMMARY:")
    print(f"  CoT Prediction: {cot_pred} {'✅' if cot_pred == true_ans else '❌'}")
    print(f"  Code Prediction: {code_pred} {'✅' if code_pred == true_ans else '❌'}")
    print(f"  True Answer: {true_ans}")

    return {
        "true": true_ans,
        "cot_pred": cot_pred,
        "code_pred": code_pred,
        "cot_ok": (cot_pred == true_ans),
        "code_ok": (code_pred == true_ans),
    }

def evaluate_problem_single_path(runner: QwenRunner, prob: Dict) -> Dict:
    """Keep the original single-path evaluation for comparison"""
    q = prob["question"]
    true_ans = int(prob["answer"])

    cot_text = runner.generate_once(cot_prompt(q))
    cot_pred = extract_final_answer(cot_text)

    code_text = runner.generate_once(code_only_prompt(q))
    print("code_text,", code_text)
    
    # Debug: Show what the model actually generated
    print(f"\n=== DEBUG: Model response for CODE prompt ===")
    print("q,", q)
    print("code text,", code_text)
    print("=== END DEBUG ===\n")
    
    # Quick diagnostic: check if we're getting valid code blocks
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
    # Check for command line argument to determine mode
    use_single_path = "--single" in sys.argv
    
    mode_str = "Single-path" if use_single_path else "N-paths with Top-5 Selection"
    print(f"🔥 AIME 2024 — CoT + Code ({mode_str})")
    print("=" * 70)

    runner = QwenRunner("openai/gpt-oss-120b")
    problems = AIME_2024_PROBLEMS

    cot_correct = 0
    code_correct = 0
    total = len(problems)

    evaluate_func = evaluate_problem_single_path if use_single_path else evaluate_problem

    for i, prob in enumerate(problems, 1):
        res = evaluate_func(runner, prob)
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
