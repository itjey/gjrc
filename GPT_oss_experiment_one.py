#!/usr/bin/env python3
# AIME 2024 evaluator — CoT vs Code with Test-Time Compute (N = 1..10) and parallel voting
# - Uses vLLM (openai/gpt-oss-120b)
# - CoT: sample once per problem (superset N=10), then compute TTC curves by taking top-k prefixes (k=1..10)
# - Code: sample once per problem (superset N=10), keep only fenced ```python blocks, parse # EXPECTED_ANSWER: <int>
# - No code execution. Majority vote over first-k predictions gives the TTC result for that k.
# - Parallelization: per-problem per-k voting computed via ThreadPoolExecutor (GPU calls remain sequential)

import re
import sys
import math
import traceback
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from vllm import LLM, SamplingParams

# -------- Load dataset (AIME 2024 only) --------
try:
    from aime_problems import AIME_2024_PROBLEMS  # list of {"question": str, "answer": str}
    print(f"✓ Loaded {len(AIME_2024_PROBLEMS)} real AIME 2024 problems")
except Exception as e:
    print(f"WARNING: Could not load real AIME problems ({e}), using fallback dummy problems")
    AIME_2024_PROBLEMS = [
        {"question": "Compute 1+2+...+10.", "answer": "55"},
        {"question": "If 2x+3=17, find x.", "answer": "7"},
        {"question": "Find the remainder when 7^5 is divided by 100.", "answer": "7"},
    ]

K_MAX = 10           # TTC budgets: 1..10
MAX_WORKERS = 16     # For CPU-side parallel voting; tweak as needed

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

def select_best_answer_from_paths(preds: List[Optional[int]]) -> Optional[int]:
    vals = [p for p in preds if p is not None]
    if not vals:
        return None
    from collections import Counter
    counter = Counter(vals)
    # tie-break: first most_common
    return counter.most_common(1)[0][0]

def _extract_python_block(text: str) -> Optional[str]:
    m = re.search(r"```python\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    code = m.group(1).strip()
    return f"```python\n{code}\n```"

def extract_expected_answer_from_code(code_text: str) -> Optional[int]:
    block = code_text
    mblk = re.search(r"```python\s*\n(.*?)```", code_text, flags=re.DOTALL | re.IGNORECASE)
    if mblk:
        block = mblk.group(1)

    patterns = [
        r"#\s*EXPECTED_ANSWER\s*:\s*(-?\d{1,3})",
        r"EXPECTED_ANSWER\s*=\s*(-?\d{1,3})",
        r"__ANS__\s*=\s*(-?\d{1,3})",
        r"answer\s*=\s*(-?\d{1,3})",
    ]
    for pat in patterns:
        m = re.search(pat, block)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass

    header = "\n".join(block.splitlines()[:30])
    m2 = re.findall(r"-?\d{1,3}", header)
    if m2:
        try:
            return int(m2[-1])
        except Exception:
            return None
    return None

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
            temperature=0.0, top_p=1.0, repetition_penalty=1.0,
            n=1, max_tokens=4000
        )
        out = self.llm.generate([prompt], params)[0]
        return (out.outputs[0].text or "").strip() if out.outputs else ""

    def generate_n_paths(self, prompt: str, n: int = 10, extract_python_only: bool = False, keep: Optional[int] = None) -> List[str]:
        """
        Generate 'n' diverse outputs once; optionally keep only fenced python blocks;
        score and return the top 'keep' (or all if keep is None). This is the superset
        used for test-time compute prefixes k=1..K_MAX (no re-sampling per k).
        """
        params = SamplingParams(
            temperature=0.8, top_p=0.9, repetition_penalty=1.1,
            n=n, max_tokens=4000
        )
        out = self.llm.generate([prompt], params)[0]
        raw_responses = [(o.text or "").strip() for o in out.outputs]
        raw_responses = [r for r in raw_responses if r]
        print(f"    Generated {len(raw_responses)} non-empty responses (requested n={n})")

        # Filter to python code blocks when requested
        if extract_python_only:
            candidates, dropped = [], 0
            for r in raw_responses:
                blk = _extract_python_block(r)
                if blk is not None:
                    candidates.append(blk)
                else:
                    dropped += 1
            if dropped:
                print(f"    Dropped {dropped} non-code responses (no ```python block found)")
            if not candidates:
                print("    WARNING: No python code blocks found; falling back to raw responses")
                candidates = raw_responses
        else:
            candidates = raw_responses

        # Score candidates (length + heuristic bonuses)
        scored = []
        for i, resp in enumerate(candidates):
            body = resp
            m = re.search(r"```python\s*\n(.*?)```", resp, flags=re.DOTALL | re.IGNORECASE)
            if m:
                body = m.group(1)
            score = len(body)
            bonus = 0
            if extract_python_only:
                if re.search(r"#\s*EXPECTED_ANSWER\s*:", body): bonus += 1200
                if "__ANS__" in body: bonus += 300
                if "def solve" in body: bonus += 200
                if "if __name__" in body: bonus += 100
            else:
                if ("Final Answer:" in resp) or ("__ANS__" in resp) or ("answer" in resp.lower()): bonus += 1000
                if "```python" in resp: bonus += 500
            scored.append((score + bonus, resp))

        scored.sort(key=lambda x: x[0], reverse=True)
        kept = [resp for _, resp in scored[: (keep if keep is not None else len(scored))]]
        print(f"    ⭐ Selected top {len(kept)} responses")
        return kept

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
    # No execution; we require a sentinel with the expected integer.
    return f"""Generate ONLY Python code. At the very top, include a single line EXACTLY like:
# EXPECTED_ANSWER: <integer>

Then write code that (if run) would compute that same integer. Do NOT print anything.
Output ONLY a fenced python block.

```python
# EXPECTED_ANSWER: <fill with the integer>
import math

def solve():
    # Problem: {problem}
    # Write code that could derive the expected answer, but do not print.
    pass

if __name__ == "__main__":
    pass
```"""

# ========================= TTC helpers (no re-sampling) ======================
def predictions_from_cot_responses(responses: List[str]) -> List[Optional[int]]:
    return [extract_final_answer(r) for r in responses]

def predictions_from_code_responses(responses: List[str]) -> List[Optional[int]]:
    return [extract_expected_answer_from_code(r) for r in responses]

def vote_prefix(preds: List[Optional[int]], k: int) -> Optional[int]:
    """Majority vote over the first-k predictions (ignore None)."""
    k = min(k, len(preds))
    return select_best_answer_from_paths(preds[:k])

def evaluate_problem_ttc(runner: QwenRunner, prob: Dict) -> Dict:
    """Return per-k predictions (k=1..K_MAX) for both CoT and Code using a single superset per method."""
    q = prob["question"]
    true_ans = int(prob["answer"])

    print(f"\n{'='*80}")
    print(f"PROBLEM: {q[:100]}{'...' if len(q) > 100 else ''}")
    print(f"TRUE ANSWER: {true_ans}")
    print(f"{'='*80}")

    # ---- Superset generation (GPU-bound; done once per method) ----
    print("\n[CoT] Generating superset (n=10) ...")
    cot_responses = runner.generate_n_paths(cot_prompt(q), n=K_MAX, extract_python_only=False, keep=K_MAX)
    cot_preds_all = predictions_from_cot_responses(cot_responses)

    print("[Code] Generating superset (n=10, python-only) ...")
    code_responses = runner.generate_n_paths(code_only_prompt(q), n=K_MAX, extract_python_only=True, keep=K_MAX)
    code_preds_all = predictions_from_code_responses(code_responses)

    # ---- Parallel per-k voting (CPU) ----
    cot_k_preds: List[Optional[int]] = [None] * K_MAX
    code_k_preds: List[Optional[int]] = [None] * K_MAX

    def _vote_task(method: str, k: int) -> Tuple[str, int, Optional[int]]:
        if method == "cot":
            return ("cot", k, vote_prefix(cot_preds_all, k))
        else:
            return ("code", k, vote_prefix(code_preds_all, k))

    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for k in range(1, K_MAX + 1):
            tasks.append(ex.submit(_vote_task, "cot", k))
            tasks.append(ex.submit(_vote_task, "code", k))
        for fut in as_completed(tasks):
            method, k, pred = fut.result()
            if method == "cot":
                cot_k_preds[k - 1] = pred
            else:
                code_k_preds[k - 1] = pred

    # Print quick per-problem TTC summary (last budget)
    print("\nRESULTS (k=1..10):")
    for k in range(1, K_MAX + 1):
        cp = cot_k_preds[k - 1]
        kp = code_k_preds[k - 1]
        print(f"  k={k:2d} | COT={cp} {'✓' if cp == true_ans else '✗'}   CODE={kp} {'✓' if kp == true_ans else '✗'}")

    return {
        "true": true_ans,
        "cot_k_preds": cot_k_preds,     # list length K_MAX
        "code_k_preds": code_k_preds,   # list length K_MAX
    }

# ========================= Main (TTC over dataset) ===========================
def main():
    print(f"🔥 AIME 2024 — Test-Time Compute (k = 1..{K_MAX}) with parallel per-k voting")
    print("=" * 70)

    runner = QwenRunner("openai/gpt-oss-120b")
    problems = AIME_2024_PROBLEMS

    # Accumulators for TTC curves
    cot_correct_k = [0] * K_MAX
    code_correct_k = [0] * K_MAX
    total = len(problems)

    for i, prob in enumerate(problems, 1):
        res = evaluate_problem_ttc(runner, prob)
        true_ans = res["true"]
        for k in range(1, K_MAX + 1):
            cp = res["cot_k_preds"][k - 1]
            kp = res["code_k_preds"][k - 1]
            cot_correct_k[k - 1] += int(cp == true_ans)
            code_correct_k[k - 1] += int(kp == true_ans)

        # Show last budget summary per problem
        print(f"\nProblem {i:02d} summary at k={K_MAX}: "
              f"COT={'✓' if res['cot_k_preds'][K_MAX-1] == true_ans else '✗'}(pred={res['cot_k_preds'][K_MAX-1]}, true={true_ans})  |  "
              f"CODE={'✓' if res['code_k_preds'][K_MAX-1] == true_ans else '✗'}(pred={res['code_k_preds'][K_MAX-1]}, true={true_ans})")

    # Final TTC accuracy table
    print("\n=== TEST-TIME COMPUTE ACCURACY (k = 1..10) ===")
    print("k   COT_Acc(%)   CODE_Acc(%)")
    for k in range(1, K_MAX + 1):
        cot_acc = 100.0 * cot_correct_k[k - 1] / total if total else 0.0
        code_acc = 100.0 * code_correct_k[k - 1] / total if total else 0.0
        print(f"{k:2d}  {cot_acc:9.2f}   {code_acc:10.2f}")
    print("=============================================")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
