# AIME 2024 evaluator — CoT + Code (ONE trial each), fixed token budgets, robust & stronger Code path
# - CoT path: compact reasoning; extracts **Final Answer: n**
# - Code path (more capable, never-None):
#     1) Model privately thinks (≤120 tokens), then commits: __COT_ANS__=n
#     2) Model emits *stronger* code between custom markers (PYCODE:/ENDPYCODE)
#        - exact integer math (math, fractions), safe bruteforce allowed
#        - verification/assertions to self-check correctness
#        - prints *exactly one* line "__ANS__=n"
#     3) Model states simulated run result: __PREDICTED_OUTPUT__=n
#   Extraction order (with validation to [0,999]):
#       PREDICTED → DIRECT_SENTINEL → CODE_PARSE → COT_COMMIT → EXECUTE → COT_FALLBACK → ZERO
#
# Single-chunk safe:
# - No literal triple-backtick fences inside prompt strings; we use PYCODE:/ENDPYCODE markers.
#
# Diagnostics:
# - ONE trace print per problem showing which extraction path was used.

import re
import io
import math
import fractions
import itertools
import collections
import functools
import traceback
from typing import Dict, Optional, List
from contextlib import redirect_stdout

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

def _normalize_aime_int(x: Optional[int]) -> Optional[int]:
    """Clamp to valid AIME range if it looks like an int; else None."""
    if x is None:
        return None
    try:
        xi = int(x)
        if 0 <= xi <= 999:
            return xi
        return None
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
            v = _coerce_int(m.group(1))
            v = _normalize_aime_int(v)
            if v is not None:
                return v
    tail = "\n".join(text.strip().splitlines()[-15:])
    nums = re.findall(r"-?\d{1,3}", tail)
    if nums:
        v = _coerce_int(nums[-1])
        return _normalize_aime_int(v)
    return None

def extract_code_block(text: str) -> Optional[str]:
    """
    Prefer custom markers to avoid Markdown fence collisions:
      PYCODE:
      ...python code...
      ENDPYCODE
    Fallbacks: ```python ... ``` or unlabeled ``` ... ```
    Else: collect 'codey' lines as a heuristic.
    """
    m = re.search(r"PYCODE:\s*\n(.*?)\nENDPYCODE", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return collect_codey_lines(text)

def collect_codey_lines(text: str) -> Optional[str]:
    kws = ("import", "def ", "for ", "while ", "if ", "=", "print", "return")
    lines, in_block = [], False
    for ln in text.splitlines():
        s = ln.strip()
        if not in_block and any(k in s for k in kws):
            in_block = True
        if in_block and s:
            lines.append(ln)
    code = "\n".join(lines).strip()
    return code if len(code) >= 4 else None

def extract_predicted_output(text: str) -> Optional[int]:
    m = re.search(r"^__PREDICTED_OUTPUT__\s*=\s*(-?\d+)\s*$", text.strip(), flags=re.MULTILINE)
    return _normalize_aime_int(_coerce_int(m.group(1))) if m else None

def extract_cot_commit(text: str) -> Optional[int]:
    m = re.search(r"^__COT_ANS__\s*=\s*(-?\d+)\s*$", text.strip(), flags=re.MULTILINE)
    return _normalize_aime_int(_coerce_int(m.group(1))) if m else None

def extract_direct_sentinel(text: str) -> Optional[int]:
    m = re.search(r"^__ANS__\s*=\s*(-?\d+)\s*$", text, flags=re.MULTILINE)
    return _normalize_aime_int(_coerce_int(m.group(1))) if m else None

def extract_answer_from_code_source(code_src: str) -> Optional[int]:
    # answer = <int>
    m = re.search(r"^\s*answer\s*=\s*(-?\d+)\s*$", code_src, flags=re.MULTILINE)
    if m:
        return _normalize_aime_int(_coerce_int(m.group(1)))
    # print("__ANS__=<int>") constant
    m2 = re.search(r'print\(\s*["\']__ANS__=\s*(-?\d+)["\']\s*\)', code_src)
    if m2:
        return _normalize_aime_int(_coerce_int(m2.group(1)))
    return None

def strict_single_sentinel(stdout_text: str) -> Optional[int]:
    lines = [ln.strip() for ln in stdout_text.strip().splitlines() if ln.strip()]
    hits = [ln for ln in lines if re.fullmatch(r"__ANS__\s*=\s*-?\d+", ln)]
    if not hits:
        return None
    return _normalize_aime_int(_coerce_int(hits[-1].split("=")[-1]))

def maybe_execute_for_last_resort(code_src: Optional[str]) -> Optional[int]:
    if not code_src:
        return None
    safe_globals = {
        "__builtins__": {
            "range": range, "len": len, "int": int, "float": float,
            "str": str, "list": list, "dict": dict, "set": set,
            "tuple": tuple, "sum": sum, "max": max, "min": min,
            "abs": abs, "round": round, "pow": pow, "print": print,
            "enumerate": enumerate, "zip": zip, "sorted": sorted,
            "reversed": reversed, "all": all, "any": any,
        },
        # Whitelist a few safe stdlib modules to enable stronger code paths:
        "math": math,
        "fractions": fractions,
        "itertools": itertools,
        "collections": collections,
        "functools": functools,
        "__name__": "__main__",
    }
    out_buf = io.StringIO()
    try:
        with redirect_stdout(out_buf):
            exec(code_src, safe_globals, {})
        printed = out_buf.getvalue()
        return strict_single_sentinel(printed)
    except Exception:
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
        # Fixed, generous but bounded generation budget
        # ↑ Slightly larger to let the Code method produce stronger algorithms.
        self.params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.15,  # stronger nudge against loops
            n=1,
            max_tokens=2000,          # give room for larger, more robust code
            stop=["assistantanalysis", "assistantfinal", "to=python"]
        )
        print("✓ Model loaded\n")
        try:
            self.tok = self.llm.get_tokenizer()
        except Exception:
            self.tok = None

    def generate_chat(self, messages: List[Dict[str, str]]) -> str:
        if self.tok is not None and hasattr(self.tok, "apply_chat_template"):
            prompt = self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback concatenation
            prompt = ""
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                prompt += f"<{role}>\n{content}\n</{role}>\n"
            prompt += "<assistant>\n"
        out = self.llm.generate([prompt], self.params)[0]
        return (out.outputs[0].text or "").strip() if out.outputs else ""


# ========================= Prompts =========================

def cot_prompt(problem: str) -> List[Dict[str, str]]:
    return [
        {"role": "system",
         "content": "You are an AIME solver. Be correct, concise, and deterministic."},
        {"role": "user",
         "content": f"""Solve the AIME problem with clear, compact steps and finish exactly with '**Final Answer: n**'.
Use ≤ 120 tokens for reasoning.

Problem:
{problem}
"""}]

def code_only_prompt(problem: str) -> List[Dict[str, str]]:
    """
    STRICT OUTPUT FORMAT (nothing else):
    1) __COT_ANS__=<integer>      ← your private micro-plan must yield this value
    2) PYCODE: ... ENDPYCODE       ← code prints exactly ONE line '__ANS__=<integer>'
    3) __PREDICTED_OUTPUT__=<int>  ← the number your program will print

    Code requirements (to increase accuracy):
    - Use exact integer math; avoid floats. You MAY import: math, fractions, itertools, collections, functools.
    - Prefer algebraic/number-theory derivation when reliable; otherwise use bounded brute force (≤ 1e7 ops).
    - Implement a verify(ans) that checks problem conditions exactly; assert it before printing.
    - If a closed-form attempt fails verification, fall back to brute force search with pruning.
    - Use helper utilities: gcd/lcm, modular arithmetic, combinatorial counting as needed.
    - Only ONE print in code (the sentinel). No input. No comments in the code. Fast (<1s typical AIME sizes).

    IMPORTANT: Always include all three items (COT_ANS line, PYCODE block, PREDICTED_OUTPUT line).
    """
    user_text = f"""Privately compute the exact integer answer (≤120 tokens). Do NOT reveal steps.
Then output EXACTLY in this order and nothing else:

__COT_ANS__=<integer>

PYCODE:
import math
from fractions import Fraction
from itertools import product, permutations, combinations

def verify(ans):
    # must return True iff ans satisfies the AIME problem's requirement exactly
    # implement deterministic checks only; no randomness
    return isinstance(ans, int) and 0 <= ans <= 999

def solve():
    # 1) Try a direct, exact derivation (integer/Fraction, modular arithmetic, etc.)
    # 2) If uncertain, do a bounded brute force with pruning (≤ 1e7 ops)
    # 3) Ensure ans is int in [0,999]
    ans = 0
    if not verify(ans):
        # fallback search skeleton (edit bounds as needed in your private thinking)
        best = None
        # example scaffold; adjust loops in your private reasoning
        for x in range(0, 1000):
            cand = x
            if verify(cand):
                best = cand
                break
        ans = 0 if best is None else int(best)
    assert isinstance(ans, int) and 0 <= ans <= 999 and verify(ans)
    return ans

answer = solve()
print("__ANS__=" + str(int(answer)))
ENDPYCODE

__PREDICTED_OUTPUT__=<integer>

Replace <integer> with the correct single number for this problem, and ensure PYCODE prints exactly one line.

Problem:
{problem}
"""
    return [
        {"role": "system",
         "content": "You are an AIME solver. Follow the format strictly; maximize correctness with exact math and verified fallbacks."},
        {"role": "user", "content": user_text}
    ]


# ========================= Evaluation =========================

def evaluate_problem(runner: QwenRunner, prob: Dict, index: int) -> Dict:
    q = prob["question"]
    true_ans = int(prob["answer"])

    # ---- CoT path ----
    cot_text = runner.generate_chat(cot_prompt(q))
    cot_pred = extract_final_answer(cot_text)

    # ---- Code path with stronger code + layered fallbacks ----
    code_text = runner.generate_chat(code_only_prompt(q))

    path = None
    po = extract_predicted_output(code_text)
    direct = extract_direct_sentinel(code_text)
    code_src = extract_code_block(code_text)
    src_parse = extract_answer_from_code_source(code_src) if code_src else None
    cot_commit = extract_cot_commit(code_text)

    # choose in order, validating AIME range at each step
    if po is not None:
        code_pred = po
        path = "PREDICTED"
    elif direct is not None:
        code_pred = direct
        path = "DIRECT_SENTINEL"
    elif src_parse is not None:
        code_pred = src_parse
        path = "CODE_PARSE"
    elif cot_commit is not None:
        code_pred = cot_commit
        path = "COT_COMMIT"
    else:
        exec_pred = maybe_execute_for_last_resort(code_src)
        if exec_pred is not None:
            code_pred = exec_pred
            path = "EXECUTE"
        elif cot_pred is not None:
            code_pred = cot_pred
            path = "COT_FALLBACK"
        else:
            code_pred = 0
            path = "ZERO_FALLBACK"

    print(f"[TRACE P{index:02d}] path={path} po={po} direct={direct} src={'Y' if code_src else 'N'} src_parse={src_parse} cot_commit={cot_commit} cot_pred={cot_pred}")

    return {
        "true": true_ans,
        "cot_pred": cot_pred,
        "code_pred": code_pred,
        "cot_ok": (cot_pred == true_ans),
        "code_ok": (code_pred == true_ans),
    }


def main():
    print("🔥 AIME 2024 — CoT + Code (ONE trial each), fixed budgets, stronger Code path")
    print("=" * 60)

    runner = QwenRunner("openai/gpt-oss-120b")
    problems = AIME_2024_PROBLEMS

    cot_correct = 0
    code_correct = 0
    total = len(problems)

    for i, prob in enumerate(problems, 1):
        res = evaluate_problem(runner, prob, i)
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
