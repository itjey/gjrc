#!/usr/bin/env python3
"""
Tree of Thoughts (ToT) AIME Problem Solver

Architecture:
1. Initial generation: 800 tokens of code (no comments)
2. First branching: 3 continuations from initial answer (800 tokens each)
3. Majority vote to select best first-level branch
4. Second branching: 3 final answers from best branch (full completion)
5. Final majority vote to determine winner
6. Extract answer with regex and validate
"""

import re
import io
import math
import fractions
import itertools
import collections
import functools
import traceback
from typing import Dict, Optional, List, Tuple
from contextlib import redirect_stdout
from collections import Counter

from vllm import LLM, SamplingParams

# Load AIME problems
try:
    from aime_problems import AIME_2024_PROBLEMS, AIME_2025_PROBLEMS
    ALL_PROBLEMS = AIME_2024_PROBLEMS + AIME_2025_PROBLEMS
    print(f"✓ Loaded {len(ALL_PROBLEMS)} AIME problems ({len(AIME_2024_PROBLEMS)} from 2024, {len(AIME_2025_PROBLEMS)} from 2025)")
except Exception as e:
    print(f"⚠️  WARNING: Could not load AIME problems ({e})")
    ALL_PROBLEMS = []


# ========================= Utilities =========================

def _coerce_int(x) -> Optional[int]:
    """Convert various types to int, return None if impossible."""
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
    """Clamp to valid AIME range [0, 999]."""
    if x is None:
        return None
    try:
        xi = int(x)
        if 0 <= xi <= 999:
            return xi
        return None
    except Exception:
        return None


def extract_answer_from_text(text: str) -> Optional[int]:
    """
    Extract final answer from code output using multiple regex patterns.
    Looks for common answer formats and the final numerical result.
    """
    # Pattern 1: answer = <number>
    m = re.search(r'^\s*answer\s*=\s*(-?\d+)\s*$', text, re.MULTILINE | re.IGNORECASE)
    if m:
        return _normalize_aime_int(_coerce_int(m.group(1)))
    
    # Pattern 2: result = <number>
    m = re.search(r'^\s*result\s*=\s*(-?\d+)\s*$', text, re.MULTILINE | re.IGNORECASE)
    if m:
        return _normalize_aime_int(_coerce_int(m.group(1)))
    
    # Pattern 3: print(<number>) at the end
    m = re.search(r'print\s*\(\s*(-?\d+)\s*\)', text)
    if m:
        return _normalize_aime_int(_coerce_int(m.group(1)))
    
    # Pattern 4: Look for last integer in the text (0-999 range)
    matches = re.findall(r'\b(\d{1,3})\b', text)
    if matches:
        # Try from the end
        for match in reversed(matches):
            val = _normalize_aime_int(_coerce_int(match))
            if val is not None:
                return val
    
    return None


def extract_code_from_text(text: str) -> str:
    """Extract code from text, removing markdown fences if present."""
    # Remove markdown code fences
    text = re.sub(r'^```python\s*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n```\s*$', '', text)
    return text.strip()


def safe_execute_code(code: str, timeout_seconds: int = 5, show_output: bool = False) -> Optional[int]:
    """
    Safely execute Python code and extract the answer.
    Returns the extracted answer or None if execution fails.
    """
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
        "fractions": fractions,
        "itertools": itertools,
        "collections": collections,
        "functools": functools,
        "__name__": "__main__",
    }
    
    out_buf = io.StringIO()
    try:
        with redirect_stdout(out_buf):
            exec(code, safe_globals, {})
        output = out_buf.getvalue()
        
        if show_output and output:
            print(f"    [Execution output]: {output.strip()}")
        
        # Try to extract answer from output
        answer = extract_answer_from_text(output)
        if answer is not None:
            if show_output:
                print(f"    [Extracted from output]: {answer}")
            return answer
        
        # If no answer in output, try to extract from code itself
        answer = extract_answer_from_text(code)
        if show_output and answer is not None:
            print(f"    [Extracted from code]: {answer}")
        return answer
        
    except Exception as e:
        if show_output:
            print(f"    [Execution failed]: {str(e)[:100]}")
        # If execution fails, try to extract answer from code source
        return extract_answer_from_text(code)


def majority_vote(answers: List[Optional[int]]) -> Optional[int]:
    """
    Perform majority voting on a list of answers.
    Returns the most common answer, or None if all are None.
    """
    # Filter out None values
    valid_answers = [a for a in answers if a is not None]
    
    if not valid_answers:
        return None
    
    # Count occurrences
    counter = Counter(valid_answers)
    
    # Return most common
    most_common = counter.most_common(1)[0][0]
    return most_common


# ========================= Model Wrapper =========================

class TreeOfThoughtsRunner:
    """Manages the Tree of Thoughts inference process."""
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-Coder-32B-Instruct"):
        print(f"🔥 Loading model: {model_path}")
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
        print("✓ Model loaded successfully\n")
        
        try:
            self.tok = self.llm.get_tokenizer()
        except Exception:
            self.tok = None
    
    def generate(self, prompt: str, max_tokens: int, temperature: float = 0.7) -> str:
        """Generate text with specified parameters."""
        params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            repetition_penalty=1.1,
        )
        
        outputs = self.llm.generate([prompt], params)
        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text.strip()
        return ""
    
    def initial_solution(self, question: str) -> str:
        """
        Generate initial solution (800 tokens, code only, no comments).
        """
        prompt = f"""You are solving an AIME mathematics problem. Write Python code to solve it.

Requirements:
- Write ONLY code, NO comments, NO explanations
- The code should be complete and executable
- Use proper mathematical libraries (math, fractions, itertools, etc.)
- Calculate the final answer and store it in a variable called 'answer'
- Print the answer at the end

Problem:
{question}

Code:"""
        
        return self.generate(prompt, max_tokens=800, temperature=0.7)
    
    def continue_solution(self, question: str, partial_code: str) -> str:
        """
        Continue from a partial solution (800 new tokens).
        """
        prompt = f"""You are solving an AIME mathematics problem. Below is partial code that was started.
Continue the code to complete the solution.

Requirements:
- Write ONLY code, NO comments, NO explanations
- Continue from where the partial code left off
- Complete the solution and calculate the final answer
- Store the answer in a variable called 'answer'
- Print the answer at the end

Problem:
{question}

Partial code:
{partial_code}

Continue the code:"""
        
        return self.generate(prompt, max_tokens=800, temperature=0.7)
    
    def finalize_solution(self, question: str, code_so_far: str) -> str:
        """
        Finalize the solution with full generation.
        """
        prompt = f"""You are solving an AIME mathematics problem. Below is code that was developed so far.
Complete and finalize the solution.

Requirements:
- Write ONLY code, NO comments, NO explanations
- Continue/fix the code to get the final answer
- Store the answer in a variable called 'answer'
- Print the answer at the end

Problem:
{question}

Code so far:
{code_so_far}

Final code:"""
        
        return self.generate(prompt, max_tokens=1500, temperature=0.7)


# ========================= Tree of Thoughts Logic =========================

def solve_with_tot(runner: TreeOfThoughtsRunner, question: str, true_answer: str, problem_idx: int) -> Dict:
    """
    Solve a single problem using Tree of Thoughts.
    
    Steps:
    1. Generate initial 800-token solution
    2. Branch into 3 continuations (800 tokens each)
    3. Majority vote on first level
    4. Branch into 3 final solutions from winner
    5. Majority vote on final level
    6. Extract and validate answer
    """
    print(f"\n{'='*80}")
    print(f"Problem {problem_idx}: {question[:100]}...")
    print(f"True answer: {true_answer}")
    print(f"{'='*80}")
    
    # Step 1: Initial solution (800 tokens)
    print("\n[Step 1] Generating initial solution (800 tokens)...")
    initial_code = runner.initial_solution(question)
    print(f"Initial code length: {len(initial_code)} chars")
    print("\n--- Initial Code Output ---")
    print(initial_code)
    print("--- End Initial Code ---\n")
    
    # Step 2: Generate 3 continuations (first branching)
    print("\n[Step 2] First branching: 3 continuations from initial...")
    first_level_branches = []
    first_level_answers = []
    
    for i in range(3):
        print(f"\n  Branch 1.{i+1}: Generating continuation...")
        continuation = runner.continue_solution(question, initial_code)
        print(f"\n  --- Branch 1.{i+1} Continuation Output ---")
        print(continuation)
        print(f"  --- End Branch 1.{i+1} Continuation ---\n")
        
        full_code = initial_code + "\n" + continuation
        full_code = extract_code_from_text(full_code)
        first_level_branches.append(full_code)
        
        # Try to extract answer
        print(f"  Branch 1.{i+1}: Executing code...")
        answer = safe_execute_code(full_code, show_output=True)
        first_level_answers.append(answer)
        print(f"  Branch 1.{i+1} final extracted answer: {answer}")
    
    # Step 3: Majority vote on first level
    print("\n[Step 3] First level majority vote...")
    first_vote_winner = majority_vote(first_level_answers)
    print(f"  First level winner: {first_vote_winner}")
    print(f"  Answers were: {first_level_answers}")
    
    # Find the branch that produced the winning answer
    if first_vote_winner is not None:
        try:
            winner_idx = first_level_answers.index(first_vote_winner)
            winning_branch = first_level_branches[winner_idx]
        except ValueError:
            # Fallback: use first branch if winner not found
            winning_branch = first_level_branches[0]
    else:
        winning_branch = first_level_branches[0]
    
    print(f"  Using branch from continuation {winner_idx + 1 if first_vote_winner is not None else 1}")
    
    # Step 4: Generate 3 final solutions (second branching)
    print("\n[Step 4] Second branching: 3 final solutions...")
    second_level_branches = []
    second_level_answers = []
    
    for i in range(3):
        print(f"\n  Branch 2.{i+1}: Generating final solution...")
        final_code = runner.finalize_solution(question, winning_branch)
        print(f"\n  --- Branch 2.{i+1} Final Solution Output ---")
        print(final_code)
        print(f"  --- End Branch 2.{i+1} Final Solution ---\n")
        
        final_code = extract_code_from_text(final_code)
        second_level_branches.append(final_code)
        
        # Try to extract answer
        print(f"  Branch 2.{i+1}: Executing code...")
        answer = safe_execute_code(final_code, show_output=True)
        second_level_answers.append(answer)
        print(f"  Branch 2.{i+1} final extracted answer: {answer}")
    
    # Step 5: Final majority vote
    print("\n[Step 5] Final majority vote...")
    final_answer = majority_vote(second_level_answers)
    print(f"  Final answer: {final_answer}")
    print(f"  Answers were: {second_level_answers}")
    
    # Step 6: Validate
    true_ans_int = _normalize_aime_int(_coerce_int(true_answer))
    is_correct = (final_answer == true_ans_int)
    
    print(f"\n{'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
    print(f"  Predicted: {final_answer}")
    print(f"  True: {true_ans_int}")
    
    return {
        "problem_idx": problem_idx,
        "question": question,
        "true_answer": true_ans_int,
        "predicted_answer": final_answer,
        "correct": is_correct,
        "first_level_answers": first_level_answers,
        "second_level_answers": second_level_answers,
    }


# ========================= Main Execution =========================

def main():
    print("="*80)
    print("Tree of Thoughts AIME Problem Solver")
    print("="*80)
    
    # Initialize model
    runner = TreeOfThoughtsRunner(model_path="Qwen/Qwen2.5-Coder-32B-Instruct")
    
    # Track results
    results = []
    correct_count = 0
    total_count = 0
    
    # Solve each problem
    for idx, problem in enumerate(ALL_PROBLEMS, 1):
        try:
            result = solve_with_tot(
                runner=runner,
                question=problem["question"],
                true_answer=problem["answer"],
                problem_idx=idx
            )
            results.append(result)
            
            if result["correct"]:
                correct_count += 1
            total_count += 1
            
        except Exception as e:
            print(f"\n❌ Error on problem {idx}: {e}")
            traceback.print_exc()
            results.append({
                "problem_idx": idx,
                "question": problem["question"],
                "true_answer": problem["answer"],
                "predicted_answer": None,
                "correct": False,
                "error": str(e)
            })
            total_count += 1
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    print(f"\nTotal problems: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {total_count - correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Detailed results
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    for result in results:
        status = "✓" if result["correct"] else "✗"
        pred = result.get("predicted_answer", "N/A")
        true = result.get("true_answer", "N/A")
        print(f"{status} Problem {result['problem_idx']:2d}: Predicted={pred}, True={true}")
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()

