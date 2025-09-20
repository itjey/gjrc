#!/usr/bin/env python3

import re
import json
import os
import sys
import math
import traceback
import io
import signal
from typing import List, Dict, Optional
from datetime import datetime
from collections import Counter
from contextlib import redirect_stdout

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# NOTE: You must have a file named 'aime_problems.py' in the same directory
# with the following structure:
# AIME_2024_PROBLEMS = [{"question": "...", "answer": "..."}, ...]
# AIME_2025_PROBLEMS = [{"question": "...", "answer": "..."}, ...]
from aime_problems import AIME_2024_PROBLEMS, AIME_2025_PROBLEMS


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Code execution timed out")


class GPTRunner:
    def __init__(self, model_path: str = "gpt-oss-20b"):
        # Convert relative path to absolute path to ensure it's recognized as a local path
        if model_path.startswith("./"):
            model_path = os.path.abspath(model_path)
        elif not os.path.isabs(model_path):
            # If it's just a directory name, make it an absolute path
            model_path = os.path.abspath(model_path)
            
        self.model_path = model_path
        print(f"Loading GPT-OSS-20B from: {model_path}")
        
        # Check if the path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
            local_files_only=True  # Force local loading
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading model (this may take a minute)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=True,
            local_files_only=True  # Force local loading
        )
        self.model.eval()
        print("Model loaded successfully!")
        print(f"Model device: {next(self.model.parameters()).device}")

    def generate_response(self, prompt: str, max_new_tokens: int = 1536) -> str:
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.8,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )

            input_length = inputs['input_ids'].shape[1]
            response = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            return response.strip()
        except Exception as e:
            print(f"Generation error: {e}")
            return ""


class AIMEEvaluator:
    def __init__(self, model_runner: GPTRunner):
        self.model = model_runner
        # Define the different iteration counts to test
        self.iteration_counts = [1, 5, 10, 16]

    def create_math_examples(self) -> str:
        return """Here are examples of how to solve AIME problems:

**Example 1:**
Problem: Find the number of ordered pairs $(a,b)$ of integers such that $|a + bi| = 5$.

Solution: I need to find pairs $(a,b)$ where $|a + bi| = 5$.
This means $\\sqrt{a^2 + b^2} = 5$, so $a^2 + b^2 = 25$.

I'll find all integer solutions systematically:
- $a = 0$: $b^2 = 25 \\Rightarrow b = \\pm 5$ → 2 pairs: $(0,5), (0,-5)$
- $a = \\pm 3$: $b^2 = 16 \\Rightarrow b = \\pm 4$ → 4 pairs: $(\\pm 3, \\pm 4)$
- $a = \\pm 4$: $b^2 = 9 \\Rightarrow b = \\pm 3$ → 4 pairs: $(\\pm 4, \\pm 3)$
- $a = \\pm 5$: $b^2 = 0 \\Rightarrow b = 0$ → 2 pairs: $(\\pm 5, 0)$

Total: $2 + 4 + 4 + 2 = 12$ pairs.
**Final Answer: 12**

**Example 2:**
Problem: Find the remainder when $2^{100}$ is divided by 125.

Solution: I'll use Euler's theorem. First, $125 = 5^3$ and $\\gcd(2, 125) = 1$.
By Euler's theorem: $\\phi(125) = 125(1 - 1/5) = 100$.
So $2^{100} \\equiv 1 \\pmod{125}$.
**Final Answer: 1**

Now solve this problem step by step:"""

    def create_code_prompt(self, problem: str) -> str:
        return f"""Solve this AIME problem using Python code. Write clean, executable code that computes the answer.

Problem: {problem}

Requirements:
- You have access to standard Python built-in functions and the 'math' library. Do not import any other libraries.
- Show your reasoning in comments.
- Store the final answer in a variable called 'answer'.
- Add a comment with your final answer just before the last line, for example: '# Final Answer: 123'
- Print the final integer answer at the end.
- The answer must be an integer from 0 to 999.

```python"""

    def create_cot_prompt(self, problem: str) -> str:
        return f"""{self.create_math_examples()}

Problem: {problem}

Solve this step-by-step. Show all work clearly and end with "**Final Answer: [number]**" where [number] is your final integer answer from 0 to 999."""

    def create_hybrid_prompt(self, problem: str) -> str:
        return f"""Solve this AIME problem by first reasoning through it mathematically, then writing Python code to verify your answer.

Problem: {problem}

Format:
1. **Mathematical Reasoning:** [Explain your approach]
2. **Python Verification:** [Write code to compute/verify. In your code, include a comment with the computed answer, like '# Final Answer: 123']
3. **Final Answer: [number]**

Mathematical Reasoning:"""

    def extract_answer_robust(self, text: str) -> Optional[int]:
        patterns = [
            # New pattern to find the answer in a comment
            r"#\s*Final Answer[:\s]*(\d{1,3})\b",
            # Existing patterns
            r"\*\*Final Answer[:\s]*(\d{1,3})\*\*",
            r"Final Answer[:\s]*(\d{1,3})",
            r"The final answer is \.*?(\d{1,3})",
            r"answer is[:\s]*(\d{1,3})",
            r"answer\s*=\s*(\d{1,3})",
            r"print\s*\(\s*(\d{1,3})\s*\)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                try:
                    answer = int(matches[-1])
                    if 0 <= answer <= 999:
                        return answer
                except (ValueError, TypeError):
                    continue

        lines = text.split('\n')
        for line in reversed(lines[-10:]):
            numbers = re.findall(r'\b(\d{1,3})\b', line)
            for num_str in reversed(numbers):
                try:
                    num = int(num_str)
                    if 0 <= num <= 999:
                        return num
                except (ValueError, TypeError):
                    continue

        return None

    def execute_python_code(self, code_text: str, timeout: int = 5) -> Optional[int]:
        code_match = re.search(r'```python\s*\n?(.*?)```', code_text, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            code = code_text.strip()

        if not code:
            return None

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        output_buffer = io.StringIO()

        try:
            safe_globals = {
                '__builtins__': {
                    'range': range, 'len': len, 'int': int, 'float': float,
                    'str': str, 'list': list, 'dict': dict, 'set': set,
                    'sum': sum, 'max': max, 'min': min, 'abs': abs,
                    'round': round, 'pow': pow, 'print': print,
                    'enumerate': enumerate, 'zip': zip, 'sorted': sorted,
                },
                'math': math
            }
            safe_locals = {}

            with redirect_stdout(output_buffer):
                exec(code, safe_globals, safe_locals)
            
            signal.alarm(0)

            answer_vars = ['answer', 'result', 'final_answer', 'ans', 'solution']
            for var in answer_vars:
                if var in safe_locals:
                    try:
                        val = safe_locals[var]
                        answer = int(round(float(val)))
                        if 0 <= answer <= 999:
                            return answer
                    except (ValueError, TypeError):
                        continue
            
            output = output_buffer.getvalue().strip()
            if output:
                numbers = re.findall(r'\b(\d+)\b', output)
                if numbers:
                    try:
                        answer = int(numbers[-1])
                        if 0 <= answer <= 999:
                            return answer
                    except (ValueError, TypeError):
                        pass

            return None

        except TimeoutException as e:
            print(f"    Code execution error: {e}")
            return None
        except Exception as e:
            print(f"    Code execution error: {type(e).__name__}: {e}")
            return None
        finally:
            signal.alarm(0)

    def evaluate_single_method_iterations(self, problem: Dict, method: str, attempts: int) -> Dict:
        """Evaluate a single method with specified number of iterations"""
        question = problem['question']
        true_answer = int(problem['answer'])
        
        results = []
        correct_count = 0

        for attempt in range(attempts):
            try:
                if method == "cot":
                    prompt = self.create_cot_prompt(question)
                elif method == "code":
                    prompt = self.create_code_prompt(question)
                elif method == "hybrid":
                    prompt = self.create_hybrid_prompt(question)
                else:
                    raise ValueError(f"Unknown method: {method}")

                response = self.model.generate_response(prompt)
                if not response:
                    results.append({"attempt": attempt+1, "answer": None, "correct": False, "error": "No response"})
                    continue

                if method in ["code", "hybrid"]:
                    answer = self.execute_python_code(response)
                    if answer is None:
                        answer = self.extract_answer_robust(response)
                else:
                    answer = self.extract_answer_robust(response)

                is_correct = answer == true_answer if answer is not None else False
                if is_correct:
                    correct_count += 1
                    
                results.append({
                    "attempt": attempt + 1,
                    "raw_response": response[:300] + "..." if len(response) > 300 else response,
                    "extracted_answer": answer,
                    "correct": is_correct
                })

            except Exception as e:
                results.append({"attempt": attempt+1, "error": str(e), "extracted_answer": None, "correct": False})

        # Determine final answer using majority vote or first correct answer
        correct_results = [r for r in results if r.get("correct", False)]
        if correct_results:
            final_answer = correct_results[0]["extracted_answer"]
            final_correct = True
        else:
            valid_answers = [r["extracted_answer"] for r in results if r.get("extracted_answer") is not None]
            if valid_answers:
                # Use majority vote
                answer_counter = Counter(valid_answers)
                final_answer = answer_counter.most_common(1)[0][0]
                final_correct = final_answer == true_answer
            else:
                final_answer = None
                final_correct = False

        success_rate = correct_count / attempts if attempts > 0 else 0
        
        return {
            "method": method,
            "iterations": attempts,
            "attempts": results,
            "correct_count": correct_count,
            "final_answer": final_answer,
            "correct": final_correct,
            "success_rate": success_rate
        }

    def evaluate_problem_all_iterations(self, problem: Dict, problem_num: int) -> Dict:
        """Evaluate a single problem with all methods and all iteration counts"""
        question = problem['question']
        true_answer = int(problem['answer'])
        print(f"\nProblem {problem_num}")
        print(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        print(f"Expected Answer: {true_answer}")

        methods = ["cot", "code", "hybrid"]
        all_results = {}
        
        for method in methods:
            print(f"  Method: {method.upper()}")
            method_results = {}
            
            for iteration_count in self.iteration_counts:
                print(f"    Running {iteration_count} iteration(s)...")
                try:
                    result = self.evaluate_single_method_iterations(problem, method, iteration_count)
                    method_results[f"{iteration_count}_iterations"] = result
                    
                    status = "CORRECT" if result["correct"] else "INCORRECT"
                    print(f"      Result: {status} ({result['correct_count']}/{iteration_count} correct attempts)")
                    print(f"      Final Answer: {result['final_answer']} (Expected: {true_answer})")
                    
                except Exception as e:
                    print(f"      Error in {iteration_count} iterations: {e}")
                    method_results[f"{iteration_count}_iterations"] = {
                        "method": method, 
                        "iterations": iteration_count,
                        "error": str(e), 
                        "correct": False, 
                        "success_rate": 0,
                        "correct_count": 0
                    }
            
            all_results[method] = method_results

        # Determine if problem was solved by any method/iteration combination
        any_solved = any(
            result.get("correct", False) 
            for method_results in all_results.values() 
            for result in method_results.values()
        )
        
        # Find best performing method/iteration combination
        best_performance = {"method": None, "iterations": 0, "success_rate": 0}
        for method, method_results in all_results.items():
            for iteration_key, result in method_results.items():
                if result.get("success_rate", 0) > best_performance["success_rate"]:
                    best_performance = {
                        "method": method,
                        "iterations": result.get("iterations", 0),
                        "success_rate": result.get("success_rate", 0)
                    }

        status = "SOLVED" if any_solved else "UNSOLVED"
        print(f"  Overall Result: {status}")
        if best_performance["method"]:
            print(f"  Best Performance: {best_performance['method']} with {best_performance['iterations']} iterations ({best_performance['success_rate']:.1%} success rate)")

        return {
            "problem_number": problem_num,
            "question": question,
            "true_answer": true_answer,
            "methods": all_results,
            "solved": any_solved,
            "best_performance": best_performance
        }

    def evaluate_dataset(self, problems: List[Dict], dataset_name: str) -> Dict:
        print(f"\n{'='*80}\nEVALUATING {dataset_name.upper()} DATASET\n{'='*80}")
        print(f"Total Problems: {len(problems)}")
        print(f"Iteration counts to test: {self.iteration_counts}")

        results = []
        solved_count = 0
        
        for i, problem in enumerate(problems, 1):
            try:
                result = self.evaluate_problem_all_iterations(problem, i)
                results.append(result)
                if result["solved"]:
                    solved_count += 1
                    
                # Save progress every 3 problems
                if i % 3 == 0:
                    self.save_progress(results, dataset_name, i)
                    
            except Exception as e:
                print(f"Failed on problem {i}: {e}")
                traceback.print_exc()
                results.append({
                    "problem_number": i, 
                    "question": problem.get('question', 'Unknown'), 
                    "true_answer": int(problem.get('answer', -1)), 
                    "error": str(e), 
                    "solved": False
                })

        accuracy = (solved_count / len(problems)) * 100 if problems else 0
        print(f"\n{dataset_name} Complete: {solved_count}/{len(problems)} solved ({accuracy:.1f}%)")

        return {
            "dataset_name": dataset_name,
            "total_problems": len(problems),
            "solved_problems": solved_count,
            "accuracy_percent": round(accuracy, 1),
            "iteration_counts_tested": self.iteration_counts,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    def save_progress(self, results: List[Dict], dataset_name: str, progress: int):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gpt_multi_iter_progress_{dataset_name.lower()}_{progress}_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Progress saved to {filename}")

    def generate_summary(self, all_results: List[Dict]) -> Dict:
        total_problems = sum(r["total_problems"] for r in all_results)
        total_solved = sum(r["solved_problems"] for r in all_results)
        overall_accuracy = (total_solved / total_problems * 100) if total_problems > 0 else 0

        # Collect statistics for each method and iteration count
        method_iteration_stats = {}
        for method in ["cot", "code", "hybrid"]:
            method_iteration_stats[method] = {}
            for iteration_count in self.iteration_counts:
                method_iteration_stats[method][f"{iteration_count}_iterations"] = {
                    "total_problems": 0,
                    "solved_problems": 0,
                    "total_correct_attempts": 0,
                    "total_attempts": 0
                }

        # Aggregate statistics
        for dataset_result in all_results:
            for problem_result in dataset_result.get("results", []):
                if "methods" in problem_result:
                    for method, method_results in problem_result["methods"].items():
                        for iteration_key, iteration_result in method_results.items():
                            if iteration_key in method_iteration_stats[method]:
                                stats = method_iteration_stats[method][iteration_key]
                                stats["total_problems"] += 1
                                if iteration_result.get("correct", False):
                                    stats["solved_problems"] += 1
                                stats["total_correct_attempts"] += iteration_result.get("correct_count", 0)
                                stats["total_attempts"] += iteration_result.get("iterations", 0)

        # Calculate final statistics
        method_summary = {}
        for method, iteration_data in method_iteration_stats.items():
            method_summary[method] = {}
            for iteration_key, stats in iteration_data.items():
                if stats["total_problems"] > 0:
                    problem_solve_rate = stats["solved_problems"] / stats["total_problems"]
                    attempt_success_rate = stats["total_correct_attempts"] / stats["total_attempts"] if stats["total_attempts"] > 0 else 0
                    method_summary[method][iteration_key] = {
                        "problems_solved": f"{stats['solved_problems']}/{stats['total_problems']}",
                        "problem_solve_rate": round(problem_solve_rate * 100, 1),
                        "attempt_success_rate": round(attempt_success_rate * 100, 1),
                        "total_correct_attempts": stats["total_correct_attempts"],
                        "total_attempts": stats["total_attempts"]
                    }

        return {
            "evaluation_summary": {
                "total_problems": total_problems,
                "total_solved": total_solved,
                "overall_accuracy": round(overall_accuracy, 1),
                "iteration_counts_tested": self.iteration_counts,
                "dataset_breakdown": {
                    r["dataset_name"]: f"{r['solved_problems']}/{r['total_problems']} ({r['accuracy_percent']}%)" 
                    for r in all_results
                },
                "method_iteration_performance": method_summary,
                "evaluation_timestamp": datetime.now().isoformat()
            }
        }


def main():
    print("Starting GPT-OSS-20B Multi-Iteration AIME Evaluation")
    print("This will test 1, 5, 10, and 16 iterations for each method on each problem")
    
    # Try different possible locations for the model
    model_paths = [
        "gpt-oss-20b",           # Direct directory name
        "./gpt-oss-20b",         # Relative path
        "/workspace/gpt-oss-20b" # Absolute path
    ]
    
    model = None
    for path in model_paths:
        if os.path.exists(path):
            print(f"Found model directory at: {path}")
            try:
                model = GPTRunner(path)
                break
            except Exception as e:
                print(f"Failed to load model from {path}: {e}")
                continue
    
    if model is None:
        print("Failed to load model from any of the expected locations.")
        print("Please ensure the model is downloaded and located in one of these paths:")
        for path in model_paths:
            print(f"  - {path}")
        return

    evaluator = AIMEEvaluator(model)
    datasets = [
        (AIME_2024_PROBLEMS, "AIME_2024"),
        (AIME_2025_PROBLEMS, "AIME_2025")
    ]
    all_results = []

    for problems, dataset_name in datasets:
        try:
            result = evaluator.evaluate_dataset(problems, dataset_name)
            all_results.append(result)
        except Exception as e:
            print(f"Failed to evaluate {dataset_name}: {e}")
            traceback.print_exc()

    if all_results:
        summary = evaluator.generate_summary(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"gpt_multi_iteration_aime_results_{timestamp}.json"
        
        final_output = {
            "model_info": {
                "model_name": "GPT-OSS-20B", 
                "model_path": model.model_path,
                "iteration_counts_tested": evaluator.iteration_counts
            },
            "datasets": all_results,
            "summary": summary
        }
        
        with open(final_filename, 'w') as f:
            json.dump(final_output, f, indent=2)

        print(f"\n{'='*80}\nEVALUATION COMPLETED!\n{'='*80}")
        print(f"Results saved to: {final_filename}")
        
        summary_data = summary["evaluation_summary"]
        print(f"Overall: {summary_data['total_solved']}/{summary_data['total_problems']} problems solved ({summary_data['overall_accuracy']}%)")
        
        print(f"Iteration counts tested: {summary_data['iteration_counts_tested']}")
        
        for dataset, performance in summary_data["dataset_breakdown"].items():
            print(f"   - {dataset}: {performance}")
        
        print("\nMethod and Iteration Performance:")
        for method, iteration_data in summary_data.get("method_iteration_performance", {}).items():
            print(f"   {method.upper()}:")
            for iteration_key, stats in iteration_data.items():
                iteration_count = iteration_key.replace("_iterations", "")
                print(f"     {iteration_count} iterations: {stats['problems_solved']} problems solved "
                      f"({stats['problem_solve_rate']}% problem solve rate, "
                      f"{stats['attempt_success_rate']}% attempt success rate)")
        
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
