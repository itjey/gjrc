#!/usr/bin/env python3

import re
import json
import os
import sys
import math
import traceback
import io
import signal
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import statistics

from vllm import LLM, SamplingParams

# Import problems from the provided dataset
# Make sure you have a file named aime_problems.py with these lists, for example:
# AIME_2024_PROBLEMS = [{'question': 'Problem 1...', 'answer': '123'}, ...]
# AIME_2025_PROBLEMS = [{'question': 'Problem 1...', 'answer': '456'}, ...]
from aime_problems import AIME_2024_PROBLEMS, AIME_2025_PROBLEMS


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Code execution timed out")


class QwenRunner:
    def __init__(self, model_path: str = "Qwen/Qwen3-8B"):
        self.model_path = model_path
        print(f"Loading {model_path} with vLLM...")
        
        # Optimized vLLM configuration for Qwen3-8B
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=16384,  # 16K context
            dtype="bfloat16",
            enforce_eager=False,
            disable_log_stats=True,
            enable_prefix_caching=True,
            max_num_seqs=256,  # Good for multi-pass batching
        )
        
        print("✓ vLLM model loaded successfully!")

    def generate_responses_batch(self, prompts: List[str], temperature: float, 
                               max_tokens: int, num_samples: int = 1) -> List[List[str]]:
        """Generate multiple responses for each prompt"""
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.8,
            repetition_penalty=1.05,
            max_tokens=max_tokens,
            n=num_samples,  # Generate multiple samples per prompt
        )
        
        try:
            outputs = self.llm.generate(prompts, sampling_params)
            results = []
            for output in outputs:
                prompt_results = []
                for choice in output.outputs:
                    prompt_results.append(choice.text.strip())
                results.append(prompt_results)
            return results
        except Exception as e:
            print(f"Error in batch generation: {e}")
            return [[""] * num_samples for _ in prompts]


class AIMEEvaluator:
    def __init__(self, model_runner: QwenRunner):
        self.model = model_runner
        # Fixed parameters for consistency and reproducibility
        self.temperature = 0.3  # Fixed optimal temperature
        self.max_tokens = 2048  # Fixed optimal token limit
        self.pass_configs = [1, 5, 10, 16]  # Only vary number of passes
        
    def create_math_examples(self) -> str:
        return """Here are examples of how to solve AIME problems step by step:

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
- Show your reasoning in comments as you code.
- Store the final answer in a variable called 'answer'.
- Print the final integer answer at the end.
- The answer must be an integer from 0 to 999.
- Be thorough in your solution and verify your logic.

```python"""

    def create_cot_prompt(self, problem: str) -> str:
        return f"""{self.create_math_examples()}

Problem: {problem}

Solve this step-by-step. Show all mathematical work clearly and end with "**Final Answer: [number]**" where [number] is your final integer answer from 0 to 999."""

    def create_hybrid_prompt(self, problem: str) -> str:
        return f"""Solve this AIME problem by first reasoning through it mathematically, then writing Python code to verify your answer.

Problem: {problem}

Format your solution as follows:

1. **Mathematical Reasoning:**
[Explain your mathematical approach step by step]

2. **Python Verification:**
```python
# Verify the mathematical solution with code
# [Your verification code here]
# Store final answer in variable 'answer'
print(answer)  # Final answer
```

3. **Final Answer: [number]**

Begin with your mathematical reasoning:"""

    def extract_answer_robust(self, text: str) -> Optional[int]:
        """Enhanced answer extraction with multiple patterns"""
        patterns = [
            r"#\s*Final Answer[:\s]*(\d{1,3})\b",
            r"\*\*Final Answer[:\s]*(\d{1,3})\*\*",
            r"Final Answer[:\s]*(\d{1,3})",
            r"The final answer is[:\s]*(\d{1,3})",
            r"answer is[:\s]*(\d{1,3})",
            r"answer\s*=\s*(\d{1,3})",
            r"print\s*\(\s*(\d{1,3})\s*\)",
            r"result\s*=\s*(\d{1,3})",
            r"solution\s*=\s*(\d{1,3})",
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

        # Enhanced fallback: look for numbers in context
        lines = text.split('\n')
        for line in reversed(lines[-15:]):  # Look at more lines
            # Skip lines that are clearly not answers
            if any(skip_word in line.lower() for skip_word in 
                  ['example', 'step', 'let', 'first', 'next', 'then', 'also']):
                continue
            
            numbers = re.findall(r'\b(\d{1,3})\b', line)
            for num_str in reversed(numbers):
                try:
                    num = int(num_str)
                    if 0 <= num <= 999:
                        return num
                except (ValueError, TypeError):
                    continue

        return None

    def execute_python_code(self, code_text: str, timeout: int = 15) -> Optional[int]:
        """Enhanced code execution with better error handling"""
        code_match = re.search(r'```python\s*\n?(.*?)```', code_text, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            # Try to extract code without markdown
            lines = code_text.split('\n')
            code_lines = []
            in_code = False
            for line in lines:
                if line.strip().startswith('#') or any(keyword in line for keyword in 
                    ['import', 'def ', 'for ', 'while ', 'if ', '=', 'print']):
                    in_code = True
                if in_code:
                    code_lines.append(line)
            code = '\n'.join(code_lines)

        if not code or len(code.strip()) < 10:
            return None

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        output_buffer = io.StringIO()

        try:
            safe_globals = {
                '__builtins__': {
                    'range': range, 'len': len, 'int': int, 'float': float,
                    'str': str, 'list': list, 'dict': dict, 'set': set,
                    'tuple': tuple, 'sum': sum, 'max': max, 'min': min, 
                    'abs': abs, 'round': round, 'pow': pow, 'print': print,
                    'enumerate': enumerate, 'zip': zip, 'sorted': sorted,
                    'reversed': reversed, 'all': all, 'any': any,
                },
                'math': math
            }
            safe_locals = {}

            with redirect_stdout(output_buffer):
                exec(code, safe_globals, safe_locals)
            
            signal.alarm(0)

            # Check for answer variable (priority order)
            answer_vars = ['answer', 'result', 'final_answer', 'ans', 'solution', 'output']
            for var in answer_vars:
                if var in safe_locals:
                    try:
                        val = safe_locals[var]
                        answer = int(round(float(val)))
                        if 0 <= answer <= 999:
                            return answer
                    except (ValueError, TypeError):
                        continue
            
            # Check output for final answer
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

        except TimeoutException:
            return None
        except Exception:
            return None
        finally:
            signal.alarm(0)

    def save_progress(self, all_results: List[Dict], dataset_name: str, problem_idx: int):
        """Save intermediate progress"""
        filename = f"progress_{dataset_name}_{problem_idx}.json"
        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  Progress saved to {filename}")

    def evaluate_multipass(self, problems: List[Dict], dataset_name: str) -> Dict:
        """Simplified multi-pass evaluation with fixed temperature and tokens"""
        print(f"\n{'='*80}\nMULTI-PASS EVALUATION: {dataset_name.upper()}\n{'='*80}")
        print(f"Total Problems: {len(problems)}")
        print(f"Methods: CoT, Code, Hybrid")
        print(f"Temperature: {self.temperature} (fixed)")
        print(f"Max Tokens: {self.max_tokens} (fixed)")
        print(f"Passes: {self.pass_configs}")
        
        total_configs = len(self.pass_configs) * 3  # 3 methods
        print(f"Total configurations per problem: {total_configs}")
        
        all_results = []
        
        for problem_idx, problem in enumerate(problems, 1):
            print(f"\nProblem {problem_idx}/{len(problems)}")
            problem_results = self.evaluate_single_problem_multipass(problem, problem_idx)
            all_results.append(problem_results)
            
            # Save progress every 5 problems
            if problem_idx % 5 == 0:
                self.save_progress(all_results, dataset_name, problem_idx)
        
        return {
            "dataset_name": dataset_name,
            "total_problems": len(problems),
            "results": all_results,
            "timestamp": datetime.now().isoformat(),
            "configuration_summary": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "passes": self.pass_configs,
                "total_configs_per_problem": total_configs
            }
        }

    def evaluate_single_problem_multipass(self, problem: Dict, problem_num: int) -> Dict:
        """Evaluate a single problem with multiple passes"""
        question = problem['question']
        true_answer = int(problem['answer'])
        
        print(f"  Question: {question[:100]}...")
        print(f"  Expected: {true_answer}")
        
        methods = {
            'cot': self.create_cot_prompt,
            'code': self.create_code_prompt, 
            'hybrid': self.create_hybrid_prompt
        }
        
        method_results = {}
        
        for method_name, prompt_func in methods.items():
            print(f"  Evaluating {method_name.upper()}...")
            method_results[method_name] = {}
            
            base_prompt = prompt_func(question)
            
            for num_passes in self.pass_configs:
                config_key = f"passes_{num_passes}"
                
                # Generate multiple responses
                responses = self.model.generate_responses_batch(
                    [base_prompt], temperature=self.temperature, 
                    max_tokens=self.max_tokens, num_samples=num_passes
                )[0]  # Get responses for the single prompt
                
                # Extract answers from all responses
                extracted_answers = []
                correct_answers = []
                
                for response in responses:
                    if method_name in ['code', 'hybrid']:
                        answer = self.execute_python_code(response)
                        if answer is None:
                            answer = self.extract_answer_robust(response)
                    else:
                        answer = self.extract_answer_robust(response)
                    
                    extracted_answers.append(answer)
                    if answer == true_answer:
                        correct_answers.append(answer)
                
                # Calculate statistics
                valid_answers = [a for a in extracted_answers if a is not None]
                success_rate = len(correct_answers) / len(responses) * 100 if responses else 0
                extraction_rate = len(valid_answers) / len(responses) * 100 if responses else 0
                
                # Find most common answer (mode)
                if valid_answers:
                    answer_counts = Counter(valid_answers)
                    mode_answer = answer_counts.most_common(1)[0][0]
                    mode_is_correct = mode_answer == true_answer
                else:
                    mode_answer = None
                    mode_is_correct = False
                
                # Best single answer (any correct answer)
                best_single_correct = len(correct_answers) > 0
                
                method_results[method_name][config_key] = {
                    'num_passes': num_passes,
                    'responses': responses,
                    'extracted_answers': extracted_answers,
                    'valid_answers': valid_answers,
                    'correct_answers': correct_answers,
                    'success_rate': success_rate,
                    'extraction_rate': extraction_rate,
                    'best_single_correct': best_single_correct,
                    'mode_answer': mode_answer,
                    'mode_is_correct': mode_is_correct,
                    'answer_distribution': dict(Counter(valid_answers)) if valid_answers else {},
                    'response_lengths': [len(r) for r in responses],
                    'avg_response_length': statistics.mean([len(r) for r in responses]) if responses else 0
                }
                
                status_single = "✓" if best_single_correct else "✗"
                status_mode = "✓" if mode_is_correct else "✗" 
                print(f"    P={num_passes}: Single={status_single} Mode={status_mode} Success={success_rate:.1f}%")
        
        # Find best configurations
        best_single_config = None
        best_mode_config = None
        best_success_rate_config = None
        
        for method_name, method_data in method_results.items():
            for config_key, config_data in method_data.items():
                if config_data['best_single_correct'] and best_single_config is None:
                    best_single_config = {
                        'method': method_name,
                        'config': config_key,
                        'num_passes': config_data['num_passes'],
                        'success_rate': config_data['success_rate']
                    }
                
                if config_data['mode_is_correct'] and best_mode_config is None:
                    best_mode_config = {
                        'method': method_name,
                        'config': config_key,
                        'num_passes': config_data['num_passes'],
                        'success_rate': config_data['success_rate']
                    }
                
                if (best_success_rate_config is None or 
                    config_data['success_rate'] > best_success_rate_config['success_rate']):
                    best_success_rate_config = {
                        'method': method_name,
                        'config': config_key,
                        'num_passes': config_data['num_passes'],
                        'success_rate': config_data['success_rate']
                    }
        
        return {
            'problem_number': problem_num,
            'question': question,
            'true_answer': true_answer,
            'methods': method_results,
            'best_configs': {
                'best_single': best_single_config,
                'best_mode': best_mode_config,
                'best_success_rate': best_success_rate_config
            },
            'solved_by_any_single': best_single_config is not None,
            'solved_by_mode': best_mode_config is not None
        }

    def create_comprehensive_analysis(self, all_dataset_results: List[Dict]) -> Dict:
        """Create detailed analysis with multi-pass statistics"""
        
        # Collect all configuration results
        method_stats = defaultdict(lambda: defaultdict(list))
        pass_stats = defaultdict(lambda: defaultdict(list))
        
        detailed_data = []
        
        for dataset_result in all_dataset_results:
            dataset_name = dataset_result['dataset_name']
            
            for problem_result in dataset_result['results']:
                problem_num = problem_result['problem_number']
                true_answer = problem_result['true_answer']
                
                for method_name, method_data in problem_result['methods'].items():
                    for config_key, config_data in method_data.items():
                        num_passes = config_data['num_passes']
                        success_rate = config_data['success_rate']
                        extraction_rate = config_data['extraction_rate']
                        best_single = config_data['best_single_correct']
                        mode_correct = config_data['mode_is_correct']
                        
                        # Track statistics
                        method_stats[method_name]['success_rates'].append(success_rate)
                        method_stats[method_name]['single_correct'].append(best_single)
                        method_stats[method_name]['mode_correct'].append(mode_correct)
                        method_stats[method_name]['extraction_rates'].append(extraction_rate)
                        
                        pass_stats[num_passes]['success_rates'].append(success_rate)
                        pass_stats[num_passes]['single_correct'].append(best_single)
                        pass_stats[num_passes]['mode_correct'].append(mode_correct)
                        
                        # Detailed data point
                        detailed_data.append({
                            'dataset': dataset_name,
                            'problem': problem_num,
                            'method': method_name,
                            'num_passes': num_passes,
                            'success_rate': success_rate,
                            'extraction_rate': extraction_rate,
                            'best_single_correct': best_single,
                            'mode_correct': mode_correct,
                            'true_answer': true_answer,
                            'avg_response_length': config_data['avg_response_length']
                        })
        
        # Calculate summary statistics
        def calc_summary_stats(stats_dict):
            summary = {}
            for key, values in stats_dict.items():
                summary[key] = {
                    'avg_success_rate': statistics.mean(values['success_rates']) if values['success_rates'] else 0,
                    'single_correct_rate': statistics.mean(values['single_correct']) * 100 if values['single_correct'] else 0,
                    'mode_correct_rate': statistics.mean(values['mode_correct']) * 100 if values['mode_correct'] else 0,
                    'avg_extraction_rate': statistics.mean(values['extraction_rates']) if values['extraction_rates'] else 0,
                    'count': len(values['success_rates'])
                }
            return summary
        
        return {
            'detailed_data': detailed_data,
            'method_summary': calc_summary_stats(method_stats),
            'pass_summary': calc_summary_stats(pass_stats),
            'total_evaluations': len(detailed_data)
        }

    def create_enhanced_visualizations(self, analysis: Dict, output_dir: str = "aime_multipass_results"):
        """Create comprehensive visualizations for multi-pass results"""
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        df = pd.DataFrame(analysis['detailed_data'])
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Figure 1: Method and Pass Performance (2x2)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Method comparison
        method_data = []
        for method, stats in analysis['method_summary'].items():
            method_data.append({
                'Method': method.upper(),
                'Avg Success Rate': stats['avg_success_rate'],
                'Single Correct': stats['single_correct_rate'],
                'Mode Correct': stats['mode_correct_rate']
            })
        
        method_df = pd.DataFrame(method_data)
        x = np.arange(len(method_df))
        width = 0.25
        
        ax1.bar(x - width, method_df['Avg Success Rate'], width, label='Avg Success Rate', alpha=0.8)
        ax1.bar(x, method_df['Single Correct'], width, label='Best Single Correct', alpha=0.8)
        ax1.bar(x + width, method_df['Mode Correct'], width, label='Mode Correct', alpha=0.8)
        
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title(f'Method Performance (T={self.temperature}, Tokens={self.max_tokens})', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(method_df['Method'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Pass count effects
        pass_data = []
        for passes, stats in analysis['pass_summary'].items():
            pass_data.append({
                'Passes': passes,
                'Success Rate': stats['avg_success_rate'],
                'Single Correct': stats['single_correct_rate']
            })
        
        pass_df = pd.DataFrame(pass_data).sort_values('Passes')
        ax2.plot(pass_df['Passes'], pass_df['Success Rate'], 'o-', linewidth=3, markersize=8, label='Avg Success Rate')
        ax2.plot(pass_df['Passes'], pass_df['Single Correct'], 's-', linewidth=3, markersize=8, label='Best Single Correct')
        ax2.set_xlabel('Number of Passes')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title(f'Performance vs Number of Passes (T={self.temperature}, Tokens={self.max_tokens})', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Method × Passes heatmap
        pivot_method_passes = df.groupby(['method', 'num_passes'])['success_rate'].mean().unstack()
        sns.heatmap(pivot_method_passes, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax3, 
                   cbar_kws={'label': 'Avg Success Rate (%)'})
        ax3.set_title('Method × Passes Success Rate Heatmap', fontweight='bold')
        ax3.set_ylabel('Method')
        ax3.set_xlabel('Number of Passes')
        
        # Dataset comparison
        dataset_method = df.groupby(['dataset', 'method'])['success_rate'].mean().unstack()
        dataset_method.plot(kind='bar', ax=ax4, width=0.8)
        ax4.set_title('Dataset × Method Average Success Rate', fontweight='bold')
        ax4.set_ylabel('Average Success Rate (%)')
        ax4.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/multipass_analysis.png", dpi=300, bbox_inches='tight')
        plt.close() # Close the plot to free memory
        
        # Figure 2: Detailed Analysis (1x2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pass improvement analysis
        pass_improvement = []
        baseline_rates = {}
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            baseline_rate = method_data[method_data['num_passes'] == 1]['best_single_correct'].mean() * 100
            baseline_rates[method] = baseline_rate
            
            for passes in sorted(df['num_passes'].unique()):
                if passes > 1:
                    current_rate = method_data[method_data['num_passes'] == passes]['best_single_correct'].mean() * 100
                    improvement = current_rate - baseline_rate
                    pass_improvement.append({
                        'Method': method.upper(),
                        'Passes': passes,
                        'Improvement': improvement
                    })
        
        improvement_df = pd.DataFrame(pass_improvement)
        pivot_improvement = improvement_df.pivot(index='Method', columns='Passes', values='Improvement')
        sns.heatmap(pivot_improvement, annot=True, fmt='.1f', cmap='RdBu_r', center=0, ax=ax1,
                   cbar_kws={'label': 'Improvement over 1-pass (%)'})
        ax1.set_title('Multi-pass Improvement over Single Pass', fontweight='bold')
        ax1.set_ylabel('Method')
        ax1.set_xlabel('Number of Passes')
        
        # Response length distribution
        ax2.hist(df['avg_response_length'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title(f'Response Length Distribution\n(T={self.temperature}, Max Tokens={self.max_tokens})', 
                     fontweight='bold')
        ax2.set_xlabel('Average Response Length (chars)')
        ax2.set_ylabel('Frequency')
        ax2.axvline(df['avg_response_length'].mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {df["avg_response_length"].mean():.0f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/detailed_analysis.png", dpi=300, bbox_inches='tight')
        plt.close() # Close the plot to free memory
        
        return [f"{output_dir}/multipass_analysis.png", 
                f"{output_dir}/detailed_analysis.png"]

    def create_detailed_summary_table(self, analysis: Dict) -> pd.DataFrame:
        """Create comprehensive summary table"""
        
        summary_data = []
        
        # Method summary
        for method, stats in analysis['method_summary'].items():
            summary_data.append({
                'Configuration Type': 'Method',
                'Configuration': method.upper(),
                'Avg Success Rate (%)': f"{stats['avg_success_rate']:.2f}",
                'Best Single Rate (%)': f"{stats['single_correct_rate']:.2f}",
                'Mode Correct Rate (%)': f"{stats['mode_correct_rate']:.2f}",
                'Avg Extraction Rate (%)': f"{stats['avg_extraction_rate']:.2f}",
                'Total Evaluations': stats['count']
            })
        
        # Pass summary
        for passes, stats in analysis['pass_summary'].items():
            summary_data.append({
                'Configuration Type': 'Number of Passes',
                'Configuration': f"{passes}",
                'Avg Success Rate (%)': f"{stats['avg_success_rate']:.2f}",
                'Best Single Rate (%)': f"{stats['single_correct_rate']:.2f}",
                'Mode Correct Rate (%)': f"{stats['mode_correct_rate']:.2f}",
                'Avg Extraction Rate (%)': f"{stats['avg_extraction_rate']:.2f}",
                'Total Evaluations': stats['count']
            })
        
        df = pd.DataFrame(summary_data)
        return df

    def run_full_evaluation(self, datasets: Dict[str, List[Dict]]) -> Dict:
        """Run complete evaluation on all datasets"""
        
        print(f"\n🚀 STARTING COMPREHENSIVE AIME EVALUATION")
        print(f"{'='*80}")
        print(f"Datasets: {list(datasets.keys())}")
        print(f"Model: {self.model.model_path}")
        print(f"Configuration: T={self.temperature}, Max_Tokens={self.max_tokens}")
        print(f"Passes: {self.pass_configs}")
        print(f"Methods: CoT, Code, Hybrid")
        
        start_time = datetime.now()
        all_results = []
        
        # Evaluate each dataset
        for dataset_name, problems in datasets.items():
            print(f"\n📊 Processing {dataset_name}...")
            dataset_results = self.evaluate_multipass(problems, dataset_name)
            all_results.append(dataset_results)
        
        # Create comprehensive analysis
        print(f"\n📈 Creating comprehensive analysis...")
        analysis = self.create_comprehensive_analysis(all_results)
        
        # Save all results
        print(f"\n💾 Saving comprehensive results...")
        save_info = self.save_comprehensive_results(all_results, analysis)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        print(f"\n✅ EVALUATION COMPLETE!")
        print(f"Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Total Problems: {sum(len(d['results']) for d in all_results)}")
        print(f"Total Evaluations: {analysis['total_evaluations']}")
        print(f"Results Directory: {save_info['output_directory']}")
        
        return {
            'all_dataset_results': all_results,
            'comprehensive_analysis': analysis,
            'save_info': save_info,
            'execution_time': total_time,
            'timestamp': end_time.isoformat()
        }
        
    def create_problem_level_analysis(self, all_dataset_results: List[Dict]) -> pd.DataFrame:
        """Create problem-level detailed analysis"""
        
        problem_data = []
        
        for dataset_result in all_dataset_results:
            dataset_name = dataset_result['dataset_name']
            
            for problem_result in dataset_result['results']:
                problem_num = problem_result['problem_number']
                true_answer = problem_result['true_answer']
                
                # Find best performing configurations
                best_success_rate = 0
                best_config_info = None
                total_configs = 0
                solved_configs = 0
                
                for method_name, method_data in problem_result['methods'].items():
                    for config_key, config_data in method_data.items():
                        total_configs += 1
                        if config_data['best_single_correct']:
                            solved_configs += 1
                        
                        if config_data['success_rate'] > best_success_rate:
                            best_success_rate = config_data['success_rate']
                            best_config_info = f"{method_name}_{config_key}"
                
                problem_data.append({
                    'Dataset': dataset_name,
                    'Problem': problem_num,
                    'True Answer': true_answer,
                    'Total Configs': total_configs,
                    'Solved Configs': solved_configs,
                    'Solve Rate (%)': (solved_configs / total_configs * 100) if total_configs > 0 else 0,
                    'Best Success Rate (%)': best_success_rate,
                    'Best Config': best_config_info,
                    'Solved by Any': problem_result['solved_by_any_single'],
                    'Solved by Mode': problem_result['solved_by_mode']
                })
        
        return pd.DataFrame(problem_data)

    def create_reproducibility_report(self, analysis: Dict) -> str:
        """Create a detailed reproducibility report with all hyperparameters"""
        
        report = f"""
# AIME Multi-Pass Evaluation Reproducibility Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Configuration
- **Model**: Qwen/Qwen3-8B
- **Framework**: vLLM
- **Tensor Parallel Size**: 1
- **GPU Memory Utilization**: 0.90
- **Max Model Length**: 16384 tokens
- **Data Type**: bfloat16
- **Enable Prefix Caching**: True
- **Max Num Seqs**: 256

## Sampling Parameters
- **Temperature**: {self.temperature} (FIXED)
- **Max Tokens**: {self.max_tokens} (FIXED)
- **Top P**: 0.8
- **Repetition Penalty**: 1.05
- **Use Beam Search**: False

## Evaluation Configuration
- **Pass Configurations**: {self.pass_configs}
- **Methods Evaluated**: CoT (Chain of Thought), Code, Hybrid (CoT + Code)
- **Total Configurations per Problem**: {len(self.pass_configs) * 3}
- **Datasets**: AIME 2024, AIME 2025
- **Code Execution Timeout**: 15 seconds

## Method Descriptions
### 1. Chain of Thought (CoT)
- Pure mathematical reasoning with step-by-step explanations
- Includes worked examples before the problem
- Expected output format: "**Final Answer: [number]**"

### 2. Code
- Python-based solutions using math library only
- Expected output: code that stores answer in 'answer' variable
- Includes print statement for final answer

### 3. Hybrid (CoT + Code)
- Mathematical reasoning followed by Python verification
- Combines both approaches for robustness
- Dual extraction: code execution + text parsing

## Answer Extraction Methodology
1. **Primary Patterns**: Multiple regex patterns for "Final Answer", "answer =", etc.
2. **Code Execution**: Safe execution environment with restricted imports
3. **Fallback**: Context-aware number extraction from response endings
4. **Validation**: All answers must be integers in range [0, 999]

## Statistical Metrics
- **Success Rate**: Percentage of responses with correct answers per configuration
- **Best Single Correct**: Whether any response in the pass set was correct
- **Mode Correct**: Whether the most common answer across passes was correct
- **Extraction Rate**: Percentage of responses where an answer was successfully extracted

## Key Findings Summary
- Total Evaluations: {analysis['total_evaluations']}
- Method Performance (Best Single Correct Rate):
"""
        
        # Add method performance summary
        for method, stats in analysis['method_summary'].items():
            report += f"  - {method.upper()}: {stats['single_correct_rate']:.2f}%\n"
        
        report += f"\n- Pass Count Performance (Best Single Correct Rate):\n"
        for passes, stats in sorted(analysis['pass_summary'].items()):
            report += f"  - {passes} passes: {stats['single_correct_rate']:.2f}%\n"
        
        report += f"""
## Reproducibility Checklist
- [ ] Use exact model: Qwen/Qwen3-8B
- [ ] Set temperature = {self.temperature}
- [ ] Set max_tokens = {self.max_tokens}
- [ ] Use pass configurations: {self.pass_configs}
- [ ] Use identical prompt templates (see code)
- [ ] Use same vLLM configuration parameters
- [ ] Use same random seed (if specified)
- [ ] Use same evaluation datasets (AIME 2024/2025)

## Notes
- Results may vary slightly due to non-deterministic GPU operations
- Ensure sufficient GPU memory (>= 16GB recommended for Qwen3-8B)
- Code execution uses restricted Python environment for safety
- All timestamps and response lengths are logged for analysis
"""
        
        return report

    def save_comprehensive_results(self, all_dataset_results: List[Dict], analysis: Dict, output_dir: str = "aime_multipass_results"):
        """Save all results, analysis, and visualizations"""
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save raw results
        results_file = f"{output_dir}/raw_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(all_dataset_results, f, indent=2)
        print(f"✓ Raw results saved to {results_file}")
        
        # 2. Save analysis data
        analysis_file = f"{output_dir}/analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"✓ Analysis data saved to {analysis_file}")
        
        # 3. Create and save summary tables
        summary_table = self.create_detailed_summary_table(analysis)
        summary_file = f"{output_dir}/summary_table_{timestamp}.csv"
        summary_table.to_csv(summary_file, index=False)
        print(f"✓ Summary table saved to {summary_file}")
        
        # 4. Create and save problem-level analysis
        problem_analysis = self.create_problem_level_analysis(all_dataset_results)
        problem_file = f"{output_dir}/problem_analysis_{timestamp}.csv"
        problem_analysis.to_csv(problem_file, index=False)
        print(f"✓ Problem-level analysis saved to {problem_file}")
        
        # 5. Create visualizations
        viz_files = self.create_enhanced_visualizations(analysis, output_dir)
        print(f"✓ Visualizations saved to {output_dir}")
        
        # 6. Save reproducibility report
        report = self.create_reproducibility_report(analysis)
        report_file = f"{output_dir}/reproducibility_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"✓ Reproducibility report saved to {report_file}")
        
        # 7. Save detailed DataFrame for further analysis
        df = pd.DataFrame(analysis['detailed_data'])
        df_file = f"{output_dir}/detailed_data_{timestamp}.csv"
        df.to_csv(df_file, index=False)
        print(f"✓ Detailed data saved to {df_file}")
        
        # Print final summary
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE RESULTS SAVED TO: {output_dir}")
        print(f"{'='*80}")
        print(f"Files created:")
        print(f"  - Raw results: {results_file}")
        print(f"  - Analysis data: {analysis_file}")
        print(f"  - Summary table: {summary_file}")
        print(f"  - Problem analysis: {problem_file}")
        print(f"  - Detailed data: {df_file}")
        print(f"  - Reproducibility report: {report_file}")
        print(f"  - Visualizations: {len(viz_files)} PNG files")
        
        return {
            'output_directory': output_dir,
            'files_created': {
                'raw_results': results_file,
                'analysis': analysis_file,
                'summary_table': summary_file,
                'problem_analysis': problem_file,
                'detailed_data': df_file,
                'reproducibility_report': report_file,
                'visualizations': viz_files
            }
        }

def main():
    """Main execution function"""
    print("🔥 AIME Multi-Pass Evaluation System")
    print("=" * 50)
    
    try:
        # Initialize model
        print("1️⃣ Loading Qwen3-8B model...")
        model_runner = QwenRunner("Qwen/Qwen3-8B")
        
        # Initialize evaluator
        print("2️⃣ Initializing evaluator...")
        evaluator = AIMEEvaluator(model_runner)
        
        # Prepare datasets
        print("3️⃣ Loading datasets...")
        datasets = {
            "AIME_2024": AIME_2024_PROBLEMS,
            "AIME_2025": AIME_2025_PROBLEMS
        }
        
        total_problems = sum(len(problems) for problems in datasets.values())
        print(f"   Total problems: {total_problems}")
        for name, problems in datasets.items():
            print(f"   {name}: {len(problems)} problems")
        
        # Run evaluation
        print("4️⃣ Starting evaluation...")
        results = evaluator.run_full_evaluation(datasets)
        
        print("\n🎉 SUCCESS! Check the results directory for comprehensive outputs.")
        print(f"📁 Results saved to: {results['save_info']['output_directory']}")
        
        # Quick summary
        analysis = results['comprehensive_analysis']
        print(f"\n📊 QUICK SUMMARY:")
        print(f"   Best Method (Single Correct): ", end="")
        best_method = max(analysis['method_summary'].items(), 
                         key=lambda x: x[1]['single_correct_rate'])
        print(f"{best_method[0].upper()} ({best_method[1]['single_correct_rate']:.2f}%)")
        
        print(f"   Best Pass Count (Single Correct): ", end="")
        best_passes = max(analysis['pass_summary'].items(), 
                         key=lambda x: x[1]['single_correct_rate'])
        print(f"{best_passes[0]} passes ({best_passes[1]['single_correct_rate']:.2f}%)")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()

