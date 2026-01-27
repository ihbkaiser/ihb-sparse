#!/usr/bin/env python3
"""
Release validation tests for sparse_frontier.

Run with: python assets/tests/run_release_tests.py [--test TEST_NAME]

Tests (optimized for 4xH100):
  1. multi_model_attention: 3 models × 3 attention (RULER NIAH 32k, 4 samples, TP=2, DP=2)
  2. all_attention_methods: 8 attention methods (Qwen 2.5 7B, RULER NIAH 16k, 1 sample)
  3. memory_limit_128k: 3 models × 2 attention (V&S, Quest) at 128k
  4. full_task_evaluation: 12 tasks full pipeline (Qwen 2.5 7B, 16k, 5 samples)
  5. tp_dp_consistency: Qwen 2.5 7B RULER CWE 8k 100 samples, verify TP=1/DP=1 vs TP=2/DP=2 IoU
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float
    details: str
    error: Optional[str] = None


# Test output directory (uses debug paths to avoid polluting main experiments)
TEST_OUTPUT_DIR = Path("./experiments/release_tests")


def run_command(cmd: list[str], timeout: int = 1800, env: dict = None) -> tuple[int, str, str]:
    """Run a command and return (exit_code, stdout, stderr)."""
    run_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if env:
        run_env.update(env)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


def run_hydra_command(overrides: list[str], timeout: int = 1800, gpu_id: int = None) -> tuple[int, str, str]:
    """Run sparse_frontier.main with given Hydra overrides."""
    cmd = [
        sys.executable, "-m", "sparse_frontier.main",
        f"paths.results={TEST_OUTPUT_DIR}/results",
        f"paths.predictions={TEST_OUTPUT_DIR}/predictions",
        f"paths.data={TEST_OUTPUT_DIR}/data",
        "overwrite=true",
    ] + overrides
    
    env = None
    if gpu_id is not None:
        env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    
    return run_command(cmd, timeout, env=env)


# =============================================================================
# Test 1: Multi-model + Multi-attention on RULER NIAH
# =============================================================================
def test_multi_model_attention() -> list[TestResult]:
    """Test 3 models × 3 attention methods on RULER NIAH (32k, 4 samples, TP=2, DP=2)."""
    results = []
    
    models = ["gemma3_4b", "qwen_7b", "llama_8b"]
    attentions = ["vertical_and_slash", "snapkv", "quest"]
    
    for model in models:
        for attention in attentions:
            test_name = f"multi_model_attention/{model}/{attention}"
            print(f"\n[TEST] {test_name}")
            start = time.time()
            
            # TP=2, GPUs=4 means DP=2 (4/2=2 parallel workers)
            overrides = [
                f"model={model}",
                f"attention={attention}",
                "task=ruler_niah",
                "max_input_tokens=32768",
                "samples=4",
                "tp=2",
                "gpus=4",
                # For reasoning models, we need to handle thinking mode
                # Qwen3 has thinking=true by default in config
            ]
            
            exit_code, stdout, stderr = run_hydra_command(overrides, timeout=1200)
            duration = time.time() - start
            
            # Check for success indicators
            passed = (
                exit_code == 0
                and "Evaluation results" in stdout
                and "error" not in stderr.lower()
            )
            
            details = f"Exit code: {exit_code}"
            error = None
            if not passed:
                error = stderr[-2000:] if len(stderr) > 2000 else stderr
                if not error.strip():
                    error = stdout[-2000:] if len(stdout) > 2000 else stdout
            else:
                # Extract accuracy from output
                for line in stdout.split("\n"):
                    if "accuracy" in line.lower() or "total_samples" in line.lower():
                        details += f"\n{line.strip()}"
            
            results.append(TestResult(
                name=test_name,
                passed=passed,
                duration=duration,
                details=details,
                error=error,
            ))
            
            print(f"  {'PASSED' if passed else 'FAILED'} in {duration:.1f}s")
    
    return results


# =============================================================================
# Test 2: All Attention Methods
# =============================================================================
def test_all_attention_methods() -> list[TestResult]:
    """Test all 8 attention methods with Qwen 2.5 7b on RULER NIAH (16k, 1 sample)."""
    results = []
    
    attentions = [
        "dense",
        "vertical_and_slash",
        "block_sparse",
        "snapkv",
        "ada_snapkv",
        "quest",
        "tova",
        "flexprefill",
    ]
    
    for attention in attentions:
        test_name = f"all_attention_methods/{attention}"
        print(f"\n[TEST] {test_name}")
        start = time.time()
        
        overrides = [
            "model=qwen_7b",
            f"attention={attention}",
            "task=ruler_niah",
            "max_input_tokens=16384",
            "samples=1",
            "tp=1",
            "gpus=1",
            "mode=all",
        ]
        
        exit_code, stdout, stderr = run_hydra_command(overrides, timeout=600)
        duration = time.time() - start
        
        passed = (
            exit_code == 0
            and "Evaluation results" in stdout
            and "error" not in stderr.lower()
        )
        
        details = f"Exit code: {exit_code}"
        error = None
        if not passed:
            error = stderr[-2000:] if len(stderr) > 2000 else stderr
            if not error.strip():
                error = stdout[-2000:] if len(stdout) > 2000 else stdout
        else:
            for line in stdout.split("\n"):
                if "accuracy" in line.lower() or "total_samples" in line.lower():
                    details += f"\n{line.strip()}"
        
        results.append(TestResult(
            name=test_name,
            passed=passed,
            duration=duration,
            details=details,
            error=error,
        ))
        
        print(f"  {'PASSED' if passed else 'FAILED'} in {duration:.1f}s")
    
    return results


# =============================================================================
# Test 3: Memory Limit 128k
# =============================================================================
def test_memory_limit_128k() -> list[TestResult]:
    """Test 128k context with models using V&S and Quest. Verifies no OOM errors."""
    results = []
    
    models = ["gemma3_4b", "qwen_7b", "llama_8b"]
    attentions = ["vertical_and_slash", "quest"]
    
    for model in models:
        for attention in attentions:
            test_name = f"memory_limit_128k/{model}/{attention}"
            print(f"\n[TEST] {test_name}")
            start = time.time()
            
            overrides = [
                f"model={model}",
                f"attention={attention}",
                "task=ruler_niah",
                "max_input_tokens=128000",
                "samples=1",
                "tp=1",
                "gpus=1",
                "mode=all",
            ]
            
            exit_code, stdout, stderr = run_hydra_command(overrides, timeout=1800)
            duration = time.time() - start
            
            oom_error = (
                "out of memory" in stderr.lower()
                or "cuda out of memory" in stderr.lower()
                or "oom" in stderr.lower()
            )
            
            passed = exit_code == 0 and not oom_error and "Evaluation results" in stdout
            
            details = f"Exit code: {exit_code}"
            error = None
            if not passed:
                if oom_error:
                    details += " - OOM detected"
                error = stderr[-2000:] if len(stderr) > 2000 else stderr
                if not error.strip():
                    error = stdout[-2000:] if len(stdout) > 2000 else stdout
            else:
                for line in stdout.split("\n"):
                    if "accuracy" in line.lower():
                        details += f"\n{line.strip()}"
            
            results.append(TestResult(
                name=test_name,
                passed=passed,
                duration=duration,
                details=details,
                error=error,
            ))
            
            print(f"  {'PASSED' if passed else 'FAILED'} in {duration:.1f}s")
    
    return results


# =============================================================================
# Test 4: Full Task Evaluation
# =============================================================================
def test_full_task_evaluation() -> list[TestResult]:
    """Run full pipeline (prep + pred + eval) for all 12 tasks with Qwen 2.5 7b (16k, 5 samples)."""
    results = []
    
    tasks = [
        # RULER tasks
        "ruler_niah",
        "ruler_cwe",
        "ruler_vt",
        # Story tasks
        "story_multihop",
        "story_filtering",
        "story_retrieval",
        # QA tasks
        "qa_quality",
        "qa_squad",
        "qa_toefl",
        # MATH tasks
        "math_500",
        "math_aime24",
        "math_aime25",
    ]
    
    for task_name in tasks:
        test_name = f"full_task_evaluation/{task_name}"
        print(f"\n[TEST] {test_name}")
        start = time.time()
        
        overrides = [
            "model=qwen_7b",
            "attention=dense",
            f"task={task_name}",
            "samples=5",
            "max_input_tokens=16384",
            "tp=1",
            "gpus=1",
            "mode=all",
        ]
        
        exit_code, stdout, stderr = run_hydra_command(overrides, timeout=900)
        duration = time.time() - start
        
        passed = (
            exit_code == 0
            and "Evaluation results" in stdout
            and "error" not in stderr.lower()
        )
        
        details = f"Exit code: {exit_code}"
        error = None
        if not passed:
            error = stderr[-2000:] if len(stderr) > 2000 else stderr
            if not error.strip():
                error = stdout[-2000:] if len(stdout) > 2000 else stdout
        else:
            for line in stdout.split("\n"):
                # Capture all relevant metrics: accuracy, accuracy_math, iou, f1, em
                if any(k in line.lower() for k in ["accuracy", "iou", "f1", "em", "total_samples"]):
                    details += f"\n{line.strip()}"
        
        results.append(TestResult(
            name=test_name,
            passed=passed,
            duration=duration,
            details=details,
            error=error,
        ))
        
        print(f"  {'PASSED' if passed else 'FAILED'} in {duration:.1f}s")
    
    return results


# =============================================================================
# Test 5: TP Consistency Check
# =============================================================================
def test_tp_dp_consistency() -> list[TestResult]:
    """Verify that TP=1/DP=4 and TP=2/DP=2 produce consistent IoU.
    
    Runs Qwen 2.5 7b on RULER CWE with 100 samples at 8k in two configurations:
    - tp=1, gpus=4 (TP=1, DP=4)
    - tp=2, gpus=4 (TP=2, DP=2)
    
    Checks that IoU difference is within 2%.
    """
    test_name = "tp_dp_consistency"
    print(f"\n[TEST] {test_name}")
    results = []
    
    configs = [
        ("tp1_dp4", ["tp=1", "gpus=4"]),
        ("tp2_dp2", ["tp=2", "gpus=4"]),
    ]
    
    ious = {}
    
    for config_name, tp_overrides in configs:
        sub_test_name = f"{test_name}/{config_name}"
        print(f"  Running {config_name}...")
        start = time.time()
        
        overrides = [
            "model=qwen_7b",
            "attention=dense",
            "task=ruler_cwe",
            "samples=100",
            "max_input_tokens=8192",
            "mode=all",
        ] + tp_overrides
        
        exit_code, stdout, stderr = run_hydra_command(overrides, timeout=1800)
        duration = time.time() - start
        
        passed = (
            exit_code == 0
            and "Evaluation results" in stdout
            and "error" not in stderr.lower()
        )
        
        details = f"Exit code: {exit_code}, Config: {config_name}"
        error = None
        iou = None
        
        if not passed:
            error = stderr[-2000:] if len(stderr) > 2000 else stderr
            if not error.strip():
                error = stdout[-2000:] if len(stdout) > 2000 else stdout
        else:
            # Extract IoU from output (ruler_cwe outputs "iou" metric)
            for line in stdout.split("\n"):
                if '"iou"' in line.lower():
                    details += f"\n  {line.strip()}"
                    # Parse iou value from JSON output like: "iou": 0.85,
                    match = re.search(r'"iou"[\'"]?\s*:\s*([0-9.]+)', line.lower())
                    if match:
                        iou = float(match.group(1))
                        ious[config_name] = iou
        
        results.append(TestResult(
            name=sub_test_name,
            passed=passed,
            duration=duration,
            details=details,
            error=error,
        ))
        
        print(f"    {config_name}: {'PASSED' if passed else 'FAILED'} in {duration:.1f}s" + 
              (f" (iou: {iou:.4f})" if iou else ""))
    
    # Check consistency between configurations
    if len(ious) == 2:
        iou1 = ious.get("tp1_dp4", 0)
        iou2 = ious.get("tp2_dp2", 0)
        diff = abs(iou1 - iou2)
        
        consistency_passed = diff <= 0.02  # 2% tolerance
        consistency_details = f"tp1_dp4={iou1:.4f}, tp2_dp2={iou2:.4f}, diff={diff:.4f}"
        
        results.append(TestResult(
            name=f"{test_name}/consistency_check",
            passed=consistency_passed,
            duration=0,
            details=consistency_details,
            error=None if consistency_passed else f"IoU difference {diff:.4f} exceeds 2% tolerance",
        ))
        
        print(f"  Consistency: {'PASSED' if consistency_passed else 'FAILED'} ({consistency_details})")
    
    return results


# =============================================================================
# Test Registry
# =============================================================================
TESTS = {
    "multi_model_attention": test_multi_model_attention,
    "all_attention_methods": test_all_attention_methods,
    "memory_limit_128k": test_memory_limit_128k,
    "full_task_evaluation": test_full_task_evaluation,
    "tp_dp_consistency": test_tp_dp_consistency,
}


def print_summary(all_results: list[TestResult]):
    """Print a summary of all test results."""
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in all_results if r.passed)
    failed = sum(1 for r in all_results if not r.passed)
    total_time = sum(r.duration for r in all_results)
    
    print(f"\nTotal: {len(all_results)} tests | Passed: {passed} | Failed: {failed}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    if failed > 0:
        print("\n" + "-" * 40)
        print("FAILED TESTS:")
        print("-" * 40)
        for r in all_results:
            if not r.passed:
                print(f"\n{r.name}")
                print(f"  Duration: {r.duration:.1f}s")
                print(f"  Details: {r.details}")
                if r.error:
                    print(f"  Error: {r.error[:2000]}...")
    
    print("\n" + "-" * 40)
    print("ALL RESULTS:")
    print("-" * 40)
    for r in all_results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"  {status} {r.name} ({r.duration:.1f}s)")
    
    # Save results to JSON
    results_file = TEST_OUTPUT_DIR / "test_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(
            {
                "summary": {"passed": passed, "failed": failed, "total_time": total_time},
                "results": [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "duration": r.duration,
                        "details": r.details,
                        "error": r.error,
                    }
                    for r in all_results
                ],
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {results_file}")
    
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Run release validation tests")
    parser.add_argument(
        "--test", "-t",
        type=str,
        default=None,
        help=f"Run specific test: {list(TESTS.keys())}",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available tests",
    )
    args = parser.parse_args()
    
    if args.list:
        print("Available tests:")
        for name in TESTS:
            print(f"  - {name}")
        return
    
    # Create test output directory
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("SPARSE FRONTIER RELEASE TESTS")
    print("=" * 80)
    print(f"Output directory: {TEST_OUTPUT_DIR}")
    
    all_results = []
    
    if args.test:
        if args.test not in TESTS:
            print(f"Unknown test: {args.test}")
            print(f"Available: {list(TESTS.keys())}")
            sys.exit(1)
        tests_to_run = {args.test: TESTS[args.test]}
    else:
        tests_to_run = TESTS
    
    for test_name, test_func in tests_to_run.items():
        print(f"\n{'='*40}")
        print(f"Running test suite: {test_name}")
        print("=" * 40)
        results = test_func()
        all_results.extend(results)
    
    success = print_summary(all_results)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
