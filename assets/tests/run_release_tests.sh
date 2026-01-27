#!/bin/bash
# Release validation tests for sparse_frontier (optimized for 4xH100)
# Usage: ./run_release_tests.sh [test_name]
#
# Available tests:
#   multi_model_attention   - 3 models × 3 attention (RULER NIAH 32k, TP=2, DP=2)
#   all_attention_methods   - 8 attention methods (Qwen 2.5 7B, RULER NIAH 16k)
#   memory_limit_128k       - 3 models × 2 attention (V&S, Quest) at 128k
#   full_task_evaluation    - 12 tasks full pipeline (Qwen 2.5 7B, 16k, 5 samples)
#   tp_dp_consistency       - Verify TP=1/DP=1 vs TP=2/DP=2 produce consistent results
#
# Examples:
#   ./run_release_tests.sh                        # Run all tests
#   ./run_release_tests.sh -t full_task_evaluation    # Run only full task evaluation
#   ./run_release_tests.sh --list                 # List available tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Activate venv if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the tests
python assets/tests/run_release_tests.py "$@"
