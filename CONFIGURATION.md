# Configuration Guide

This document provides comprehensive information about the configuration system used in the Sparse Frontier framework. The framework uses [Hydra](https://hydra.cc/) for configuration management, providing a flexible and hierarchical system for organizing experiment parameters.

## Configuration Structure

All configurations are stored in YAML files within the `sparse_frontier/configs/` directory:

```
sparse_frontier/configs/
├── default.yaml              # Main configuration file with global settings
├── attention/                # Attention mechanism configurations
│   ├── dense.yaml           # Standard dense attention
│   ├── quest.yaml           # Quest sparse attention
│   ├── snapkv.yaml          # SnapKV attention
│   ├── ada_snapkv.yaml      # Ada-SnapKV attention
│   ├── flexprefill.yaml     # FlexPrefill attention
│   ├── block_sparse.yaml    # Block-sparse attention
│   ├── shadowkv.yaml         # ShadowKV accuracy baseline
│   └── vertical_and_slash.yaml # Vertical-Slash attention
├── task/                    # Task-specific configurations
│   ├── ruler_niah.yaml      # RULER NIAH task
│   ├── ruler_vt.yaml        # RULER VT task
│   ├── ruler_cwe.yaml       # RULER CWE task
│   ├── qa_squad.yaml        # SQuAD QA task
│   ├── qa_toefl.yaml        # TOEFL QA task
│   ├── qa_quality.yaml      # Quality QA task
│   ├── story_retrieval.yaml # Story retrieval task
│   ├── story_multihop.yaml  # Story multi-hop task
│   └── story_filtering.yaml # Story filtering task
└── model/                   # Model configurations
    ├── qwen_7b.yaml         # Qwen2.5-7B model
    ├── qwen_14b.yaml        # Qwen2.5-14B model
    ├── qwen_32b.yaml        # Qwen2.5-32B model
    └── qwen_72b.yaml        # Qwen2.5-72B model
```

## Main Configuration (`default.yaml`)

The `default.yaml` file contains global settings that apply to all experiments:

### Experiment Control
- **`mode`**: Execution mode (`"all"`, `"prep"`, `"pred"`, `"eval"`)
- **`overwrite`**: Whether to overwrite existing results
- **`debug`**: Enable debug mode (changes output directories for safe testing)

### Hardware Settings
- **`gpus`**: Number of GPUs to use
- **`tp`**: Tensor parallelism degree (set in model configs)

### Evaluation Parameters
- **`samples`**: Number of samples to evaluate
- **`max_input_tokens`**: Maximum input sequence length
- **`max_output_tokens`**: Maximum output sequence length
- **`kv_cache_block_size`**: Block size for vLLM's KV cache management (don't really impact efficiency)
- **`random_seed`**: Random seed for reproducibility

### Paths
- **`results`**: Directory for evaluation results
- **`predictions`**: Directory for model predictions
- **`data`**: Directory for task data
- **`debug`**: Directory for debug outputs
- **`checkpoints`**: Directory for model checkpoints

### Hydra Settings
- **`hydra.run.dir`**: Output directory for Hydra logs
- **`hydra.job.env_set`**: Environment variables to set

## Modifying Configurations

You can override any configuration parameter directly from the command line:

```bash
# Override attention mechanism and its parameters
python -m sparse_frontier.main attention=quest attention.args.token_budget=2048

# Override task and model
python -m sparse_frontier.main task=ruler_vt model=qwen_14b

# Override global settings
python -m sparse_frontier.main samples=50 gpus=2 max_input_tokens=16384

# Override multiple parameters at once
python -m sparse_frontier.main \
  attention=quest \
  attention.args.token_budget=2048 \
  attention.args.page_size=32 \
  task=ruler_niah \
  samples=200
```
