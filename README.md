
<p align="center">
  <img src="./assets/photo.png" width="100%" alt="logo">
</p>

## Latest News🔥

- [2025/12] 💻 Added **MATH** tasks to the evaluation suite: `AIME24/25`, `MATH500`
- [2025/12] 🧠 Added support for **reasoning models** like **Qwen3** and **hybrid models** like **Gemma 3**
- [2025/12] ⚡ Added support for **vLLM v1** (re-architected vLLM engine)
- [2025/12] 🧠️ Added support for **TOVA (Token Omission Via Attention)**, training-free KV-cache compression via attention-based token omission


## About

**The evaluation framework for training-free sparse attention in LLMs**

This repository serves two main purposes:
1. **Reproducing results** from our paper "[The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs](https://arxiv.org/abs/2504.17768)".
2. **Providing a starting point** for your own training-free sparse attention research and development.


### Why This Framework?

**The Problem**: vLLM is a highly optimized framework supporting hundreds of models, but its extensive codebase makes integrating custom sparse attention patterns extremely challenging. Researchers face a difficult choice: build from scratch with limited model support, use Hugging Face which often lacks efficient inference support (TP), or navigate vLLM's complex internals.

**Our Solution**: We provide a clean abstraction that lets you focus on your sparse attention logic while automatically inheriting all of vLLM's optimizations and model compatibility. Here's what makes our framework unique:

- **🎯 Elegant vLLM Integration**: Seamless sparse attention integration through our `AttentionHandler` that intercepts vLLM's execution flow. Write your sparse attention in 50 lines, not 5000—evaluate on 100 models, not 1. By implementing sparse attention in our framework, you automatically gain compatibility with all models supported by vLLM, from small 7B models to large 405B+ models across different architectures (Qwen, Gemma, etc.).
- **⚡ State-of-the-art Baselines**: 7 representative SOTA patterns spanning key design dimensions for both inference phases—sparse prefilling (Vertical-Slash, Block-Sparse, FlexPrefill), sparse decoding (Quest), and KV cache compression (SnapKV, Ada-SnapKV, TOVA)—with optimized Triton implementations.
- **🔬 Comprehensive Evaluation**: A diverse suite of tasks covering retrieval, multi-hop reasoning, and information aggregation, math and code, with rigorous sequence length control and standardized preprocessing.
- **🧪 Research-Grade Extensibility**: Clean modular architecture with abstract base classes designed for rapid prototyping of novel sparse attention patterns and tasks.

### Getting Started with Sparse Attention

If you're new to sparse attention and want to understand how these patterns work, we recommend starting with our companion repository: [nano-sparse-attention](https://github.com/PiotrNawrot/nano-sparse-attention). It provides clean, educational PyTorch implementations with interactive Jupyter notebooks for experimenting and learning before diving into the optimized implementations here. Originally created for the [NeurIPS 2024 Dynamic Sparsity Workshop](https://dynamic-sparsity.github.io/), it serves as an excellent starting point for understanding sparse attention fundamentals.

## Setup

````bash
python -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir --upgrade pip setuptools wheel psutil ninja
pip install --no-cache-dir torch==2.8.0
pip install --no-cache-dir -e .
MAX_JOBS=8 python compile.py build_ext --inplace --build-lib ./sparse_frontier/modelling/attention/minference
````

For reference, the complete list of dependencies used in our experiments is available in `./assets/pipfreeze.txt`. We tested the codebase on both A100 and H100 GPUs.

1. **Configure Paths:**
   Modify the default configuration file to specify where data, results, and checkpoints should be stored on your system.

   * Edit the `paths` section in `configs/default.yaml`.

2. **Model Checkpoints:**
   Model checkpoints are automatically downloaded from HuggingFace Hub on first run. The models are saved to the `paths.checkpoints` directory specified in `configs/default.yaml`.

   * For gated models (e.g., Llama, Gemma), ensure you have accepted the model license on HuggingFace and are logged in via `huggingface-cli login`.

## Where should I look at if I want to:

### Reproduce your experiments

Experiments are launched using the main script `sparse_frontier.main`. The framework uses [Hydra](https://hydra.cc/) for configuration management. All configurations are stored in YAML files within the `sparse_frontier/configs/` directory, organized into three main categories:

* **`attention/`**: Configurations for different sparse attention mechanisms (dense, quest, snapkv, etc.)
* **`task/`**: Configurations for evaluation tasks (RULER, QA, Story, MATH)
* **`model/`**: Configurations for different model architectures

The execution pipeline typically involves three stages, controlled by the `mode` parameter (defaulting to `all`):

1. **Preparation (`preparation.py`):** Generates and saves task-specific data based on the selected `task` configuration. Tasks are defined in `sparse_frontier/tasks/` (inheriting from [AbstractTask](./sparse_frontier/tasks/abstract_task.py) and [AbstractSample](./sparse_frontier/tasks/abstract_sample.py)) and registered in [TASK_REGISTRY](sparse_frontier/tasks/registry.py).
2. **Prediction (`prediction.py`):** Runs the specified `model` with the chosen `attention` mechanism on the prepared data, saving the model outputs. Attention mechanisms are implemented in `sparse_frontier/modelling/attention/` and registered in [ATTENTION_REGISTRY](sparse_frontier/modelling/attention/registry.py).
3. **Evaluation (`evaluation.py`):** Compares the predictions against the gold answers using the task's specific evaluation logic and saves the final metrics.

**Quick Start Examples:**

```bash
# Basic experiment with command line overrides
python -m sparse_frontier.main task=ruler_niah model=qwen_7b attention=dense samples=1

# Override attention parameters
python -m sparse_frontier.main attention=quest attention.args.token_budget=2048
```

For detailed configuration documentation see [`sparse_frontier/config_schema.py`](sparse_frontier/config_schema.py).

**Note**: The current framework implementation supports only batch size = 1. This limitation stems from our initial experiments with methods that had kernels supporting only BS=1. Since then, we have followed a simple heuristic: for a given (Model Size, Method, Sequence Length) combination, we find the minimum tensor parallelism (TP) that provides sufficient total GPU memory to handle the evaluation, then use our [intra-node scheduler](./sparse_frontier/prediction.py) to distribute BS=1 evaluations across the node's GPUs. For the majority of our evaluations, we achieved >95% GPU utilization. Nevertheless, higher throughput and GPU utilization could likely be achieved with BS>1.

### Develop Your Own Training-Free Sparse Attention

Instead of wrestling with vLLM's complex internals, we provide a clean abstraction layer that lets you focus on your sparse attention logic.

#### How It Works

Our integration works by intercepting vLLM's attention execution at the FlashAttention level. When you register your sparse attention pattern, our framework:

1. **Patches vLLM's FlashAttention forward method** - The `vllm_patched_forward` function in [vllm_model.py](./sparse_frontier/modelling/models/vllm_model.py) replaces vLLM's default attention computation.
2. **Routes attention calls through our handler** - The `AttentionHandler` from [handler.py](./sparse_frontier/modelling/attention/handler.py) manages layer state, prefill vs decode phases, and KV cache updates.
3. **Executes your sparse attention** - Your implementation receives the same tensors vLLM would use, but with your custom attention logic.

The `swap_vllm_attention` function is registered as a vLLM plugin in [setup.py](./setup.py), ensuring all tensor parallel workers automatically load our custom implementation. This provides seamless Tensor Parallelism support without any additional configuration.

The integration is automatic - when you register your attention pattern, it becomes available to all vLLM-compatible models without any additional setup.

#### Implementing a New Sparse Attention Pattern

```python
from sparse_frontier.modelling.attention.abstract_attention import AbstractAttention

class MySparseAttention(AbstractAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Your initialization logic
    
    def __call__(self, queries, keys, values, layer_idx):
        # Your prefill attention logic (uses dense prefill if not implemented)
        return attention_output
    
    def decode(self, query, keys, values, k_cache, v_cache, tokens_per_head, output, layer_idx):
        # Your decoding logic (uses dense decoding if not implemented)
        pass
    
    def kv_compress(self, queries, keys, values):
        # Your KV compression logic (leaves the KV Cache intact if not implemented)
        return compressed_keys, compressed_values, seq_lens
```

That's it! No need to browse the huge vLLM codebase or worry about inference state handling, etc.

Examples can be found in: [kv_compression.py](./sparse_frontier/modelling/attention/kv_compression.py) for SnapKV and AdaSnapKV; [efficient_prefilling.py](./sparse_frontier/modelling/attention/efficient_prefilling.py) for Vertical-Slash, Block-Sparse, and FlexPrefill, [efficient_decoding.py](./sparse_frontier/modelling/attention/efficient_decoding.py) for Quest and TOVA.

#### Registration

Once you've implemented your sparse attention pattern, registration is a simple two-step process:

**1. Register in the Attention Registry**

Add your attention class to the `ATTENTION_REGISTRY` dictionary in [`sparse_frontier/modelling/attention/registry.py`](./sparse_frontier/modelling/attention/registry.py):

```python
from .your_module import MySparseAttention  # Import your implementation

ATTENTION_REGISTRY = {
    ...
    'my_sparse_attention': MySparseAttention,  # Add your pattern here
}
```

**2. Create Configuration File**

Create a YAML configuration file at `configs/attention/my_sparse_attention.yaml` that defines your attention mechanism and its parameters:

```yaml
# @package _global_

attention:
  name: my_sparse_attention
  args:
    sparsity_ratio: 0.1
```

The configuration structure follows the pattern used by existing attention mechanisms. The `name` field must match your registry key, and `args` contains all the parameters that will be passed to your attention class constructor.

**3. Run Evaluation**

Your sparse attention pattern is now ready to use! Test it with any model and task:

```bash
# Basic evaluation with your new attention pattern
python -m sparse_frontier.main task=ruler_niah model=qwen_7b attention=my_sparse_attention

# Override specific attention parameters
python -m sparse_frontier.main attention=my_sparse_attention attention.args.sparsity_ratio=0.05
```

**Result**: Your sparse attention works with any vLLM-compatible model and benefits from all vLLM optimizations.

### Extend the Test Data

Experimental data generation is handled by task-specific modules located in `sparse_frontier/tasks/`. Each task implements `AbstractTask` and `AbstractSample` subclasses (defined in `sparse_frontier/tasks/abstract_*.py`) to define input / output creation, and task-specific evaluation. Tasks are registered in `sparse_frontier/tasks/registry.py` and selected via configuration (e.g., `task=your_task_name`). The `preparation.py` script orchestrates the generation process based on the configuration, saving the formatted samples. See existing tasks like `QATask` (`qa_task.py`) or the Ruler tasks (`niah.py`, `cwe.py`, `vt.py`) for implementation examples.

## References

### Sparse Attention Patterns

In this repository, we evaluate 6 sparse attention patterns:

| Pattern                                                                     | Source                                                                    |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Vertical-Slash / Block-Sparse**                                           | [Microsoft](https://github.com/microsoft/MInference)                      |
| **FlexPrefill**                                                             | [ByteDance-Seed](https://github.com/ByteDance-Seed/FlexPrefill)           |
| **SnapKV**                                                                  | [FasterDecoding](https://github.com/FasterDecoding/SnapKV)                |
| **Ada-SnapKV**                                                              | [FFY0](https://github.com/FFY0/AdaKV)                                     |
| **Quest**                                                                   | [MIT-HAN-Lab](https://github.com/mit-han-lab/Quest)                       |
| **TOVA** (Token Omission Via Attention; training-free KV-cache compression) | [Oren et al., EMNLP 2024](https://aclanthology.org/2024.emnlp-main.1043/) |

We either re-implement these patterns based on the original code or borrow implementations including kernels (for Vertical-Slash and Block-Sparse) from MInference.

### Evaluation Tasks

Our evaluation framework includes the following tasks:

1. **RULER Tasks**: Re-implementation of NIAH, VT, and CWE tasks from [NVIDIA/RULER](https://github.com/NVIDIA/RULER)
2. **QA Tasks**:
   * Toefl and Quality datasets from [LC-VS-RAG](https://github.com/lixinze777/LC_VS_RAG)
   * SQuAD dataset from [NVIDIA/RULER](https://github.com/NVIDIA/RULER)
3. **Novel Story Tasks**: Narrative tasks developed specifically for this project.
4. **MATH Tasks**:
   * AIME24: https://huggingface.co/datasets/HuggingFaceH4/aime_2024
   * AIME25: https://huggingface.co/datasets/opencompass/AIME2025
   * MATH 500: https://huggingface.co/datasets/HuggingFaceH4/MATH-500

## Cite

If you found the repository useful consider citing the paper about this work.

```
@article{nawrot2025sparsefrontier,
      title={The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs}, 
      author={Piotr Nawrot and Robert Li and Renjie Huang and Sebastian Ruder and Kelly Marchisio and Edoardo M. Ponti},
      year={2025},
      journal={arXiv:2504.17768}
      url={https://arxiv.org/abs/2504.17768}, 
}
```

## Issues:

If you have any questions, feel free to raise a Github issue or contact me directly at: [piotr.nawrot@ed.ac.uk](mailto:piotr.nawrot@ed.ac.uk)
