"""
Configuration schema for the Sparse Frontier framework.

The framework uses Hydra for configuration management. All configurations are stored
in YAML files within `sparse_frontier/configs/`:
    - default.yaml: Main configuration file with global settings
    - attention/: Attention mechanism configurations (dense, quest, snapkv, etc.)
    - task/: Task-specific configurations (RULER, QA, Story, MATH)
    - model/: Model configurations (Qwen, Llama, Gemma)

Override any parameter from command line:
    python -m sparse_frontier.main attention=quest attention.args.token_budget=2048
    python -m sparse_frontier.main task=ruler_niah model=qwen_7b samples=100

Sample Configuration:
    The `samples` parameter controls how many samples to evaluate (must be a positive integer).
    For tasks with fixed datasets (MATH, QA), requesting more samples than available
    in the dataset will raise an error.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Mapping, Optional


@dataclass
class PathsConfig:
    results: str  # Directory for evaluation results
    predictions: str  # Directory for model predictions
    data: str  # Directory for task data
    debug: str  # Directory for debug outputs (used when debug=True)
    checkpoints: str  # Directory for model checkpoints


@dataclass
class ModelConfig:
    name: str  # Model identifier (e.g., "qwen_7b")
    path: str  # Path to model checkpoint directory
    hf_repo: str  # HuggingFace Hub repository ID (e.g., "Qwen/Qwen2.5-7B-Instruct")
    num_q_heads: int  # Number of query heads
    num_kv_heads: int  # Number of key-value heads
    num_layers: int  # Number of transformer layers


@dataclass
class AttentionConfig:
    name: str  # Attention mechanism name (must match ATTENTION_REGISTRY key)
    args: Dict[str, Any] = field(default_factory=dict)  # Attention-specific parameters


@dataclass
class TaskConfig:
    name: str  # Task name (must match TASK_REGISTRY key)
    args: Dict[str, Any] = field(default_factory=dict)  # Task-specific parameters


@dataclass
class RuntimePaths:
    """Auto-generated paths based on configuration. Not set directly by user."""
    task_params_str: str
    attn_params_str: str
    data_dir: str
    pred_dir: str
    results_dir: str
    data_path: str
    pred_path: str
    results_path: str


@dataclass
class AppConfig:
    mode: str  # Execution mode: "all", "prep", "pred", or "eval"
    overwrite: bool  # Whether to overwrite existing results
    debug: bool  # Enable debug mode (uses debug paths for safe testing)
    gpus: int  # Number of GPUs to use
    tp: int  # Tensor parallelism degree (typically set in model configs)
    max_input_tokens: int  # Maximum input sequence length
    max_output_tokens: int  # Maximum output sequence length
    kv_cache_block_size: int  # Block size for vLLM's KV cache management
    random_seed: int  # Random seed for reproducibility
    thinking: bool  # Enable thinking mode for reasoning models
    use_attention_patch: bool  # Use original vLLM attention if false
    paths: PathsConfig
    model: ModelConfig
    attention: AttentionConfig
    task: TaskConfig
    samples: int  # Number of samples to evaluate (must be positive)
    runtime: Optional[RuntimePaths] = None  # Auto-populated at runtime

    @staticmethod
    def from_hydra(cfg: Any) -> "AppConfig":
        try:
            from omegaconf import OmegaConf
            data: Mapping[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
        except Exception:
            # Already a mapping-like object
            data = cfg  # type: ignore

        app_cfg = AppConfig(
            mode=data["mode"],
            overwrite=data["overwrite"],
            debug=data["debug"],
            gpus=data["gpus"],
            tp=data["tp"],
            samples=data["samples"],
            max_input_tokens=data["max_input_tokens"],
            max_output_tokens=data["max_output_tokens"],
            kv_cache_block_size=data["kv_cache_block_size"],
            random_seed=data["random_seed"],
            thinking=data["thinking"],
            use_attention_patch=data["use_attention_patch"],
            paths=PathsConfig(**data["paths"]),
            model=ModelConfig(**data["model"]),
            attention=AttentionConfig(**data["attention"]),
            task=TaskConfig(**data["task"]),
        )
        return app_cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build_task_params_str(task_args: Mapping[str, Any], max_input_tokens: int) -> str:
    params = [f"{name}@{value}" for name, value in task_args.items()]
    params.append(f"max_input_tokens@{max_input_tokens}")
    return "+".join(params)


def _build_attn_params_str(attn_args: Mapping[str, Any]) -> str:
    if not attn_args:
        return "default"
    return "+".join(f"{name}@{value}" for name, value in attn_args.items())


def build_runtime_paths(cfg: AppConfig) -> RuntimePaths:
    task_params_str = _build_task_params_str(cfg.task.args, cfg.max_input_tokens)
    attn_params_str = _build_attn_params_str(cfg.attention.args)

    base_data_root = cfg.paths.debug if cfg.debug else cfg.paths.data
    base_pred_root = cfg.paths.debug if cfg.debug else cfg.paths.predictions
    base_results_root = cfg.paths.debug if cfg.debug else cfg.paths.results

    data_dir = os.path.join(
        base_data_root,
        cfg.task.name,
        task_params_str,
        cfg.model.name,
    )

    pred_dir = os.path.join(
        base_pred_root,
        cfg.task.name,
        task_params_str,
        cfg.model.name,
        cfg.attention.name,
        attn_params_str,
    )

    results_dir = os.path.join(
        base_results_root,
        cfg.task.name,
        task_params_str,
        cfg.model.name,
        cfg.attention.name,
        attn_params_str,
    )

    data_path = os.path.join(data_dir, "data.jsonl")
    pred_path = os.path.join(pred_dir, "pred.jsonl")
    results_path = os.path.join(results_dir, f"evaluation_results_{cfg.samples}.json")

    return RuntimePaths(
        task_params_str=task_params_str,
        attn_params_str=attn_params_str,
        data_dir=data_dir,
        pred_dir=pred_dir,
        results_dir=results_dir,
        data_path=data_path,
        pred_path=pred_path,
        results_path=results_path,
    )


__all__ = [
    "PathsConfig",
    "ModelConfig",
    "AttentionConfig",
    "TaskConfig",
    "RuntimePaths",
    "AppConfig",
    "build_runtime_paths",
]
