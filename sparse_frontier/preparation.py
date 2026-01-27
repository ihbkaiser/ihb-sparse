import os

from sparse_frontier.utils.data import write_jsonl
from sparse_frontier.utils.general import save_config
from sparse_frontier.tasks.registry import TASK_REGISTRY


def check_args(cfg):
    from transformers import AutoTokenizer, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path)
    config = AutoConfig.from_pretrained(cfg.model.path)

    if tokenizer.model_max_length < cfg.max_input_tokens + cfg.max_output_tokens:
        raise ValueError(f"Model maximum sequence length ({tokenizer.model_max_length}) is less than required length ({cfg.max_input_tokens + cfg.max_output_tokens})")
    
    max_pos_embeddings = getattr(config, "max_position_embeddings", 131072)
    if max_pos_embeddings < cfg.max_input_tokens + cfg.max_output_tokens and 'qwen' not in cfg.model.name:
        raise ValueError(f"Model maximum position embeddings ({max_pos_embeddings}) is less than required length ({cfg.max_input_tokens + cfg.max_output_tokens})")
    
    seq_length = getattr(config, "seq_length", 131072)
    if seq_length < cfg.max_input_tokens + cfg.max_output_tokens:
        raise ValueError(f"Model maximum sequence length ({seq_length}) is less than required length ({cfg.max_input_tokens + cfg.max_output_tokens})")


def get_task_generator(cfg):
    from sparse_frontier.modelling.tokenizer import Tokenizer
    task_kwargs = {
        'num_samples': cfg.samples,
        'max_input_tokens': cfg.max_input_tokens,
        'max_output_tokens': cfg.max_output_tokens,
        'tokenizer': Tokenizer(cfg.model.path, device='cpu', thinking=cfg.thinking),
        'random_seed': cfg.random_seed,
        **cfg.task.args,
    }
    return TASK_REGISTRY[cfg.task.name](**task_kwargs)


FIXED_DATASET_TASKS = {"math", "qa"}


def validate_sample_count(cfg) -> None:
    """Validate that requested samples don't exceed dataset size for fixed-dataset tasks.
    
    Args:
        cfg: Application configuration
        
    Raises:
        ValueError: If requested samples exceed available dataset size
    """
    task_name = cfg.task.name
    task_type = task_name.split("_")[0]
    
    if task_type not in FIXED_DATASET_TASKS:
        return
    
    dataset_name = cfg.task.args.get("dataset_name", task_name.split("_", 1)[1] if "_" in task_name else task_name)
    
    if task_type == "math":
        from sparse_frontier.tasks.math.math_data import get_dataset
        available_samples = len(get_dataset(dataset_name))
    else:  # qa
        from sparse_frontier.tasks.qa.qa_data import get_dataset
        available_samples = len(get_dataset(dataset_name).qa_samples)
    
    if cfg.samples > available_samples:
        raise ValueError(
            f"Requested {cfg.samples} samples but only {available_samples} "
            f"are available in the '{dataset_name}' dataset."
        )


def prepare_task(cfg):
    # Explicitly enable tokenizers parallelism during preparation
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    check_args(cfg)
    validate_sample_count(cfg)

    data_path = cfg.runtime.data_path
    generator = get_task_generator(cfg)
    
    print(f"Preparing {cfg.task.name} with {cfg.samples} samples")
    samples = generator.generate_samples()

    write_jsonl(data_path, samples)
    save_config(os.path.dirname(data_path), cfg)
    print(f"Saved {cfg.task.name} with {cfg.samples} samples to {data_path}")
    
    # Disable tokenizers parallelism after preparation
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
