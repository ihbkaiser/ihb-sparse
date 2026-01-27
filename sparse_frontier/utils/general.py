import os
import json
from dataclasses import asdict


def ensure_model_downloaded(model_path: str, hf_repo: str) -> None:
    """Download model from HuggingFace Hub if not already present.
    
    Args:
        model_path: Local path where model should be stored
        hf_repo: HuggingFace Hub repository ID (e.g., "Qwen/Qwen2.5-7B-Instruct")
    """
    import glob
    
    # Check for actual model weight files, not just config
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    bin_files = glob.glob(os.path.join(model_path, "*.bin"))
    
    if safetensors_files or bin_files:
        return
    
    print(f"Model weights not found at {model_path}. Downloading from {hf_repo}...")
    
    from huggingface_hub import snapshot_download
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    snapshot_download(
        repo_id=hf_repo,
        local_dir=model_path,
        local_dir_use_symlinks=False,
    )
    
    print(f"Model downloaded to {model_path}")


def get_latest_commit_id():
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except Exception:
        return None


def save_config(dir_path: str, cfg):
    from datetime import datetime

    config_dict = asdict(cfg)

    config_dict['commit_id'] = get_latest_commit_id()
    config_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    config_path = os.path.join(dir_path, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
