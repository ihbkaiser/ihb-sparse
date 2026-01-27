import hydra
from omegaconf import DictConfig

from sparse_frontier.config_schema import AppConfig, build_runtime_paths
from sparse_frontier.utils.general import ensure_model_downloaded

def setting_up(cfg: AppConfig):
    import os
    import random
    import numpy as np

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    # In case of torch seed it's set in vLLM Model class

    os.makedirs(os.path.dirname(cfg.runtime.data_path), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.runtime.pred_path), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.runtime.results_path), exist_ok=True)


def run(cfg: AppConfig):
    ensure_model_downloaded(cfg.model.path, cfg.model.hf_repo)
    setting_up(cfg)

    from sparse_frontier.utils.checks import (
        prepration_needed,
        prediction_needed,
        evaluation_needed,
    )

    if cfg.mode in ["prep", "all"]:
        if prepration_needed(cfg):
            from sparse_frontier.preparation import prepare_task
            prepare_task(cfg)
    
    if cfg.mode in ["pred", "pred+eval", "all"]:
        if prediction_needed(cfg):
            from sparse_frontier.prediction import predict_task
            predict_task(cfg)
    
    if cfg.mode in ["eval", "pred+eval", "all"]:
        if evaluation_needed(cfg):
            from sparse_frontier.evaluation import evaluate_task
            evaluate_task(cfg)
    
    if cfg.mode not in ["prep", "pred", "eval", "pred+eval", "all"]:
        raise ValueError(f'Invalid mode: {cfg.mode}')


@hydra.main(config_path="configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    app_cfg = AppConfig.from_hydra(cfg)
    app_cfg.runtime = build_runtime_paths(app_cfg)
    run(app_cfg)


if __name__ == "__main__":
    main()
