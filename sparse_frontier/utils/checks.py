import os

from sparse_frontier.utils.data import read_jsonl


def prepration_needed(cfg):
    data_path = cfg.runtime.data_path

    if not os.path.exists(data_path):
        return True

    if cfg.overwrite:
        os.remove(data_path)
        return True

    data = read_jsonl(data_path)
    return len(data) < cfg.samples


def prediction_needed(cfg):
    pred_path = cfg.runtime.pred_path

    if not os.path.exists(pred_path):
        return True

    if cfg.overwrite:
        os.remove(pred_path)
        return True

    data = read_jsonl(pred_path)
    return len(data) < cfg.samples


def evaluation_needed(cfg):
    results_path = cfg.runtime.results_path

    if not os.path.exists(results_path):
        return True

    if cfg.overwrite:
        os.remove(results_path)
        return True

    return False
