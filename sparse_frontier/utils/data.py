import json
import logging
import os
from pathlib import Path
from typing import List, Union, Any


def read_jsonl(manifest: Union[Path, str]) -> List[dict]:
    """Read and parse a JSONL file into a list of dictionaries.

    Args:
        manifest: Path to JSONL file to read

    Returns:
        List of dictionaries parsed from JSONL
    
    Raises:
        json.JSONDecodeError: If JSONL parsing fails
        Exception: If file cannot be read
    """
    try:
        with open(manifest, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse line in manifest file {manifest}: {e}")
        raise
    except Exception as e:
        raise Exception(f"Could not read manifest file {manifest}") from e


def write_jsonl(output_path: Union[Path, str], data: List[dict]) -> None:
    """Write a list of dictionaries to a JSONL file.

    Args:
        output_path: Path to output JSONL file
        data: List of dictionaries to serialize
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def load_data_without_predictions(cfg: Any) -> List[dict]:
    """Load task data excluding samples that already have predictions using provided cfg.

    Args:
        cfg: Dict-like or OmegaConf config object.

    Returns:
        List of data samples that haven't been predicted yet, limited by index <= cfg['samples'].
    """
    data_path = cfg.runtime.data_path
    pred_path = cfg.runtime.pred_path
    samples = cfg.samples

    if os.path.exists(pred_path):
        pred_index = {sample['index'] for sample in read_jsonl(pred_path)}
        data = [
            sample for sample in read_jsonl(data_path)
            if sample['index'] not in pred_index and sample['index'] < samples
        ]
    else:
        data = [sample for sample in read_jsonl(data_path) if sample['index'] < samples]

    return data
