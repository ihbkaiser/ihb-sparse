import json
from pathlib import Path
from typing import List, Dict

DATA_DIR = Path(__file__).parent / "data"

DATASETS = {
    "aime24": "aime24.jsonl",
    "aime25": "aime25.jsonl",
    "math_500": "math_500.jsonl",
}


def get_dataset(dataset_name: str) -> List[Dict]:
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")

    path = DATA_DIR / DATASETS[dataset_name]
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                samples.append({
                    "question": str(row["question"]),
                    "answer": str(row["answer"]),
                })
    return samples
