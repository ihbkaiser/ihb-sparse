from typing import List, Dict, Any

from sparse_frontier.tasks.abstract_task import AbstractTask
from sparse_frontier.tasks.abstract_sample import AbstractSample

from sparse_frontier.tasks.math.math_data import get_dataset
from sparse_frontier.tasks.abstract_prompt import SINGLEQ_PROMPT_TEMPLATE

MATH_PROMPT_TEMPLATE = """
Solve the following math problem. Think step by step before answering to ensure your answer is correct.

Question: {prompt}

Your response must follow this structure exactly:

<explanation>
(Brief explanation of your reasoning process here)
</explanation>
<answer>
ANSWER: $ANSWER
</answer>

Where $ANSWER is the final answer to the problem.

Important:
- Keep your explanations clear, coherent, concise, and to the point.
- Do not include any additional text, explanations, or reasoning in the answer section. Follow the answer format exactly.

""".strip()

class MathSample(AbstractSample):
    @staticmethod
    def format_sample(question: str) -> str:
        return MATH_PROMPT_TEMPLATE.format(prompt=question)

    def _find_prefix_length(self, text: str, max_tokens: int) -> str:
        """Longest prefix of `text` that fits within `max_tokens` tokens."""
        pass

    def _generate_sample(self):
        dataset = self.task_params["processed_dataset"]
        sample = dataset[self.sample_id]

        question = sample["question"]
        gold_answer = sample["answer"]
        input_text = MathSample.format_sample(question)

        extra = {
            "question": question,
            "dataset": self.task_params.get("dataset_name", "math"),
            "sample_id": self.sample_id,
        }
        return input_text, gold_answer, extra
    
class MathTask(AbstractTask):
    """Task for Math evaluation."""

    def __init__(self, dataset_name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._load_and_process_dataset(dataset_name)
        self.check_params()

    def _load_and_process_dataset(self, dataset_name: str) -> None:
        """Load and slice the math dataset."""
        full_dataset = get_dataset(dataset_name)
        math_dataset = full_dataset[:self.num_samples]

        self.task_params["processed_dataset"] = math_dataset
        self.task_params["dataset_name"] = dataset_name

    def check_params(self) -> None:
        """Validate task parameters."""
        if not self.task_params.get("processed_dataset"):
            raise ValueError("Dataset not loaded")
            
    def check_sample_length(self, input_text: str, gold_answer: str) -> None:
        """Math problems are short, skip the min-length constraint."""
        return
    
    @property
    def sample_class(self):
        return MathSample
        
    @staticmethod
    def evaluate(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate predictions and compute accuracy."""
        from sparse_frontier.tasks.math.math_utils import (
            extract_tagged_response,
            answers_equal,
            normalize_answer,
        )

        if not predictions:
            return {"accuracy": 0.0}

        correct = 0
        for item in predictions:
            raw_pred = item.get("pred", "")
            gold = item.get("gold_answer", "")

            golds: List[str] = gold if isinstance(gold, list) else [gold]
            pred_str = raw_pred[0] if isinstance(raw_pred, list) else raw_pred

            norm_pred = normalize_answer(
                extract_tagged_response(pred_str) if isinstance(pred_str, str) else ""
            )
            norm_golds: List[str] = [normalize_answer(g) for g in golds]

            if answers_equal(pred=norm_pred, golds=norm_golds):
                correct += 1

        accuracy = correct / len(predictions)
        return {"accuracy": accuracy}
