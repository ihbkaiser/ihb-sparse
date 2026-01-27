from typing import List, Dict, Any, Tuple

from sparse_frontier.tasks.abstract_task import AbstractTask
from sparse_frontier.tasks.abstract_sample import AbstractSample
from sparse_frontier.tasks.qa.qa_utils import (
    normalize_mcq_answer, normalize_answer, f1_score, extract_tagged_response
)
from sparse_frontier.tasks.qa.qa_data import get_dataset
from sparse_frontier.tasks.abstract_prompt import SINGLEQ_PROMPT_TEMPLATE

# Task introduction
TASK_INTRO = "I will provide you with multiple documents and ask you a question about one specific document."

OPEN_ENDED_FORMAT = "Your answer here..."
MCQ_FORMAT = "Only the letter(s) of the correct option(s) (e.g., 'A' or 'AB')."

# Extra instructions for different question types
OPEN_ENDED_INSTRUCTIONS = """
- Do not use complete sentences in the answer.
- For dates: Include ONLY the COMPLETE date if specifically asked.
- For locations: Use the shortest unambiguous form (e.g., 'New York' not 'New York City').
- For comparisons: State ONLY the answer that matches the criteria
""".strip()

MCQ_INSTRUCTIONS = """
- Multiple correct options are possible (e.g., 'AB' is valid).
""".strip()


class QASample(AbstractSample):
    SAFETY_TOKENS_FOR_PACKING = 25
    """Handles generation of individual QA samples."""
    
    @staticmethod
    def format_sample(context: str, question: str, is_mcq: bool, target_doc: int) -> str:
        """Format a QA sample with appropriate prompt template."""
        question_text = f"Question about document {target_doc}:\n{question}"
        
        return SINGLEQ_PROMPT_TEMPLATE.format(
            task_intro=TASK_INTRO,
            context=context,
            question=question_text,
            answer_format=MCQ_FORMAT if is_mcq else OPEN_ENDED_FORMAT,
            extra_instructions=MCQ_INSTRUCTIONS if is_mcq else OPEN_ENDED_INSTRUCTIONS
        )

    def _find_prefix_length(self, text: str, max_tokens: int) -> str:
        """Find the longest prefix of text that fits within max_tokens.
        
        Uses binary search to efficiently find the prefix from text.
        
        Args:
            text: Text to find prefix from
            max_tokens: Maximum number of tokens allowed
            
        Returns:
            The prefix of text that fits within max_tokens
        """
        words = text.split()
        left, right = 0, len(words)
        
        while left < right:
            mid = (left + right + 1) // 2
            prefix = " ".join(words[:mid])
            tokens = len(self.tokenizer.text_to_tokens(prefix))
            
            if tokens <= max_tokens:
                left = mid
            else:
                right = mid - 1
                
        return " ".join(words[:left])
    
    def _generate_sample(self) -> Tuple[str, str, Dict[str, Any]]:
        """Generate a single QA sample."""
        dataset = self.task_params["processed_dataset"]
        context_pool = self.task_params["context_pool"]
        sample = dataset[self.sample_id]
        
        # Calculate tokens used by question and original context
        original_context = sample["context"]
        question = sample["question"]
        current_tokens = sample["total_token_length"]
        
        # Select additional contexts that fit within token budget
        selected_contexts = []

        # Get shuffled contexts excluding the original one
        available_contexts = [ctx for ctx in context_pool if ctx["text"] != original_context]
        self.random_obj.shuffle(available_contexts)

        # Keep track of remaining contexts
        remaining_contexts = available_contexts.copy()
        
        for ctx in available_contexts:
            if current_tokens + ctx["token_length"] < self.max_tokens:
                selected_contexts.append(ctx["text"])
                current_tokens += ctx["token_length"]
                remaining_contexts.remove(ctx)
        
        # Calculate remaining tokens and add a prefix if possible
        remaining_tokens = self.max_tokens - current_tokens - self.SAFETY_TOKENS_FOR_PACKING
        if remaining_tokens >= 0 and remaining_contexts:
            for ctx in remaining_contexts:
                if ctx["token_length"] > remaining_tokens:
                    prefix = self._find_prefix_length(ctx["text"], remaining_tokens)
                    if prefix:
                        selected_contexts.append(prefix)
                    break
                
        # Combine all documents and shuffle using sample-specific random object
        all_documents = selected_contexts + [original_context]
        self.random_obj.shuffle(all_documents)
        
        # Find index of original context after shuffling
        target_doc = all_documents.index(original_context) + 1
        
        # Format context with document markers
        formatted_context = "\n\n".join(
            f"Document {i+1}:\n{doc}" 
            for i, doc in enumerate(all_documents)
        )
        
        # Format final input with prompt
        input_text = self.format_sample(
            context=formatted_context,
            question=question,
            is_mcq=sample["is_mcq"],
            target_doc=target_doc
        )

        # Get gold answer and metadata
        gold_answer = sample["answer"]
        extra_data = {
            "question": question,
            "dataset": sample["dataset"],
            "is_mcq": sample["is_mcq"],
            "num_documents": len(all_documents),
            "target_doc": target_doc
        }

        return input_text, gold_answer, extra_data


class QATask(AbstractTask):
    """Task for question answering evaluation."""
    MAX_SAMPLE_TOKENS = 8172  # Maximum tokens for a single-document sample (8192 - 20 safe tokens)
    
    def __init__(self, dataset_name, **kwargs) -> None:
        """Initialize QA task.
        
        Args:
            dataset_name: Name of the dataset to load
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self._load_and_process_dataset(dataset_name)
        self.check_params()

    def _load_and_process_dataset(self, dataset_name: str) -> None:
        """Load and preprocess the QA dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Note:
            Samples are filtered based on min(MAX_SAMPLE_TOKENS, max_input_tokens)
            to ensure consistency across runs while respecting the token budget.
        """
        # Load dataset using qa_data module
        qa_dataset = get_dataset(dataset_name)

        # Pre-compute token lengths for all contexts
        context_token_lengths = {
            idx: (len(self.tokenizer.text_to_tokens(f"\n\nDocument XX:\n{context}")) + 3) # We add 3 tokens as a placeholder for the document number
            for idx, context in enumerate(qa_dataset.unique_contexts)
        }

        # Create initial samples list with token lengths
        processed_samples = []
        context_usage = {i: 0 for i in range(len(qa_dataset.unique_contexts))}
        
        # Pre-compute instruction tokens
        is_mcq = qa_dataset.is_mcq
        
        for sample in qa_dataset.qa_samples:
            # Get context and question
            context_idx = sample["context_idx"]
            context = qa_dataset.unique_contexts[context_idx]
            question = sample["question"]
            
            # Format the full prompt to get accurate token count
            formatted_context = f"\n\nDocument XX:\n{context}"
            input_text = QASample.format_sample(
                context=formatted_context,
                question=question,
                is_mcq=is_mcq,
                target_doc=2137
            )
            
            # Get total tokens from the full formatted prompt
            total_tokens = len(self.tokenizer.text_to_tokens(input_text))
            
            # Only keep samples that fit within token budget
            # Use min of MAX_SAMPLE_TOKENS and max_input_tokens to ensure compatibility
            max_allowed = min(self.MAX_SAMPLE_TOKENS, self.max_input_tokens) - self.template_tokens
            if total_tokens <= max_allowed:
                processed_samples.append({
                    "context": context,
                    "question": question,
                    "answer": sample["answer"],
                    "dataset": dataset_name,
                    "is_mcq": is_mcq,
                    "total_token_length": total_tokens
                })
                context_usage[context_idx] += 1
        
        # Only keep contexts that have associated questions
        valid_contexts = [
            {
                "text": qa_dataset.unique_contexts[idx],
                "token_length": context_token_lengths[idx]
            }
            for idx, count in context_usage.items()
            if count > 0
        ]
        
        # Shuffle samples using task random seed
        self.random_obj.shuffle(processed_samples)

        self.task_params["processed_dataset"] = processed_samples
        self.task_params["context_pool"] = valid_contexts

    def check_params(self) -> None:
        """Validate task parameters."""
        if not self.task_params.get("processed_dataset"):
            raise ValueError("Dataset not loaded")

        available = len(self.task_params["processed_dataset"])
        if available < self.num_samples:
            raise ValueError(
                f"Not enough samples in dataset. Found {available} samples after filtering "
                f"by max_input_tokens={self.max_input_tokens}, but {self.num_samples} were requested"
            )

    @property
    def sample_class(self):
        return QASample

    @staticmethod
    def evaluate(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate predictions using appropriate metrics.
        
        Args:
            predictions: List of prediction dictionaries containing:
                - pred: Model's raw response
                - gold_answer: Correct answer(s)
                - is_mcq: Boolean indicating if multiple choice
                
        Returns:
            Dictionary of metrics including mean and variance
        """
        import numpy as np
        mcq_predictions = [p for p in predictions if p["is_mcq"]]
        open_predictions = [p for p in predictions if not p["is_mcq"]]
        
        metrics = {}
        
        # Evaluate MCQ samples
        if mcq_predictions:
            sample_accuracies = []
            for p in mcq_predictions:
                # Extract answer from tagged response
                _, pred_answer = extract_tagged_response(p["pred"])
                norm_pred = normalize_mcq_answer(pred_answer)
                norm_gold = normalize_mcq_answer(p["gold_answer"])
                # Binary accuracy for this sample (1 if correct, 0 if incorrect)
                accuracy = 1.0 if norm_pred == norm_gold else 0.0
                sample_accuracies.append(accuracy)
                
            metrics["accuracy"] = np.mean(sample_accuracies)
            # Use ddof=1 for unbiased estimate of the variance
            metrics["accuracy_variance"] = np.var(sample_accuracies, ddof=1) if len(sample_accuracies) > 1 else 0.0
        
        # Evaluate open-ended samples
        if open_predictions:
            sample_f1_scores = []
            
            for pred in open_predictions:
                # Extract answer from tagged response
                _, pred_answer = extract_tagged_response(pred["pred"])
                norm_pred = normalize_answer(pred_answer)
                
                # Calculate maximum F1 across all reference answers
                max_f1 = max(
                    f1_score(norm_pred, normalize_answer(gold))
                    for gold in pred["gold_answer"]
                )
                sample_f1_scores.append(max_f1)
            
            metrics["f1"] = np.mean(sample_f1_scores)
            # Use ddof=1 for unbiased estimate of the variance
            metrics["f1_variance"] = np.var(sample_f1_scores, ddof=1) if len(sample_f1_scores) > 1 else 0.0
        
        return metrics
