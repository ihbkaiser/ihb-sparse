from abc import ABC, abstractmethod
from typing import Dict, Any
import torch


class AbstractModel(ABC):
    @abstractmethod
    def __init__(
        self,
        model_path: str,
        max_input_tokens: int = 8192,
        max_output_tokens: int = 256,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        """Initialize the model.

        Args:
            model_path: Path or identifier for the model
            max_input_tokens: Maximum number of input tokens
            max_output_tokens: Maximum number of new tokens to generate
            device: Device to run the model on
            dtype: Data type for model parameters
        """
        pass

    @abstractmethod
    def _load_model(self, model_path: str) -> Any:
        """Load the underlying model.

        Args:
            model_path: Path or identifier for the model

        Returns:
            The loaded model
        """
        pass

    @abstractmethod
    def _sampling_config(self, max_output_tokens: int) -> Dict:
        """Get configuration for a generation call.

        Args:
            max_output_tokens: Maximum number of new tokens to generate

        Returns:
            Dictionary of generation configuration parameters
        """
        pass

    @abstractmethod
    def generate(
        self,
        input_text: str,
        max_output_tokens: int = None,
    ) -> str:
        """Generate text from input.

        Args:
            input_text: The input text to generate from
            max_output_tokens: Maximum number of new tokens to generate

        Returns:
            Generated text string
        """
        pass
