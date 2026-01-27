from transformers import AutoTokenizer
from typing import List, Dict, Union

import torch

class Tokenizer:
    def __init__(
        self,
        model_path: str,
        device: torch.device = None,
        thinking: bool = False,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.thinking = thinking

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def model_max_length(self) -> int:
        return self.tokenizer.model_max_length
    
    def text_to_tokens(self, input_text: str) -> List[int]:
        return self.tokenizer.encode(input_text, add_special_tokens=False)

    def encode_for_generation(self, input_text: str, return_tensors: bool = True) -> Dict:
        input_text_with_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )

        if return_tensors:
            encoded_input = self.tokenizer(
                input_text_with_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.device)

            return {
                **encoded_input,
                'input_length': encoded_input['input_ids'].size(1),
            }
        else:
            input_ids = self.text_to_tokens(input_text_with_prompt)
            return {
                'input_ids': input_ids,
                'input_length': len(input_ids),
            }

    def decode(self, outputs: Union[List[int], List[List[int]]], input_length: int = 0) -> Union[str, List[str]]:
        if isinstance(outputs[0], int):
            # Single list of tokens
            return self.tokenizer.decode(
                outputs[input_length:],
                skip_special_tokens=True
            )
        else:
            # List of lists of tokens
            return [
                self.tokenizer.decode(
                    output[input_length:],
                    skip_special_tokens=True
                )
                for output in outputs
            ]
