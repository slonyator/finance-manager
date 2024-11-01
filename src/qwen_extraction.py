import sys

import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)


class QwenExtractor:

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    def load_model_and_tools(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        precision=torch.float16,
    ):
        """Load the Qwen2-VL model, tokenizer, and processor for Apple Silicon."""
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=precision
            )
            model.to(self.device)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            processor = AutoProcessor.from_pretrained(model_name)
            return model, tokenizer, processor
        except Exception as e:
            print(f"Failed to load the model: {e}")
            sys.exit(1)
