import os
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class DifficultyClassifier:
    """Classifies user queries as easy (0) or hard (1)."""

    def __init__(self, model_dir: Optional[str] = None, device_id: int = 0):
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, "embedding_router", "embedding_model")

        self.model_dir = model_dir

        if torch.cuda.is_available():
            if device_id < 0 or device_id >= torch.cuda.device_count():
                print(f"Warning: device_id {device_id} is invalid, falling back to cuda:0")
                device_id = 0

            self.device = torch.device(f"cuda:{device_id}")
            print(f"Using device: GPU {device_id} ({torch.cuda.get_device_name(device_id)})")
        else:
            self.device = torch.device("cpu")
            print("No GPU detected, using CPU")

        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded from {self.model_dir}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

    def predict(self, text: str) -> int:
        """Return 0 for easy queries, 1 for hard queries."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=-1).item()

        return int(pred_id)
