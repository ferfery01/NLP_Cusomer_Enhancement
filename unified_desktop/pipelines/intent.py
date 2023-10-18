from pathlib import Path
from typing import Any, Union, Optional

import torch
from transformers import pipeline
from unified_desktop.pipelines.base import UDBase

class UDIntentClassification(UDBase):
    def __init__(
        self,
        task: Optional[str] = None,
        name: str = "vineetsharma/customer-support-intent-albert",
        device: Union[str, torch.device, None] = None,
    ):
        self.name = name
        self.task = task
        super().__init__(device=device)

    def _validate_args(self):
        if self.name is None:
            raise ValueError(f"Model {self.name} not found")

    def _load_model(self):
        self.model = pipeline("text-classification", model=self.name, top_k=None)

    def _preprocess(self, input_text):
        return str(input_text)

    def _predict(self, input_text, **kwargs):
        cls_output = self.model(input_text, **kwargs)
        return cls_output


    def _postprocess(self, predictions):
        results = []
        for prediction in predictions[0]:
            intent_class = prediction["label"]
            intent_score = prediction["score"]
            results.append((intent_class, intent_score))
        return results

    def __call__(self, input_text, **kwargs):
        input_text = self._preprocess(input_text)
        predictions = self._predict(input_text, **kwargs)
        return self._postprocess(predictions)
