from typing import Any, List, Optional, Tuple, Union

import torch
from transformers import pipeline

from unified_desktop.pipelines.base import UDBase


class UDIntentClassification(UDBase):
    """
    This class is used for intent classification using the ALBERT model.

    ALBERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion.

    Goal: Detect the intent of the customers using their text queries.
    """

    def __init__(
        self,
        name: str = "vineetsharma/customer-support-intent-albert",
        device: Optional[Union[str, torch.device]] = None,
        available_models: List[str] = [],
    ):
        """
        Initialize the UDIntentClassification class.

        Args:
            name (str): The name of the ALBERT model to use.
            device (Union[str, torch.device, None]):
            The device to run the model on. The device can be
            specified as a string, a torch.device, or left
            as None to use the default device.
        """
        self.name = name
        self.available_models = available_models
        super().__init__(device=device)

    def _validate_args(self) -> None:
        """
        Validate the provided arguments.

        Raises:
            ValueError: If the model name is None.
        """
        if self.name not in self.available_models:
            raise ValueError(f"Model {self.name} not found; available models: {self.available_models}")

    def _load_model(self) -> None:
        """
        Load the ALBERT model for text classification.
        """
        self.model = pipeline("text-classification", model=self.name, top_k=None)

    def _preprocess(self, input_text: str) -> str:
        """
        Preprocess the input text.

        Args:
            input_text: The input text to preprocess.

        Returns:
            str: The preprocessed input text as a string.
        """
        return input_text

    def _predict(self, input_text: str) -> List[Any | None]:
        """
        Predict the intent of the input text.

        Args:
            input_text: The input text for intent classification.

        Returns:
            list[Any | None]: A list of classification results.
        """
        cls_output = self.model(input_text)
        return cls_output

    def _postprocess(self, predictions, top_k: int) -> List[Tuple[List[str], List[float]]]:
        """
        Postprocess the classification predictions.

        Args:
            predictions: The raw classification predictions.
            top_k: Number of top brobability predictions

        Returns:
            List[Tuple[Any, Any]]: A list of tuples containing intent class and confidence score.
        """
        results = [(prediction["label"], prediction["score"]) for prediction in predictions[0][:top_k]]
        return results

    def __call__(self, input_text: str, top_k: int) -> List[Tuple[List[str], List[float]]]:
        """
        Make an intent classification prediction.

        Args:
            input_text: The input text for intent classification.
            top_k: Number of top brobability predictions

        Returns:
            List[Tuple[List[str], List[float]]]: A list of tuples containing intent class and confidence score.
        """
        input_text = self._preprocess(input_text)
        predictions = self._predict(input_text)
        return self._postprocess(predictions, top_k)
