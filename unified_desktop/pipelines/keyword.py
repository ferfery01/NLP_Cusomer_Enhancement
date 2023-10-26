from typing import ClassVar, List, Optional, Tuple, TypedDict, Union

import torch
from transformers import pipeline

from unified_desktop.pipelines.base import UDBase


class IntentPredictions(TypedDict):
    """The predictions from the Whisper ASR model."""

    entity: str
    score: float
    index: int
    word: str
    start: int
    end: int


class UDKeyExtraction(UDBase):
    """
    This class is used for keyword extraction using the ALBERT model.

    ALBERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion.

    Goal: Detect the intent of the customers using their text queries.
    """

    """
    available_models are the list of all transformer models
    that works well for the IntentDetection purpose.
    More will be added to the list after testing each one.
    """
    available_models: ClassVar[List[str]] = ["yanekyuk/bert-uncased-keyword-extractor"]

    def __init__(
        self,
        name: str = "yanekyuk/bert-uncased-keyword-extractor",
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize the UDKeyExtraction class.

        Args:
            name (str): The name of the ALBERT model to use.
            device (Union[str, torch.device, None]):
            The device to run the model on. The device can be
            specified as a string, a torch.device, or left
            as None to use the default device.
        """
        self.name = name
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
        # Use a pipeline as a high-level helper

        self.model = pipeline("token-classification", model=self.name, device=self.device)

    def _preprocess(self, input_text: str) -> str:
        """
        Preprocess the input text.

        Args:
            input_text: The input text to preprocess.

        Returns:
            str: The preprocessed input text as a string.
        """
        return input_text

    def _predict(self, input_text: str) -> List[IntentPredictions]:
        """
        Predict the intent of the input text.

        Args:
            input_text: The input text for keyword extraction.

        Returns:
            classification results.
        """
        cls_output = self.model(input_text)
        return cls_output

    def _postprocess(self, predictions: List[IntentPredictions]) -> List[Tuple[int, str, float]]:
        """
        Postprocess the classification predictions.

        Args:
            predictions: The raw classification predictions.

        Returns:
            A list of list containing keywords and confidence score.
        """
        prediction = [
            (item["index"], item["word"], item["score"])
            for item in predictions
            if item["word"] not in ["and", "or", "is", "are", "can", "can't"]
        ]
        return prediction

    def __call__(self, input_text: str) -> List[Tuple[int, str, float]]:
        """
        Make an keyword extraction prediction.

        Args:
            input_text: The input text for keyword extraction.

        Returns:
            A list of dictionaries containing intent label and score.
        """
        input_text = self._preprocess(input_text)
        predictions = self._predict(input_text)
        return self._postprocess(predictions)
