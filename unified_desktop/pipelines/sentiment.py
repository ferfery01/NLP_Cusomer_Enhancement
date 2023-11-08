from typing import Any, ClassVar, List, NamedTuple, Optional, TypedDict, Union

import torch
from transformers import pipeline

from unified_desktop.pipelines.base import UDBase


class SentimentModelPredictions(TypedDict):
    """The predicted parameter from the sentiment detecton model."""

    label: str
    score: float


class SentimentPredictions(NamedTuple):
    """The predicted sentiment label and score."""

    sentiment: str
    score: float


def sentiment_degree(cur_senti: str, senti_score: float) -> SentimentPredictions:
    """
    Assigns an emoji and degree label to a sentiment based on the provided sentiment score.

    Args:
        cur_senti (str): The sentiment category, either "positive" or "negative" or "neutral".
        senti_score (float): The sentiment score, typically ranging from 0.0 to 1.0.

    Returns:
        SentimentPredictions: The display sentiment and the sentiment score.

    The function takes a sentiment category ('positive' or 'negative' or 'neutral') and a sentiment score as input.
    It assigns an appropriate emoji based on the score and returns a formatted label string.

    If 'cur_senti' is 'positive', it assigns a 😀 emoji.
    If 'cur_senti' is 'negative' and 'senti_score' is greater than 0.9, it assigns a 😡 emoji with a 'High' label.
    If 'cur_senti' is 'negative' and 'senti_score' is less than or equal to 0.9, it assigns a 😒 emoji.
    If none of the above conditions are met, it assigns a 😐 emoji which is for neutral label.
    """

    if cur_senti == "positive":
        emj = "😀 High" if senti_score >= 0.9 else "😊"
    elif cur_senti == "negative":
        emj = "😡 High" if senti_score >= 0.9 else "😒"
    else:
        emj = "😐"

    return SentimentPredictions(sentiment=f"{emj} {cur_senti.capitalize()}", score=senti_score)


class UDSentimentDetector(UDBase):
    """
    Utility class for performing sentiment detection using a specific model.

    Args:
        name (str): The name of the sentiment analysis model.
        device (Union[str, torch.device]): The device on which to run the model.

    This class is used for sentiment analysis with a specific model. It inherits from UDBase.
    """

    available_models: ClassVar[List[str]] = [
        "j-hartmann/sentiment-roberta-large-english-3-classes",
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
    ]

    def __init__(
        self,
        name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Initialize the UDSentimentDetection instance.

        Args:
            name (str): The name of the sentiment analysis model.
            device (Union[str, torch.device]): The device on which to run the model.
        """
        self.name = name
        super().__init__(device=device)

    def _validate_args(self) -> None:
        """
        Validate the model name to ensure it's a supported model.

        Raises:
            ValueError: If the provided model name is not supported.
        """
        if self.name not in self.available_models:
            raise ValueError(f"Model {self.name} not found; available models: {self.available_models}")

    def _load_model(self) -> None:
        """
        Load the sentiment analysis model using the specified name.
        """
        self.model = pipeline(
            "text-classification", model=self.name, device=self.device, return_all_scores=False
        )

    def _preprocess(self, input_text: str) -> str:
        """
        Preprocess the input text.

        Args:
            input_text (str): The input text to be preprocessed.

        Returns:
            str: The preprocessed input text.
        """
        return input_text

    def _predict(self, input_text: str, **kwargs: Any) -> SentimentModelPredictions:
        """
        Perform sentiment prediction on the preprocessed input text.

        Args:
            input_text (str): The preprocessed input text.
            kwargs: Additional keyword arguments for prediction.

        Returns:
            SentimentModelPredictions: The predicted sentiment label and score.
        """
        cls_output = self.model(input_text, **kwargs)
        return cls_output[0]

    def _postprocess(self, predictions: SentimentModelPredictions) -> SentimentPredictions:
        """
        Postprocess sentiment predictions and return the display sentiment.

        Args:
            predictions (dict): The output of the model containing the sentiment label and score.

        Returns:
            SentimentPredictions: The display sentiment and the sentiment score.
        """
        sentiment, score = predictions["label"], predictions["score"]
        return sentiment_degree(sentiment, score)

    def __call__(self, input_text: str, **kwargs: Any) -> SentimentPredictions:
        """
        Call the UDSentimentDetection instance to analyze the sentiment of the input text.

        Args:
            input_text (str): The input text to analyze.
            kwargs: Additional keyword arguments for prediction.

        Returns:
            SentimentPredictions: The display sentiment and the sentiment score.
        """
        input_text = self._preprocess(input_text)
        predictions = self._predict(input_text, **kwargs)
        return self._postprocess(predictions)
