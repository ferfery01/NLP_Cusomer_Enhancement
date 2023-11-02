from typing import ClassVar, List, Optional, TypedDict, Union

import torch
from transformers import pipeline

from unified_desktop.pipelines.base import UDBase


def sentiment_degree(cur_senti: str, senti_score: float) -> str:
    """
    Assigns an emoji and degree label to a sentiment based on the provided sentiment score.

    Args:
    cur_senti (str): The sentiment category, either "positive" or "negative" or "neutral".
    senti_score (float): The sentiment score, typically ranging from 0.0 to 1.0.

    Returns:
    str: A label containing an emoji and the degree of the sentiment.

    The function takes a sentiment category ('positive' or 'negative' or 'neutral') and a sentiment score as input.
    It assigns an appropriate emoji based on the score and returns a formatted label string.

    If 'cur_senti' is 'positive', it assigns a 😀 emoji.
    If 'cur_senti' is 'negative' and 'senti_score' is greater than 0.9, it assigns a 😡 emoji with a 'High' label.
    If 'cur_senti' is 'negative' and 'senti_score' is less than or equal to 0.9, it assigns a 😒 emoji.
    If none of the above conditions are met, it assigns a 😐 emoji which is for neutral label.

    The resulting label string is then returned.
    """

    if cur_senti == "positive":
        emj = f'😀 {"High"}' if senti_score >= 0.9 else "\U0001F60A"
    elif cur_senti == "negative":
        emj = f'\U0001F621 {"High"}' if senti_score >= 0.9 else "\U0001F612"
    else:
        emj = "\U0001F610"

    label_out = f"{emj} {cur_senti.capitalize()}"
    return label_out


class SentimentPredictions(TypedDict):
    """The predicted parameter from the sentiment detecton model."""

    label: str
    score: float


class UDSentimentDetector(UDBase):
    """
    Utility class for performing sentiment detection using a specific model.

    Args:
    - name (str): The name of the sentiment analysis model.
    - device (Union[str, torch.device]): The device on which to run the model.

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
        - name (str): The name of the sentiment analysis model.
        - device (Union[str, torch.device]): The device on which to run the model.
        """
        self.name = name
        super().__init__(device=device)

    def _validate_args(self) -> None:
        """
        Validate the model name to ensure it's a supported model.

        Raises:
        - ValueError: If the provided model name is not supported.
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
        - input_text (str): The input text to be preprocessed.

        Returns:
        - str: The preprocessed input text.
        """
        return input_text

    def _predict(self, input_text: str, **kwargs) -> SentimentPredictions:
        """
        Perform sentiment prediction on the preprocessed input text.

        Args:
        - input_text (str): The preprocessed input text.
        - **kwargs: Additional keyword arguments for prediction.

        Returns:
        - dict: Output of the model.
        """
        cls_output = self.model(input_text, **kwargs)
        return cls_output[0]

    def _postprocess(self, predictions: SentimentPredictions) -> str:
        """
        Postprocess sentiment predictions and return the display sentiment.

        Args:
        - predictions (list): List of sentiment predictions.

        Returns:
        - str: Display sentiment with an emoji and degree label.
        """
        current_sentiment = predictions["label"]
        sentiment_score = predictions["score"]
        display_sentiment = sentiment_degree(current_sentiment, sentiment_score)
        return display_sentiment

    def __call__(self, input_text: str, **kwargs) -> str:
        """
        Call the UDSentimentDetection instance to analyze the sentiment of the input text.

        Args:
        - input_text (str): The input text to analyze.
        - **kwargs: Additional keyword arguments for prediction.

        Returns:
        - str: Display sentiment with an emoji and degree label.
        """
        input_text = self._preprocess(input_text)
        predictions = self._predict(input_text, **kwargs)
        output_sentiment = self._postprocess(predictions)
        return output_sentiment
