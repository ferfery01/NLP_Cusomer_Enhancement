from typing import Any, ClassVar, NamedTuple, Optional, Sequence, TypedDict, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoProcessor, pipeline

from unified_desktop.core.utils.logging import setup_logger
from unified_desktop.pipelines.base import UDBase

logger = setup_logger()


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
    """A class for performing Sentiment Detection using transformer models."""

    models_list: ClassVar[Sequence[str]] = (
        "j-hartmann/sentiment-roberta-large-english-3-classes",
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
    )

    def __init__(
        self,
        model_id: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Initialize the UDSentimentDetection instance.

        Args:
            model_id (str): The name of the ALBERT model to use.
            torch_dtype (torch.dtype): The data type for torch tensors.
            device (Union[str, torch.device, None]): The device to run the model on. If None, the
                default device is used.
        """
        super().__init__(model_id=model_id, torch_dtype=torch_dtype, device=device)

    def _load_model_or_pipeline(self) -> None:
        """
        Load the sentiment analysis model using the specified name.
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        model = model.to_bettertransformer().to(self.device)
        processor = AutoProcessor.from_pretrained(self.model_id)
        logger.info(f"Loaded model {self.model_id} on device {self.device}")

        self.pipeline = pipeline(
            task="text-classification",
            model=model,
            tokenizer=processor,
            use_fast=True,
            device=self.device,
            torch_dtype=self.torch_dtype,
            top_k=1,
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
        cls_output = self.pipeline(input_text, **kwargs)
        return cls_output[0][0]

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
