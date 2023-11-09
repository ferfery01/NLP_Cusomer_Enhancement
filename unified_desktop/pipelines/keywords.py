import re
from typing import ClassVar, List, Optional, Sequence, TypedDict, Union

import torch
from nltk.corpus import stopwords
from textblob import Word
from transformers import pipeline

from unified_desktop.core.utils.logging import setup_logger
from unified_desktop.pipelines.base import UDBase

logger = setup_logger()


class KeyPredictions(TypedDict):
    """The predictions from the model."""

    summary_text: str


class UDKeyExtractor(UDBase):
    """A class for performing Keyword Extraction using the BERT based model."""

    models_list: ClassVar[Sequence[str]] = ["transformer3/H2-keywordextractor"]
    regex: re.Pattern[str] = re.compile(r"([^\W\d_])\1{2,}")

    def __init__(
        self,
        model_id: str = "transformer3/H2-keywordextractor",
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize the UDKeyExtraction class.

        Args:
            model_id (str): The name of the ALBERT model to use.
            torch_dtype (torch.dtype): The data type for torch tensors.
            device (Union[str, torch.device, None]): The device to run the model on. If None, the
                default device is used.
        """
        super().__init__(model_id=model_id, torch_dtype=torch_dtype, device=device)
        self.sw_nltk = stopwords.words("english")

    def _load_model_or_pipeline(self) -> None:
        """Load the model for text classification."""
        self.pipeline = pipeline("summarization", model=self.model_id, device=self.device)

    def _preprocess(self, input_text: str) -> str:
        """Preprocess the input text.

        Args:
            input_text: The input text to preprocess.

        Returns:
            str: The preprocessed input text as a string: removing repetetive letters in wordsxs.
        """
        input_text = re.sub(
            r"[^\W\d_]+",
            lambda x: Word(self.regex.sub(r"\1\1", x.group())).correct()
            if self.regex.search(x.group())
            else x.group(),
            input_text,
        )

        return input_text

    def _predict(self, input_text: str) -> List[KeyPredictions]:
        """Predict the intent of the input text.

        Args:
            input_text: The input text for keyword extraction.

        Returns:
            prediction results (list of keys in "KeyPredictions")
        """
        return self.pipeline(input_text)

    def _postprocess(self, predictions: List[KeyPredictions]) -> List[str]:
        """Postprocess the classification predictions.

        Args:
            predictions: The list of classification predictions.
            sw_nltk: stop words

        Returns:
            List[KeyPredictions]:A list of dict [summary_text] - remove
            repetetive and stop words.
        """

        # Create an empty set to keep track of unique words
        seen_words = set()

        # Initialize a list to store the filtered predictions
        filtered_predictions = []

        for item in predictions:
            word = item["summary_text"].lower()  # Access the attributes correctly using brackets

            if word not in self.sw_nltk and word not in seen_words:
                seen_words.add(word)
                filtered_predictions.append(item["summary_text"])

        return filtered_predictions

    def __call__(self, input_text: str) -> List[str]:
        """Make a keyword extraction prediction.

        Args:
            input_text: The input text for keyword extraction.

        Returns:
            A list of [summary_text].
        """
        input_text = self._preprocess(input_text)
        predictions = self._predict(input_text)
        return self._postprocess(predictions)
