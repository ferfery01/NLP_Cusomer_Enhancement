import re
from typing import ClassVar, List, Optional, Tuple, TypedDict, Union

import torch
from nltk.corpus import stopwords
from textblob import Word
from transformers import pipeline

from unified_desktop.pipelines.base import UDBase

sw_nltk = stopwords.words("english")


class KeyPredictions(TypedDict):
    """The predictions from the model."""

    entity: str
    score: float
    index: int
    word: str
    start: int
    end: int


class UDKeyExtraction(UDBase):
    """
    This class is used for keyword extraction using the BERT model.
    available_models are the list of all transformer models
    that works well for the keyword extraction purpose.
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
            name (str): The name of the BERT model to use.
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
            ValueError: If the model name is not in the list of available_models.
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
            str: The preprocessed input text as a string: removing repetetive letters in wordsxs.
        """

        rx = re.compile(r"([^\W\d_])\1{2,}")
        input_text = re.sub(
            r"[^\W\d_]+",
            lambda x: Word(rx.sub(r"\1\1", x.group())).correct() if rx.search(x.group()) else x.group(),
            input_text,
        )

        return input_text

    def _predict(self, input_text: str) -> List[KeyPredictions]:
        """
        Predict the intent of the input text.

        Args:
            input_text: The input text for keyword extraction.

        Returns:
            prediction results (list of keys in "KeyPredictions")
        """
        cls_output = self.model(input_text)
        return cls_output

    def _postprocess(self, predictions: List[KeyPredictions]) -> List[Tuple[int, str, float]]:
        """
        Postprocess the classification predictions.

        Args:
            predictions: The raw classification predictions.

        Returns:
            A list of tuples [index of the word, the keyword, score] - remove
            repetetive and stop words.
        """

        # Create an empty set to keep track of unique words
        seen_words = set()

        # Initialize a list to store the filtered predictions
        filtered_predictions = []

        for item in predictions:
            word = item["word"].lower()

            # Check if the word is not in the stop words list and is not a repetitive word
            if word not in sw_nltk and word not in seen_words:
                seen_words.add(word)  # Add the word to the set of seen words
                filtered_predictions.append((item["index"], item["word"], item["score"]))

        return filtered_predictions

    def __call__(self, input_text: str) -> List[Tuple[int, str, float]]:
        """
        Make an keyword extraction prediction.

        Args:
            input_text: The input text for keyword extraction.

        Returns:
            A list of tuples [index of the word, the keyword, score].
        """
        input_text = self._preprocess(input_text)
        predictions = self._predict(input_text)
        return self._postprocess(predictions)
