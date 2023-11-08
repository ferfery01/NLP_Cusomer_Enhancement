import re
from typing import ClassVar, List, Optional, Sequence, TypedDict, Union

import torch
from nltk.corpus import stopwords
from textblob import Word
from transformers import AutoModelForTokenClassification, AutoProcessor, pipeline

from unified_desktop.core.utils.logging import setup_logger
from unified_desktop.pipelines.base import UDBase

logger = setup_logger()


class RawKeyPredictions(TypedDict):
    """The predictions from the model."""

    entity: str
    score: float
    index: int
    word: str
    start: int
    end: int


class KeyPredictions(TypedDict):
    """The filtered predictions from the model."""

    score: float
    index: int
    word: str


class UDKeyExtractor(UDBase):
    """A class for performing Keyword Extraction using the BERT based model."""

    models_list: ClassVar[Sequence[str]] = ["yanekyuk/bert-uncased-keyword-extractor"]
    regex: re.Pattern[str] = re.compile(r"([^\W\d_])\1{2,}")

    def __init__(
        self,
        model_id: str = "yanekyuk/bert-uncased-keyword-extractor",
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
        """Load the ALBERT model for text classification."""
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        model = model.to_bettertransformer().to(self.device)
        processor = AutoProcessor.from_pretrained(self.model_id)
        logger.info(f"Loaded model {self.model_id} on device {self.device}")

        self.pipeline = pipeline(
            task="token-classification",
            model=model,
            tokenizer=processor,
            use_fast=True,
            device=self.device,
            torch_dtype=self.torch_dtype,
        )

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

    def _predict(self, input_text: str) -> List[RawKeyPredictions]:
        """Predict the intent of the input text.

        Args:
            input_text: The input text for keyword extraction.

        Returns:
            prediction results (list of keys in "RawKeyPredictions")
        """
        return self.pipeline(input_text)

    def _postprocess(self, predictions: List[RawKeyPredictions]) -> List[KeyPredictions]:
        """Postprocess the classification predictions.

        Args:
            predictions: The list of raw classification predictions.
            sw_nltk: stop words

        Returns:
            List[KeyPredictions]:A list of dict [index, the keyword, score] - remove
            repetetive and stop words.
        """

        # Create an empty set to keep track of unique words
        seen_words = set()

        # Initialize a list to store the filtered predictions
        filtered_predictions = []

        for item in predictions:
            word = item["word"].lower()  # Access the attributes correctly using brackets

            if word not in self.sw_nltk and word not in seen_words:
                seen_words.add(word)
                filtered_predictions.append(
                    KeyPredictions(score=item["score"], index=item["index"], word=item["word"])
                )

        return filtered_predictions

    def __call__(self, input_text: str) -> List[KeyPredictions]:
        """Make a keyword extraction prediction.

        Args:
            input_text: The input text for keyword extraction.

        Returns:
            List[KeyPredictions]: A list of tuples [index of the word, the keyword, score].
        """
        input_text = self._preprocess(input_text)
        predictions = self._predict(input_text)
        return self._postprocess(predictions)
