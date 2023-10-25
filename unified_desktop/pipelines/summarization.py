from typing import Any, Optional, Union

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

from unified_desktop.pipelines.base import UDBase


class UDSummarizer(UDBase):
    """
    A class for summarizing content from a file using the BART-base model.

    Attributes:
        device (str): PyTorch device for the model.
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None) -> None:
        super().__init__(device=device)

    def _validate_args(self) -> None:
        """Validate the arguments."""
        pass

    def _load_model(self) -> None:
        """Load the BART-base model and tokenizer."""
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.model = BartForConditionalGeneration.from_pretrained("ainize/bart-base-cnn").to(self.device)
        self.model.eval()

    def _preprocess(self, content_file: str) -> torch.LongTensor:
        """
        This function tokenizes the provided content file, while ensuring it doesn't exceed
        a predefined token limit (e.g., 1024 tokens in BART model).

        Args:
            content_file (str): The content to be tokenized.

        Returns:
            torch.LongTensor: The tokenized content.
        """

        # The length of the file is needed to compare with the length of the summarization
        self.content_length = len(content_file)

        return self.tokenizer.encode(content_file[:1024], return_tensors="pt").to(self.device)

    def _predict(self, content_tokenized: torch.LongTensor, **kwargs: Any) -> torch.LongTensor:
        """Predicts a summarized version of the provided tokenized content using the BART-base model.

        Args:
            content_tokenized (torch.LongTensor): Tokenized content to be summarized.
            **kwargs: Additional keyword arguments for the model's `generate` method.

        Returns:
            torch.LongTensor: The token IDs of the generated summary.

        Note:
            bos_token_id: the representation token of the beginning of the sequence
            eos_token_id: the representation token of the end of the sequence
        """

        # Get the max_length and min_length if it is in kwargs
        max_length = kwargs.get("max_length", None)
        min_length = kwargs.get("min_length", None)

        # Constraints: 1) both min_length and max_length cannot exceed transcription length,
        #              2) max_length > min_length should be ensured if both are provided.

        # If both max_length and min_length are provided, ensure max_length > min_length
        if max_length and min_length and min_length >= max_length:
            raise ValueError("Minimum length should be smaller than maximum length.")

        # Check if max_length do not exceed content_length
        if max_length and max_length > self.content_length:
            raise ValueError("Maximum length cannot exceed transcription length.")

        summary_text_ids = self.model.generate(
            input_ids=content_tokenized,
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=self.model.config.eos_token_id,
            **kwargs,
        )

        return summary_text_ids

    def _postprocess(self, summary_text_ids: torch.LongTensor, skip_special_tokens: bool = True) -> str:
        """
        Post-process the output of the model to generate a human-readable summary.

        Args:
            summary_text_ids (torch.LongTensor): The tensor containing token IDs output by the model.
            skip_special_tokens (bool): Whether to remove special tokens like [CLS], [SEP], etc.
                                    during decoding to make the summary human-readable.

        Returns:
            str: The decoded summary text.
        """
        return self.tokenizer.decode(summary_text_ids[0], skip_special_tokens)

    def __call__(self, content_file: str, **kwargs: Any) -> str:
        """Perform summarization on the content file and return the summarized text.

        Args:
            content_file (str): The content to be summarized, provided as a string.
            **kwargs (Any): Additional keyword arguments to be passed to the prediction method.

        Returns:
            summary (str): The summary of the input content.

        Workflow:
            1. Preprocess the content to tokenize it.
            2. Predict/Generate the summarized content token IDs.
            3. Postprocess the predictions to obtain human-readable text.
        """
        content_tokenized = self._preprocess(content_file)
        predictions = self._predict(content_tokenized, **kwargs)
        summary = self._postprocess(predictions)

        return summary
