from typing import ClassVar, List, Optional, Sequence, TypedDict, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoProcessor, pipeline

from unified_desktop.core.utils.logging import setup_logger
from unified_desktop.pipelines.base import UDBase

logger = setup_logger()


class IntentPredictions(TypedDict):
    """The predictions from the model."""

    label: str
    score: float


class UDIntentClassifier(UDBase):
    """This class is used for intent classification using the ALBERT model.

    ALBERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion.

    Goal: Detect the intent of the customers using their text queries.
    """

    models_list: ClassVar[Sequence[str]] = ("vineetsharma/customer-support-intent-albert",)
    """These are the list of all transformer models that works well for the IntentDetection purpose.
    More will be added to the list after testing each one.
    """

    def __init__(
        self,
        model_id: str = "vineetsharma/customer-support-intent-albert",
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize the UDIntentClassifier class.

        Args:
            model_id (str): The name of the ALBERT model to use.
            torch_dtype (torch.dtype): The data type for torch tensors.
            device (Union[str, torch.device, None]): The device to run the model on. If None, the
                default device is used.
        """
        super().__init__(model_id=model_id, torch_dtype=torch_dtype, device=device)

    def _load_model_or_pipeline(self) -> None:
        """
        Load the ALBERT model for text classification.
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
            top_k=None,
        )

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
            input_text: The input text for intent classification.

        Returns:
            classification results.
        """
        cls_output = self.pipeline(input_text)

        # As we use a batch size of 1 during the inference, we need to extract the first element
        # from the list.
        return cls_output[0]

    def _postprocess(
        self, predictions: List[IntentPredictions], top_k: Optional[int]
    ) -> List[IntentPredictions]:
        """
        Postprocess the classification predictions.

        Args:
            predictions: The raw classification predictions.
            top_k: The number of top predictions to return. If None, all predictions are returned.

        Returns:
            A list of tuples containing intent class and confidence score.
        """
        return predictions[:top_k] if top_k else predictions

    def __call__(self, input_text: str, top_k: Optional[int] = None) -> List[IntentPredictions]:
        """
        Make an intent classification prediction.

        Args:
            input_text: The input text for intent classification.
            top_k: The number of top predictions to return. If None, all predictions are returned.

        Returns:
            A list of dictionaries containing intent label and score.
        """
        input_text = self._preprocess(input_text)
        predictions = self._predict(input_text)
        return self._postprocess(predictions, top_k)
