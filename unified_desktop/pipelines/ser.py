from pathlib import Path
from typing import ClassVar, Mapping, NamedTuple, Optional, Sequence, Union

import torch
from speechbrain.pretrained.interfaces import foreign_class

from unified_desktop import CACHE_DIR
from unified_desktop.core.utils.logging import setup_logger
from unified_desktop.pipelines.base import UDBase

logger = setup_logger()

EMOTIONS_MAP: Mapping[str, str] = {"neu": "😐", "ang": "😡", "hap": "😀", "sad": "😞"}
"""A mapping from the emotion labels to the corresponding human-readable labels.
"""


class EmotionPredictions(NamedTuple):
    """A named tuple for holding the emotion predictions."""

    logits: torch.Tensor
    """The log posterior probabilities of each class ([1, N_class])
    """
    score: float
    """Value of the log-posterior for the best class
    """
    label: str
    """The emotion label of the best class
    """


class UDSpeechEmotionRecognizer(UDBase):
    """Detect emotions in speech using SpeechBrain's emotion recognition model.

    Currently, the model predicts one of the following emotion: Anger, Happy, Neutral, Sad

    Attributes:
        model (EncoderWav2vec2Classifier): The loaded SpeechBrain emotion recognition model.
    """

    models_list: ClassVar[Sequence[str]] = ("speechbrain/emotion-recognition-wav2vec2-IEMOCAP",)

    def __init__(
        self,
        model_id: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(model_id=model_id, torch_dtype=torch_dtype, device=device)

    def _validate_args(self) -> None:
        """Validate the arguments passed to the constructor."""
        super()._validate_args()
        if self.device == torch.device("mps"):
            raise ValueError("SER does not support MPS. Please use another device.")

    def _load_model_or_pipeline(self) -> None:
        """Load the speech emotion recognition model."""
        self.model = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir=CACHE_DIR,
        ).to(self.device)

        # Need to set the device manually so that input tensors are on the correct device
        self.model.device = self.device
        self.model.eval()
        logger.info(f"Loaded SpeechBrain emotion recognition (SER) model on {self.device}")

    def _preprocess(self, audio_file: Union[str, Path]) -> str:
        """Convert the audio file path to a string."""
        return str(audio_file)

    def _predict(self, audio_file: str) -> EmotionPredictions:
        """Run the speech emotion recognition model on the audio file."""
        pred = self.model.classify_file(audio_file)
        return EmotionPredictions(pred[0], pred[1].item(), pred[3][0])

    def _postprocess(self, predictions: EmotionPredictions) -> EmotionPredictions:
        """Extract the emotion label from the predictions."""
        return EmotionPredictions(predictions.logits, predictions.score, EMOTIONS_MAP[predictions.label])

    def __call__(self, audio_file: Union[str, Path]) -> EmotionPredictions:
        """Perform speech emotion recognition on the audio file and return the emotion label."""
        audio_file_str = self._preprocess(audio_file)
        predictions = self._predict(audio_file_str)
        return self._postprocess(predictions)
