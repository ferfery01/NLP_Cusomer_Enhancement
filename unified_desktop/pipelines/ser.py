from pathlib import Path
from typing import NamedTuple, Optional, Union

import torch
from speechbrain.pretrained.interfaces import foreign_class

from unified_desktop import CACHE_DIR
from unified_desktop.pipelines.base import UDBase


class EmotionPredictions(NamedTuple):
    """A named tuple for holding the emotion predictions."""

    logits: torch.Tensor
    """The log posterior probabilities of each class ([1, N_class])
    """
    score: torch.Tensor
    """Value of the log-posterior for the best class
    """
    cls_index: torch.Tensor
    """The indexes of the best class
    """
    label: str
    """The emotion label of the best class
    """


class UDSpeechEmotionRecognition(UDBase):
    """Detect emotions in speech using SpeechBrain's emotion recognition model.

    Attributes:
       device (str, torch.device): PyTorch device for the model. If None, defaults to the best
           available device.
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None) -> None:
        super().__init__(device=device)

    def _load_model(self) -> None:
        """Load the emotion recognition model."""
        self.model = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir=CACHE_DIR,
        )

    def _preprocess(self, audio_file: Union[str, Path]) -> str:
        """Convert the audio file path to a string."""
        return str(audio_file)

    def _predict(self, audio_file: str) -> EmotionPredictions:
        """Run the ASR model on the audio file."""
        pred = self.model.classify_file(audio_file)
        return EmotionPredictions(pred[0], pred[1], pred[2], pred[3][0])

    def _postprocess(self, predictions: EmotionPredictions) -> str:
        """Extract the transcribed text from the ASR predictions."""
        return predictions.label

    def __call__(self, audio_file: Union[str, Path]) -> str:
        audio_file_str = self._preprocess(audio_file)
        predictions = self._predict(audio_file_str)
        transcribed_text = self._postprocess(predictions)
        return transcribed_text
