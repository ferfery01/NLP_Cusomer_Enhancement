from pathlib import Path
from typing import Mapping, NamedTuple, Optional, Union

import torch
from speechbrain.pretrained.interfaces import foreign_class

from unified_desktop import CACHE_DIR
from unified_desktop.pipelines.base import UDBase

EMOTIONS_MAP: Mapping[str, str] = {"ang": "Anger", "hap": "Happy", "neu": "Neutral", "sad": "Sad"}
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
        device (str, torch.device): PyTorch device for the model. If None, defaults to the best
           available device.
        model (EncoderWav2vec2Classifier): The loaded SpeechBrain emotion recognition model.
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None) -> None:
        super().__init__(device=device)

    def _load_model(self) -> None:
        """Load the speech emotion recognition model."""
        self.model = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir=CACHE_DIR,
        ).to(self.device)
        self.model.eval()

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
