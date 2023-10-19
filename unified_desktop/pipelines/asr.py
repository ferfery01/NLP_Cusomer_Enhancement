from pathlib import Path
from typing import Any, List, Optional, TypedDict, Union

import torch
import whisper

from unified_desktop.pipelines.base import UDBase


class SegmentDetails(TypedDict):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class WhisperPredictions(TypedDict):
    """The predictions from the Whisper ASR model."""

    text: str
    """Resulting transcribed text.
    """
    segment: List[SegmentDetails]
    """Segment-level details.
    """
    language: str
    """Spoken language of the audio.
    """


class UDSpeechRecognizer(UDBase):
    """Perform Automatic Speech Recognition (ASR) using OpenAI's Whisper ASR model.

    This class transcribes audio files and offers an option to translate the transcribed text to English.
    The model and task to perform (either 'transcribe' or 'translate') can be specified during instantiation.

     Attributes:
        name (str): One of the official model names listed by `whisper.available_models()`
        task (str, optional): Task to perform, either 'translate' or 'transcribe'. Defaults to None.
        device (str, torch.device): PyTorch device for the model. If None, defaults to the best
            available device.
        model (whisper.ASRModel): The loaded Whisper ASR model.
    """

    def __init__(
        self,
        name: str = "tiny.en",
        task: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.name = name
        self.task = task
        super().__init__(device=device)

    def _validate_args(self) -> None:
        """Validate the arguments."""
        available_models = whisper.available_models()
        if self.name not in available_models:
            raise ValueError(f"Model {self.name} not found; available models: {available_models}")
        if self.task and self.task not in ("translate", "transcribe"):
            raise ValueError(f"Task {self.task} not supported; supported tasks: ('translate', 'transcribe')")

    def _load_model(self) -> None:
        """Load the ASR model."""
        self.model = whisper.load_model(self.name, device=self.device)
        self.model.eval()

    def _preprocess(self, audio_file: Union[str, Path]) -> str:
        """Convert the audio file path to a string."""
        return str(audio_file)

    def _predict(self, audio_file: str, **kwargs: Any) -> WhisperPredictions:
        """Run the ASR model on the audio file."""
        return self.model.transcribe(audio_file, **kwargs)

    def _postprocess(self, predictions: WhisperPredictions) -> str:
        """Extract the transcribed text from the ASR predictions."""
        return predictions["text"]

    def __call__(self, audio_file: Union[str, Path], **kwargs: Any) -> str:
        """Perform ASR on the audio file and return the transcribed or translated text."""
        audio_file_str = self._preprocess(audio_file)
        predictions = self._predict(audio_file_str, **kwargs)
        transcribed_text = self._postprocess(predictions)
        return transcribed_text
