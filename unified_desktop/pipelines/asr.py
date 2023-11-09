import io
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Sequence, Union

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from unified_desktop.core.utils.logging import setup_logger
from unified_desktop.pipelines.base import UDBase

logger = setup_logger()


class UDSpeechRecognizer(UDBase):
    """A class for Automatic Speech Recognition using transformer models.

    Attributes:
        model_id (str): Identifier for the model to be used.
        chunk_length_s (int): Length of audio chunks to be processed in seconds.
        batch_size (int): Batch size for processing.
        torch_dtype (torch.dtype): Data type for torch tensors.
        device (torch.device): Device on which the model will run.
        pipeline (transformers.pipeline): Pipeline for ASR.
    """

    models_list: ClassVar[Sequence[str]] = (
        "openai/whisper-tiny.en",
        "openai/whisper-tiny",
        "openai/whisper-base.en",
        "openai/whisper-base",
        "openai/whisper-small.en",
        "openai/whisper-small",
        "openai/whisper-medium.en",
        "openai/whisper-medium",
        "openai/whisper-large",
        "openai/whisper-large-v1",
        "openai/whisper-large-v2",
        "openai/whisper-large-v3",
        "distil-whisper/distil-medium.en",
        "distil-whisper/distil-large-v2",
    )

    def __init__(
        self,
        model_id: str = "openai/whisper-tiny.en",
        chunk_length_s: Optional[int] = 15,
        batch_size: Optional[int] = 4,
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.chunk_length_s = chunk_length_s
        self.batch_size = batch_size
        super().__init__(model_id=model_id, torch_dtype=torch_dtype, device=device)

    def _validate_args(self) -> None:
        """Validate the provided arguments."""
        super()._validate_args()

        if self.device == torch.device("mps"):
            raise ValueError("ASR does not support MPS. Please use another device.")

    def _load_model_or_pipeline(self) -> None:
        """Initializes and returns the ASR pipeline with the specified model."""
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model = model.to_bettertransformer().to(self.device)
        processor = AutoProcessor.from_pretrained(self.model_id)
        logger.info(f"Loaded model {self.model_id} on device {self.device}")

        self.pipeline = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            use_fast=True,
            device=self.device,
            torch_dtype=self.torch_dtype,
            chunk_length_s=self.chunk_length_s,
            batch_size=self.batch_size,
        )

    def _preprocess(self, audio: Union[str, Path, bytes, io.BytesIO, np.ndarray]) -> Union[str, np.ndarray]:
        """Preprocesses the audio input for the pipeline."""
        if isinstance(audio, (str, Path)):
            return str(audio)
        elif isinstance(audio, np.ndarray):
            return audio
        elif isinstance(audio, bytes):
            audio = io.BytesIO(audio)

        if isinstance(audio, io.BytesIO):
            audio.seek(0)
            audio_array, _ = sf.read(audio, dtype="float32")
            return audio_array

        raise TypeError(f"Invalid audio type: {type(audio)}")

    def _predict(self, audio: Union[str, bytes, np.ndarray], **kwargs: Any) -> Dict[str, str]:
        """Run the ASR model on the audio file."""
        return self.pipeline(audio, **kwargs)

    def _postprocess(self, predictions: Dict[str, str]) -> str:
        """Postprocesses the model's prediction to extract the text."""
        return predictions["text"].strip()

    def __call__(self, audio: Union[str, Path, bytes, io.BytesIO, np.ndarray], **kwargs: Any) -> str:
        """Perform ASR on the audio file and return the transcribed or translated text."""
        audio_data = self._preprocess(audio)
        predictions = self._predict(audio_data, **kwargs)
        transcribed_text = self._postprocess(predictions)
        return transcribed_text
