import io
import queue
import threading
from pathlib import Path
from typing import Optional, Union

import numpy as np
import speech_recognition as sr

from unified_desktop.core.utils.logging import setup_logger
from unified_desktop.core.utils.timer import timer
from unified_desktop.pipelines.asr import UDSpeechRecognizer

logger = setup_logger()


class SpeechTranscriber:
    """Provides a service to transcribe speech from various audio sources using the UDSpeechRecognizer.

    The class can transcribe live audio from a selected microphone or pre-recorded audio files. It operates by
    capturing audio in real-time, queuing the audio data, and processing the queue in a separate thread to transcribe
    the audio using a transformer-based ASR model.

    Usage:
    1. Live Microphone Transcription:
        - list_microphones(): Lists available microphones.
        - initialize_microphone(device_index): Sets up the desired microphone.
        - start_recording(): Begins recording and transcribing in real-time.
        - stop_recording(): Stops the recording and processing.

    2. File-based Transcription:
        - transcribe(audio): Directly transcribes provided audio (file path, BytesIO stream, or numpy array).

    Examples:
    1. Live Transcription:
        transcriber = SpeechTranscriber()
        transcriber.initialize_microphone(device_index=0)
        transcriber.start_recording()

        try:
            while True:
                pass
        finally:
            asr_service.stop_recording()

    2. File-based Transcription:
        transcriber = SpeechTranscriber()
        transcription = transcriber.transcribe('path_to_audio_file.wav')
    """

    def __init__(self, speech_recognizer: Optional[UDSpeechRecognizer] = None) -> None:
        self.speech_recognizer = speech_recognizer or UDSpeechRecognizer()

    @classmethod
    def list_microphones(self) -> None:
        """Lists the available microphones."""
        logger.info("🎤 Available microphones:")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"Microphone {index}: {name}")

    def initialize_microphone(self, device_index: Optional[int] = None, duration: int = 1) -> None:
        """Initializes microphone settings for audio capture."""
        self._recognizer = sr.Recognizer()
        self._recognizer.dynamic_energy_threshold = False

        self._microphone = sr.Microphone(device_index=device_index, sample_rate=16000)
        logger.info(f"Using {sr.Microphone.list_microphone_names()[device_index]} as microphone")

        self._audio_queue: queue.Queue[Optional[sr.AudioData]] = queue.Queue()
        self._stop_listening = None  # Handle to stop the background listener

        with self._microphone as source:
            logger.info("🎧 Adjusting for ambient noise...")
            self._recognizer.adjust_for_ambient_noise(source, duration)

    def _record_callback(self, recognizer: sr.Recognizer, audio: sr.AudioData) -> None:
        """Receive audio data and enqueue it for processing."""
        try:
            with timer(name="record_callback") as t:
                self._audio_queue.put(audio)
            logger.info(f"🎤 Recorded audio in {t.duration:0.4f}s")
        except Exception as e:
            logger.error(f"Error enqueuing audio data: {e}")

    def start_recording(self) -> None:
        """Adjust the recognizer sensitivity to ambient noise, start recording, and start processing
        the audio queue."""
        if not self._microphone:
            raise RuntimeError("Microphone not initialized. Call self.initialize_microphone() first.")

        # Adjust the recognizer sensitivity to ambient noise and start listening in the background
        with self._microphone as source:
            self._recognizer.adjust_for_ambient_noise(source)

        # Create a background thread that will pass us raw audio bytes.
        self._stop_listening = self._recognizer.listen_in_background(self._microphone, self._record_callback)

        # Start processing the audio queue
        self._processing_thread = threading.Thread(target=self._process_audio_queue)
        self._processing_thread.start()
        logger.info("🎤 Listening...")

    def stop_recording(self) -> None:
        """Stop the recording, signal the processing thread to stop, and wait for it to finish."""
        if self._stop_listening is not None:
            self._stop_listening(wait_for_stop=False)

        self._audio_queue.put(None)  # Enqueue None to signal the thread to stop
        if self._processing_thread.is_alive():
            self._processing_thread.join()
        logger.info("🛑 Stopped listening.")

    def _process_audio_queue(self) -> None:
        """Processes audio chunks from the queue and transcribes them."""
        while True:
            audio_chunk = self._audio_queue.get()
            if audio_chunk is None:  # Stop signal
                break

            # Use AudioData to convert the raw data to wav data.
            data = audio_chunk.get_wav_data()
            print(self.transcribe(data))

    def transcribe(self, audio: Union[str, Path, io.BytesIO, np.ndarray]) -> str:
        """Transcribes audio data using the UDSpeechRecognizer."""
        return self.speech_recognizer(audio)


if __name__ == "__main__":
    transcriber = SpeechTranscriber()

    transcriber.initialize_microphone()
    transcriber.start_recording()

    try:
        while True:
            pass
    finally:
        transcriber.stop_recording()
