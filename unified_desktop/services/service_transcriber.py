import io
import queue
import threading
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pyaudio
import speech_recognition as sr

from unified_desktop.core.utils.logging import setup_logger
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
        - start_processing(): Begins recording and transcribing in real-time.
        - sop_processing(): Stops the recording and processing.

    2. File-based Transcription:
        - transcribe(audio): Directly transcribes provided audio (file path, BytesIO stream, or numpy array).

    Examples:
    1. Live Transcription:
        transcriber = SpeechTranscriber()
        transcriber.initialize_microphone(device_index=0)
        transcriber.start_processing()

        while True:
            pass

        transcriber.stop_processing()

    2. File-based Transcription:
        transcriber = SpeechTranscriber()
        transcription = transcriber.transcribe('path_to_audio_file.wav')
    """

    def __init__(self, speech_recognizer: Optional[UDSpeechRecognizer] = None) -> None:
        self._speech_recognizer = speech_recognizer or UDSpeechRecognizer()
        self._transcription: str = ""
        self._is_recording: bool = False

    @classmethod
    def list_microphones(self) -> None:
        """Lists the available microphones."""
        logger.info("🎤 Available microphones:")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"Microphone {index}: {name}")

    def initialize_microphone(self, device_index: Optional[int] = None, duration: int = 2) -> None:
        """Initializes microphone settings for audio capture."""
        self._recognizer = sr.Recognizer()
        self._recognizer.dynamic_energy_threshold = False

        self._microphone = sr.Microphone(device_index=device_index, sample_rate=16000)
        device_index = device_index or pyaudio.PyAudio().get_default_input_device_info()["index"]
        logger.info(f"Using {sr.Microphone.list_microphone_names()[device_index]} as microphone")

        self._audio_queue: queue.Queue[Optional[sr.AudioData]] = queue.Queue()
        self._stop_listening = None  # Handle to stop the background listener

        # Adjust the recognizer sensitivity to ambient noise and start listening in the background
        with self._microphone as source:
            logger.info("🎧 Adjusting for ambient noise...")
            self._recognizer.adjust_for_ambient_noise(source, duration)

    def _record_callback(self, recognizer: sr.Recognizer, audio: sr.AudioData) -> None:
        """Receive audio data and enqueue it for processing."""
        try:
            self._audio_queue.put(audio)
        except Exception as e:
            logger.error(f"Error enqueuing audio data: {e}")

    def start_recording(self) -> None:
        """Adjust the recognizer sensitivity to ambient noise, start recording, and start processing
        the audio queue."""
        if not self._microphone:
            raise RuntimeError("Microphone not initialized. Call self.initialize_microphone() first.")

        if self._is_recording:
            return None  # Already recording

        # Set flag to indicate that we are recording
        self._is_recording = True

        # Create a background thread that will pass us raw audio bytes.
        self._stop_listening = self._recognizer.listen_in_background(self._microphone, self._record_callback)

    def start_processing(self):
        self.start_recording()

        # Start processing the audio queue
        self._audio_processing_thread = threading.Thread(target=self._process_audio_queue)
        self._audio_processing_thread.start()
        logger.info("🎤 Listening...")

    def stop_recording(self) -> None:
        """Stop the recording, signal the processing thread to stop, and wait for it to finish."""
        if not self._is_recording:
            return None  # Not recording

        # Set flag to indicate that we are no longer recording
        self._is_recording = False

        if hasattr(self, "_stop_listening") and self._stop_listening:
            self._stop_listening(wait_for_stop=False)

        self._audio_queue.put(None)  # Enqueue None to signal the thread to stop

    def stop_processing(self):
        self.stop_recording()

        if hasattr(self, "_audio_processing_thread"):
            self._audio_processing_thread.join()
        logger.info("🛑 Stopped listening.")

    def _process_audio_queue(self) -> None:
        """Processes audio chunks from the queue and transcribes them."""
        while True:
            audio_chunk = self._audio_queue.get()
            if audio_chunk is None:  # Stop signal
                break

            # Use AudioData to convert the raw data to wav data.
            data = audio_chunk.get_wav_data()

            # Transcribe the audio chunk
            transcription = self.transcribe(data)
            self._transcription += f"{transcription} "
            print(transcription)

    def transcribe(self, audio: Union[str, Path, bytes, io.BytesIO, np.ndarray]) -> str:
        """Transcribes audio data using the UDSpeechRecognizer."""
        return self._speech_recognizer(audio)

    @property
    def transcription(self) -> str:
        """Returns the current transcription."""
        return self._transcription

    @property
    def is_recording(self) -> bool:
        """Indicates whether the service is currently recording."""
        return self._is_recording


if __name__ == "__main__":
    transcriber = SpeechTranscriber()

    try:
        # Start microphone and processing
        transcriber.initialize_microphone()
        transcriber.start_processing()
        while True:  # Keep the main thread alive
            pass
    except KeyboardInterrupt:
        # This block will execute when Ctrl+C is pressed
        logger.warning("Received interrupt, stopping processing...")
    finally:
        # Stop processing and cleanup
        transcriber.stop_processing()
        logger.info(f"Full Audio Speech Transcription: {transcriber.transcription}")
