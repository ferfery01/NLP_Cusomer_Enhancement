import sched
import threading
import time
from typing import List, Optional

from unified_desktop.core.utils.logging import setup_logger
from unified_desktop.pipelines import (
    UDIntentClassifier,
    UDKeyExtractor,
    UDSentimentDetector,
    UDSpeechRecognizer,
    UDSummarizer,
)
from unified_desktop.services.service_transcriber import SpeechTranscriber

logger = setup_logger()


class ContentEvaluator(SpeechTranscriber):
    """Extends `SpeechTranscriber` to include real-time intent, sentiment, and keyword
    extraction on transcribed speech data. It uses provided models to periodically process
    the transcriptions collected from speech.

    This class employs a non-blocking, threaded approach with a scheduler for periodic task
    execution. Users can start the process after initializing the microphone and can stop it
    when needed. The results of the intent and sentiment analysis are accessible as properties.

    Attributes:
        intent_classifier (UDIntentClassifier): An intent classification model.
        sentiment_detector (UDSentimentDetector): A sentiment analysis model.
        process_interval (float): Interval in seconds for processing transcriptions.
        last_n_sentences (int): Number of sentences to consider for sentiment analysis.
        shutdown_event (threading.Event): Signal to shutdown the processing thread.
        scheduler (sched.scheduler): Scheduler for processing tasks.
        _intent (str): Detected intent from the latest processed transcription.
        _sentiment (str): Detected sentiment from the latest processed transcription.
        _keywords (List[str]): Extracted keywords from the latest processed transcription.

    Usage:
        # Initialize the content evaluator with required models
        content_evaluator = ContentEvaluator()

        # Start microphone and processing
        content_evaluator.initialize_microphone()
        content_evaluator.start_processing()

        while True:
            pass # Keep the main thread alive

        # To stop, call:
        content_evaluator.stop_processing()

        # Access the results
        print(content_evaluator.intent, content_evaluator.sentiment, content_evaluator.keywords)
    """

    def __init__(
        self,
        speech_recognizer: Optional[UDSpeechRecognizer] = None,
        intent_classifier: Optional[UDIntentClassifier] = None,
        sentiment_detector: Optional[UDSentimentDetector] = None,
        keyword_extractor: Optional[UDKeyExtractor] = None,
        summarizer: Optional[UDSummarizer] = None,
        process_interval: float = 45.0,
        last_n_sentences: Optional[int] = 8,
        pause_threshold: float = 0.25,
        phrase_time_limit: Optional[int] = 15,
        summary_length_limit: Optional[int] = 100,
    ) -> None:
        super().__init__(
            speech_recognizer=speech_recognizer,
            summarizer=summarizer,
            pause_threshold=pause_threshold,
            phrase_time_limit=phrase_time_limit,
            summary_length_limit=summary_length_limit,
        )
        # Initialize appropriate model class with defaults, if none provided
        self._intent_classifier = intent_classifier or UDIntentClassifier()
        self._sentiment_detector = sentiment_detector or UDSentimentDetector()
        self._keyword_extractor = keyword_extractor or UDKeyExtractor()

        self.process_interval = process_interval
        self.last_n_sentences = last_n_sentences
        self._shutdown_event = threading.Event()
        self._scheduler = sched.scheduler(time.time, time.sleep)

        # Initialize private attributes
        self._intent: Optional[str] = None
        self._sentiment: Optional[str] = None
        self._keywords: Optional[List[str]] = None

    def start_processing(self):
        """Starts the processing thread for intent, sentiment, and keyword extraction."""
        super().start_processing()

        # Ensure shutdown event is reset
        self._shutdown_event.clear()

        # Create a separate thread to handle the scheduled processing of transcriptions
        self._text_processing_thread = threading.Thread(target=self._scheduled_processing)

        # Start the newly created thread
        self._text_processing_thread.start()

    def _scheduled_processing(self):
        """Runs the scheduler to execute processing at a set interval."""
        # Prepare the first scheduled task
        self._schedule_next_run()

        # Start the scheduler to begin executing scheduled tasks
        self._scheduler.run()

    def _schedule_next_run(self):
        """Queues the processing task in the scheduler unless a shutdown has been signaled."""
        # Check if shutdown has not been signaled
        if not self._shutdown_event.is_set():
            # Schedule the processing task based on the predefined interval
            self._scheduler.enter(self.process_interval, 1, self._execute_processing)

    def _execute_processing(self):
        """Executes the processing of the current transcriptions and schedules the next run."""
        self._process_transcriptions()
        self._schedule_next_run()

    def _predict_intent(self, text: str) -> str:
        """Predicts the intent of the provided text."""
        return self._intent_classifier(text)[0]["label"]

    def _predict_sentiment(self, text: str) -> str:
        """Predicts the sentiment of the provided text."""
        recent_transcript = (
            ".".join(text.split(".")[-self.last_n_sentences :]) if self.last_n_sentences is not None else text
        )
        return self._sentiment_detector(recent_transcript).sentiment

    def _extract_keywords(self, text: str) -> List[str]:
        """Extracts keywords from the provided text."""
        return self._keyword_extractor(text)

    def _process_transcriptions(self):
        """Processes the transcription to extract keywords and perform intent and sentiment analysis."""
        # If there is no transcription, skip processing
        if len(self._transcription) == 0:
            return None

        # Predict intent and sentiment and extract keywords
        full_transcription = self._transcription
        self._intent = self._predict_intent(full_transcription)
        self._sentiment = self._predict_sentiment(full_transcription)
        self._keywords = self._extract_keywords(full_transcription)
        print(
            f"Intent: {self._intent}, Sentiment: {self._sentiment}, "
            f"Keywords: {' '.join(self._keywords) if self._keywords else None}"
        )

    def stop_processing(self):
        """Signals the processing thread to terminate and waits for it to finish."""
        super().stop_processing()

        # Signal the processing thread to shut down
        self._shutdown_event.set()

        if hasattr(self, "_text_processing_thread"):
            # Wait for the processing thread to finish its current task and terminate
            self._text_processing_thread.join()

    @property
    def intent(self) -> Optional[str]:
        """Returns the current intent if available."""
        return self._intent

    @property
    def sentiment(self) -> Optional[str]:
        """Returns the current sentiment if available."""
        return self._sentiment

    @property
    def keywords(self) -> Optional[List[str]]:
        """Returns the current keywords if available."""
        return self._keywords


if __name__ == "__main__":
    content_evaluator = ContentEvaluator()

    try:
        # Start microphone and processing
        content_evaluator.initialize_microphone()
        content_evaluator.start_processing()
        while True:  # Keep the main thread alive
            pass
    except KeyboardInterrupt:
        # This block will execute when Ctrl+C is pressed
        logger.warning("Received interrupt, stopping processing...")
    finally:
        # Stop processing and cleanup
        content_evaluator.stop_processing()
        logger.info(f"Full Audio Speech Transcription: {content_evaluator.transcription}")
        logger.info(f"Intent: {content_evaluator.intent}\nSentiment: {content_evaluator.sentiment}")
