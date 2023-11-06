# import random
# from pprint import pprint
from typing import List, Optional, Tuple

import gradio as gr

# import ipywidgets as widgets
# import pandas as pd
# import torch
# import whisper
# from IPython.display import Audio, clear_output, display
# from transformers import logging

# from unified_desktop import RESOURCES_DIR
# from unified_desktop.core.utils.io_utils import get_matching_files_in_dir
# from unified_desktop.pipelines import (
#     UDIntentClassifier,
#     UDKeyExtraction,
#     UDSentimentDetector,
#     UDSpeechEmotionRecognizer,
#     UDSpeechRecognizer,
#     UDSummarizer,
# )

# For NLP task demonstration purposes


def demo_asr(audio: bytes) -> str:
    return "This is a mock ASR result from the audio."


def demo_ser(audio: bytes) -> str:
    return "Happy"


def demo_intent_detection(text: str) -> str:
    return "Intent: Greeting"


def demo_keyword_extraction(text: str) -> List[str]:
    return ["mock", "keywords"]


def demo_summarization(text: str) -> str:
    return "This is a mock summary of the text."


def demo_sentiment_analysis(text: str) -> str:
    return "Positive"


def NLP_task_processing(
    audio: bytes,
    intent_selected: List[str],
    keyword_selected: List[str],
    summary_selected: List[str],
    sentiment_selected: List[str],
) -> Tuple[str, str, Optional[str], Optional[str], Optional[str], Optional[List[str]]]:
    asr_result = demo_asr(audio)
    ser_result = demo_ser(audio)
    summary = demo_summarization(asr_result) if "Summarization" in summary_selected else None
    sentiment = demo_sentiment_analysis(asr_result) if "Sentiment Analysis" in sentiment_selected else None
    intents = demo_intent_detection(asr_result) if "Intent Detection" in intent_selected else None
    keywords = demo_keyword_extraction(asr_result) if "Keyword Extraction" in keyword_selected else None
    return asr_result, ser_result, sentiment, summary, intents, keywords


def create_gradio_ui_elements():
    audio_input = gr.Audio(label="Upload an audio file")
    summary_checkbox = gr.CheckboxGroup(["Summarization"], label=" ")
    sentiment_checkbox = gr.CheckboxGroup(["Sentiment Analysis"], label=" ")
    intent_checkbox = gr.CheckboxGroup(["Intent Detection"], label="Select Tasks")
    keyword_checkbox = gr.CheckboxGroup(["Keyword Extraction"], label=" ")

    inputs = [audio_input, summary_checkbox, sentiment_checkbox, intent_checkbox, keyword_checkbox]

    outputs = [
        gr.Textbox(label="ASR Result"),
        gr.Textbox(label="SER Result"),
        gr.Textbox(label="Summarization"),
        gr.Textbox(label="Sentiment Analysis"),
        gr.Textbox(label="Intent Detection"),
        gr.Textbox(label="Keyword Extraction"),
    ]

    return inputs, outputs


if __name__ == "__main__":
    inputs, outputs = create_gradio_ui_elements()

    gr.Interface(NLP_task_processing, inputs, outputs).launch()
