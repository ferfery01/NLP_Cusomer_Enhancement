from typing import List, Tuple

import gradio as gr
import numpy as np
import torch

from unified_desktop.pipelines import (
    UDIntentClassifier,
    UDKeyExtractor2,
    UDSentimentDetector,
    UDSpeechEmotionRecognizer,
    UDSpeechRecognizer,
    UDSummarizer,
)

CUDA_OPTIONS = [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
device_dropdown = gr.Dropdown(label="Device", choices=["cpu"] + CUDA_OPTIONS, value="cpu")

# Ud_Audio to upload for the demo:
# https://walgreens-my.sharepoint.com/:f:/p/zeinab_takbiri/EpKsj2-WSwFLhApq7pHi1Q0BwDXMT-CizP50h3gpqD6WHA?e=tHr4rc


def demo_init():
    """
    Demo initialization steps:
        1. Initialize objects
    """
    # open(log_file, "w").close()
    # logger.info("Initialization started...")

    # Set device:

    # Initialize ASR object
    global asrObj
    asrObj = None

    # Gradio components for model selection and device selection
    model_dropdown_ASR = gr.Dropdown(
        label="ASR Model Selection", choices=UDSpeechRecognizer.models_list, value="openai/whisper-tiny.en"
    )

    # model_dropdown_ASR.observe(update_asr_obj, names="value")
    asrObj = UDSpeechRecognizer(model_id=model_dropdown_ASR.value, device=device_dropdown.value)

    # initiate SER object
    global serObj
    # Attach the update function to the dropdown
    # device_dropdown.observe(update_ser_obj, names="value")
    serObj = UDSpeechEmotionRecognizer(device=device_dropdown.value)

    # intiate summarization object
    global summarizer
    summarizer = UDSummarizer(device=device_dropdown.value)


def demo_asr(audio: str) -> str:
    # Transcribe the audio file to text
    return asrObj(audio)  # type: ignore


def demo_ser(audio: str) -> str:
    return serObj(audio).label  # type: ignore


def demo_intent_detection(text: str) -> List[Tuple[str, float]]:
    # Input text for keywords extraction
    top_k = 3
    model_intent = "vineetsharma/customer-support-intent-albert"
    intentObj = UDIntentClassifier(model_id=model_intent, device=device_dropdown.value)
    intent_results = intentObj(text, top_k)
    list_intent = []
    for item in intent_results:
        list_intent.append((item["label"], np.round(item["score"], 3)))
    return list_intent


def demo_keyword_extraction(text: str) -> List[str]:
    model_key = "transformer3/H2-keywordextractor"
    KeyObj = UDKeyExtractor2(model_id=model_key, device=device_dropdown.value)
    return KeyObj(text)


def demo_summarization(text: str) -> str:
    summary = summarizer(text)  # type: ignore
    return summary


def demo_sentiment_analysis(text: str) -> str:
    sentiment_obj = UDSentimentDetector()
    output_sentiment = sentiment_obj(input_text=text)
    return output_sentiment.sentiment


def NLP_task_processing(
    audio: str,
) -> tuple[str, str, List[Tuple[str, float]], list[str], str, str]:
    asr_result = demo_asr(audio)
    ser_result = demo_ser(audio)
    summary = demo_summarization(asr_result)
    intents = demo_intent_detection(summary)
    keywords = demo_keyword_extraction(asr_result)
    sentiment = demo_sentiment_analysis(asr_result)
    return asr_result, ser_result, intents, keywords, summary, sentiment  # type: ignore


def create_gradio_ui_elements():
    audio_input = gr.Audio(type="filepath", label="Upload an audio file")
    inputs = [audio_input]

    outputs = [
        gr.Textbox(label="ASR Result"),
        gr.Textbox(label="SER Result"),
        gr.Textbox(label="Top 3 Intent Detection"),
        gr.Textbox(label="Keyword Extraction"),
        gr.Textbox(label="Summarization"),
        gr.Textbox(label="Sentiment Analysis"),
    ]
    return inputs, outputs


if __name__ == "__main__":
    demo_init()
    inputs, outputs = create_gradio_ui_elements()
    gr.Interface(NLP_task_processing, inputs, outputs).launch()
