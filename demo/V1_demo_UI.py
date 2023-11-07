from typing import List

import gradio as gr
import torch
import whisper

from unified_desktop.pipelines import (
    UDSpeechEmotionRecognizer,
    UDSpeechRecognizer,
    UDSummarizer,
)

# from unified_desktop.tools.timers import timer

# For NLP task demonstration purposes


# logger = logging.Logger("Demo_logger")
# log_file = CACHE_DIR / "demo_v1.log"
# if not log_file.exists():
#     log_file.parent.mkdir(parents=True, exist_ok=True)
#     log_file.touch()
# logger.addHandler(logging.FileHandler(log_file))


CUDA_OPTIONS = [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
device_dropdown = gr.Dropdown(label="Device", choices=["cpu"] + CUDA_OPTIONS, value="cpu")


# def update_asr_obj(change):
#     global asrObj
#     clear_output()
#     display(model_dropdown_ASR, device_dropdown)
#     asrObj = UDSpeechRecognizer(name=model_dropdown_ASR.value, device=device_dropdown.value)
#     print(f"Loaded model: {model_dropdown_ASR.value} on device: {device_dropdown.value}")


# def update_ser_obj(change):
#     global serObj
#     serObj = UDSpeechEmotionRecognizer(device=device_dropdown.value)
#     clear_output()
#     display(device_dropdown)
#     print(f"Loaded SER model on device: {device_dropdown.value}")


# def update_intent_obj(change):
#     global intentObj
#     clear_output()
#     display(model_dropdown_intent, device_dropdown)
#     intentObj = UDIntentClassifier(name=model_dropdown_intent.value, device=device_dropdown.value)
#     print(f"Loaded model: {model_dropdown_intent.value} on device: {device_dropdown.value}")


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
        label="ASR Model Selection", choices=whisper.available_models(), value="tiny.en"
    )

    # model_dropdown_ASR.observe(update_asr_obj, names="value")
    asrObj = UDSpeechRecognizer(name=model_dropdown_ASR.value, device=device_dropdown.value)

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

    # Use fp16 if on CUDA, else fp32
    fp16 = device_dropdown.value in CUDA_OPTIONS

    return asrObj(audio, verbose=False, fp16=fp16)  # type: ignore


def demo_ser(audio: str) -> str:
    return serObj(audio).label  # type: ignore


def demo_intent_detection(text: str) -> str:
    return "Intent: Greeting"


def demo_keyword_extraction(text: str) -> List[str]:
    return ["mock", "keywords"]


def demo_summarization(text: str) -> str:
    summary = summarizer(text)  # type: ignore
    return summary


def demo_sentiment_analysis(text: str) -> str:
    return "Positive"


def NLP_task_processing(
    audio: str,
    intent_selected: List[str],
    keyword_selected: List[str],
    summary_selected: List[str],
    sentiment_selected: List[str],
) -> tuple[str, str, str, str, str, list[str]]:
    asr_result = demo_asr(audio)
    ser_result = demo_ser(audio)
    intents = demo_intent_detection(asr_result)
    keywords = demo_keyword_extraction(asr_result)
    summary = demo_summarization(asr_result)
    sentiment = demo_sentiment_analysis(asr_result)

    # summary = demo_summarization(asr_result) if "Summarization" in summary_selected else None
    # sentiment = demo_sentiment_analysis(asr_result) if "Sentiment Analysis" in sentiment_selected else None
    # intents = demo_intent_detection(asr_result) if "Intent Detection" in intent_selected else None
    # keywords = demo_keyword_extraction(asr_result) if "Keyword Extraction" in keyword_selected else None
    return asr_result, ser_result, intents, keywords, summary, sentiment  # type: ignore


def create_gradio_ui_elements():
    audio_input = gr.Audio(type="filepath", label="Upload an audio file")
    # summary_checkbox = gr.CheckboxGroup(["Summarization"], label=" ")
    # sentiment_checkbox = gr.CheckboxGroup(["Sentiment Analysis"], label=" ")
    # intent_checkbox = gr.CheckboxGroup(["Intent Detection"], label="Select Tasks")
    # keyword_checkbox = gr.CheckboxGroup(["Keyword Extraction"], label=" ")

    # inputs = [audio_input, summary_checkbox, sentiment_checkbox, intent_checkbox, keyword_checkbox]
    inputs = [audio_input]

    outputs = [
        gr.Textbox(label="ASR Result"),
        gr.Textbox(label="SER Result"),
        gr.Textbox(label="Intent Detection"),
        gr.Textbox(label="Keyword Extraction"),
        gr.Textbox(label="Summarization"),
        gr.Textbox(label="Sentiment Analysis"),
    ]

    return inputs, outputs


if __name__ == "__main__":
    demo_init()
    inputs, outputs = create_gradio_ui_elements()
    gr.Interface(NLP_task_processing, inputs, outputs).launch()
