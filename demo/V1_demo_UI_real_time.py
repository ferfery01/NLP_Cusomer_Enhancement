import gradio as gr

from unified_desktop.services import ContentEvaluator

# Initialize the content evaluator service with the defined interval
ud_service = ContentEvaluator(process_interval=30)

# Initialize the microphone for audio input
ud_service.initialize_microphone()


def display_transcript() -> str:
    """Fetch and display the current transcript."""
    transcript = ud_service.transcription
    return transcript if transcript else "None"


def display_sentiment() -> str:
    """Fetch and display the current sentiment analysis."""
    sentiment = ud_service.sentiment
    return sentiment if sentiment else "None"


def display_intent() -> str:
    """Fetch and display the current intent detection."""
    intent = ud_service.intent
    return intent if intent else "None"


def display_keywords() -> str:
    """Fetch and display the current keywords extracted."""
    keywords = ud_service.keywords
    return ", ".join(keywords) if keywords else "None"


# Define and setup the Gradio UI components
with gr.Blocks(theme=gr.themes.Default()) as demo:
    # Define a row for the record and stop buttons
    with gr.Row():
        record_btn = gr.Button("Record", variant="primary")
        stop_btn = gr.Button("Stop", variant="primary")

    # Define a column for displaying the results of content analysis
    with gr.Column():
        transcript = gr.Textbox(label="Transcript", value=display_transcript, every=1, interactive=False)
        sentiment = gr.Textbox(label="Sentiment", value=display_sentiment, every=5, interactive=False)
        intent = gr.Textbox(label="Intent", value=display_intent, every=5, interactive=False)
        keywords = gr.Textbox(label="Keywords", value=display_keywords, every=5, interactive=False)
        summary = gr.Textbox(label="Summary", interactive=False)
        summary_btn = gr.Button("Generate Summary")

    # Define the actions for different buttons
    @record_btn.click()
    def start_recording() -> None:
        """Start the content analysis processing."""
        ud_service.start_processing()

    @stop_btn.click()
    def stop_recording() -> None:
        """Stop the content analysis processing."""
        ud_service.stop_processing()

    @summary_btn.click(inputs=[], outputs=summary)
    def display_summary() -> str:
        """Fetch and display the current summary."""
        return ud_service.summary


# Launch the Gradio interface when the script is run
if __name__ == "__main__":
    # Use queue to handle possibly asynchronous updates safely
    demo.queue().launch()
