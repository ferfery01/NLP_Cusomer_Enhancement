from pprint import pprint

import ipywidgets as widgets
import torch
import whisper
from IPython.display import clear_output, display

from unified_desktop import RESOURCES_DIR
from unified_desktop.pipelines import UDIntentClassification



# Input text for intent detection
input_text = RESOURCES_DIR/"input_text"/"sample.txt"
top_k = 5

# Instantiate the intent detection model
intent_detector = UDIntentClassification()

# Perform intent detection
top_kintent_results = intent_detector(input_text)

# Print the top-5 predictions and their probabilities for the intent detection
print("Top-5 Intent Predictions:")
print("------------------------------------------------------")
print("| Intent                    | Probability")
print("------------------------------------------------------")
for intent, probability in intent_results[:top_k]:
    print(f"| {intent:<25} | {probability:.4f}")
print("------------------------------------------------------")