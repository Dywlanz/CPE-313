import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from transformers import AutoTokenizer, DistilBertForSequenceClassification

# --- Configuration (MUST EDIT THESE) ---
# Replace with the actual base model name you fine-tuned from (e.g., 'bert-base-uncased', 'roberta-base')
BASE_MODEL_NAME = 'distilbert-base-uncased'
# Path to your fine-tuned model weights file
MODEL_PATH = 'fine_tuned_model.pth'
# Adjust this to the number of output classes your model predicts (e.g., 2 for binary, 5 for ratings)
NUM_LABELS = 3
# --- End Configuration ---

# --- Model and Tokenizer Loading ---
@st.cache_resource # Cache loading
def load_model_and_tokenizer(base_model_name, model_path, num_labels):
    """Loads the tokenizer and model architecture, then loads fine-tuned weights."""
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at {model_path}")
        return None, None

    try:
        # Load tokenizer associated with the base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Load the model architecture with the specified number of labels
        # We initialize with the base model config but specify the number of labels
        # from our fine-tuned model.
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels
        )

        # Load the fine-tuned weights from the .pth file
        # Important: Ensure the state_dict keys match the model architecture.
        # This assumes the .pth file contains a state_dict compatible with
        # AutoModelForSequenceClassification from Hugging Face.
        state_dict = torch.load(model_path, map_location=torch.device('cpu')) # Load to CPU

        # Handle potential differences in state_dict keys (e.g., extra "module." prefix)
        # Common issue if saved from DataParallel or DistributedDataParallel
        if all(key.startswith('module.') for key in state_dict.keys()):
             state_dict = {k.partition('module.')[2]: v for k,v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.eval() # Set model to evaluation mode
        st.success(f"Model loaded successfully from {model_path}")
        return tokenizer, model

    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        st.error(f"Check if BASE_MODEL_NAME ('{base_model_name}'), MODEL_PATH ('{model_path}'), and NUM_LABELS ({num_labels}) are correct.")
        return None, None

tokenizer, model = load_model_and_tokenizer(BASE_MODEL_NAME, MODEL_PATH, NUM_LABELS)

# --- Prediction Function ---
def predict(text):
    """Tokenizes text and uses the loaded model for prediction."""
    if not tokenizer or not model:
        st.error("Tokenizer or model not available for prediction.")
        return None, None

    if not text:
        return None, None

    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class_id = torch.argmax(probabilities, dim=-1).item()
            return predicted_class_id, probabilities.numpy().tolist()[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# --- Streamlit UI ---
st.title("Deploy Fine-Tuned Model")

if tokenizer and model:
    st.write(f"Using model based on: `{BASE_MODEL_NAME}`")
    st.write(f"Loaded weights from: `{MODEL_PATH}`")
    st.write(f"Predicting {NUM_LABELS} classes.")

    text_input = st.text_area("Enter text to classify:", height=100)

    if st.button("Classify"):
        if text_input:
            predicted_id, probs = predict(text_input)

            if predicted_id is not None:
                st.subheader("Prediction Results")
                st.write(f"**Predicted Class ID:** {predicted_id}")
                st.write("**Probabilities per Class:**")
                # Create a dictionary for the bar chart {Class 0: prob0, Class 1: prob1, ...}
                prob_dict = {f"Class {i}": prob for i, prob in enumerate(probs)}
                st.bar_chart(prob_dict)
            else:
                st.warning("Could not get prediction.")
        else:
            st.warning("Please enter text to classify.")
else:
    st.error("Application cannot start. Failed to load model or tokenizer. Please check configuration and file path.")
