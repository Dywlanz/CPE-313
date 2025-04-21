import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- BERT Sentiment Analysis Model ---

# Use Streamlit's caching for efficiency
@st.cache_resource # Cache the model and tokenizer loading
def load_model():
    """Loads the pre-trained BERT model and tokenizer."""
    print("Loading BERT model and tokenizer...") # Added print statement for clarity
    try:
        tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        print("Model loaded successfully.") # Confirmation message
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}") # Also print error to console
        return None, None

tokenizer, model = load_model()

def get_sentiment(text):
    """Analyzes sentiment using the loaded BERT model."""
    if not model or not tokenizer:
        st.error("Model or tokenizer not loaded. Cannot perform analysis.")
        return None, None
    if not text:
        return None, None # Return None if no text is provided

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512) # Added truncation
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # The model predicts classes 0-4, corresponding to 1-5 stars.
            sentiment_score = torch.argmax(probabilities, dim=-1).item() + 1
            return sentiment_score, probabilities.numpy().tolist()[0] # Return score and probabilities list
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        return None, None

# --- Streamlit UI ---
st.title("Simple Sentiment Analysis")
st.write("Enter text below to analyze its sentiment (scored 1-5 stars).")

# Check if the model loaded correctly before proceeding
if model and tokenizer:
    text_input = st.text_area("Text to analyze:", height=100)

    if st.button("Analyze Sentiment"):
        if text_input:
            score, probabilities = get_sentiment(text_input)
            if score is not None:
                st.metric(label="Sentiment Score (Stars)", value=f"{score} / 5")
                st.write("Probabilities per score (1-5):")
                # Create a dictionary for the bar chart {1: prob1, 2: prob2, ...}
                prob_dict = {i+1: prob for i, prob in enumerate(probabilities)}
                st.bar_chart(prob_dict)
            else:
                # Error messages are handled within get_sentiment
                st.warning("Could not analyze sentiment. See error message above if applicable.")
        else:
            st.warning("Please enter some text.")
else:
    st.error("Application cannot start because the sentiment analysis model failed to load.")
