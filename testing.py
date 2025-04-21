# Combined Python script from BERT_testing 3.pdf and testing 2.pdf

# Required packages from BERT_testing 3.pdf:
# pip install torch transformers requests beautifulsoup4 pandas numpy

# Required packages from testing 2.pdf:
# pip install speechrecognition pyttsx3 pydub

# --- Imports (Combined and Deduplicated) ---
import torch
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd # Assuming pandas might be needed based on installs, though not used in snippets
import numpy as np # Assuming numpy might be needed based on installs, though not used in snippets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import speech_recognition as sr
import pyttsx3
import csv
from pydub import AudioSegment
import os
import time # Added for saving audio files with unique names

# --- Code from BERT_testing 3.pdf ---

print(f"PyTorch version: {torch.__version__}") # Note: Corrected from _version_

# Instantiate Model
# Note: Handling potential line breaks in the original PDF extraction
tokenizer_bert = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment') # Renamed to avoid conflict
model_bert = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment') # Renamed

# Function to Encode and Calculate Sentiment (derived from In[4], In[5], In[7], In[8])
def get_bert_sentiment(text):
    """
    Analyzes the sentiment of a given text using a pre-trained BERT model.

    Args:
        text (str): The input text to analyze.

    Returns:
        int: The predicted sentiment score (1-5), or None if an error occurs.
             Returns probabilities as well for inspection.
        torch.Tensor: Probabilities for each class (1 to 5 stars).
    """
    try:
        tokens = tokenizer_bert.encode(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model_bert(tokens)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_index = torch.argmax(probabilities, dim=-1).item()
            sentiment_score = predicted_class_index + 1 # Scores are 1-5, index is 0-4
            print(f"Input text: {text}")
            print(f"Probabilities: {probabilities}")
            print(f"Predicted class index: {predicted_class_index}, Sentiment score: {sentiment_score}")
            return sentiment_score, probabilities
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return None, None

# Example Usage for Sentiment Analysis (derived from In[4], In[5])
example_text = "It was good but couldve been better. Great"
sentiment, probabilities = get_bert_sentiment(example_text)
if sentiment is not None:
    print(f"\nExample Sentiment Score for '{example_text}': {sentiment}")
    print(f"Example Probabilities: {probabilities}")

# Collect Reviews (derived from In[10])
def collect_yelp_reviews(url='https://www.yelp.com/biz/mejico-sydney-2'):
    """
    Collects reviews from a Yelp page.

    Args:
        url (str): The URL of the Yelp business page.

    Returns:
        list: A list of review texts, or empty list if an error occurs.
    """
    reviews = []
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}) # Added User-Agent
        r.raise_for_status() # Check if the request was successful
        soup = BeautifulSoup(r.text, 'html.parser')
        # The regex might need adjustment based on current Yelp structure
        # Trying a more specific selector based on common Yelp review structures
        # This is fragile and likely to break if Yelp changes its HTML.
        # Found potential class names from the HTML snippet provided in the PDF:
        # Look for spans within paragraphs with class containing 'comment'
        regex = re.compile('.*comment.*')
        # A more robust approach might target specific data attributes if available
        # Example using a hypothetical structure found in the HTML dump:
        # results = soup.select('div[data-testid="review-container"] p span.raw__09f24__T4Ezm') # Example selector
        # Using the original regex approach as direct selectors are hard to guarantee from the dump:
        results = soup.find_all('p', {'class': regex}) # Original approach from PDF
        reviews = [result.text for result in results]

        # If the above yields no results, try another potential selector (less specific)
        if not reviews:
             review_spans = soup.select('span[lang="en"]') # General selector for English text spans
             reviews = [span.text for span in review_spans if len(span.text) > 50] # Filter short spans

        print(f"Found {len(reviews)} potential reviews.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Yelp page: {e}")
    except Exception as e:
        print(f"Error parsing Yelp page: {e}")
    return reviews

# Example Usage for Review Collection and Analysis
yelp_url = 'https://www.yelp.com/biz/mejico-sydney-2' # As used in In[10]
print(f"\nCollecting reviews from {yelp_url}...")
reviews = collect_yelp_reviews(yelp_url)

if reviews:
    print("\nAnalyzing sentiments of collected reviews:")
    # Analyze the first few reviews as an example
    for i, review in enumerate(reviews[:3]): # Analyze first 3 reviews
        print(f"\nReview {i+1}: {review[:100]}...") # Print snippet
        sentiment, _ = get_bert_sentiment(review)
        if sentiment:
            print(f"Sentiment Score: {sentiment}")
else:
    print("No reviews collected or error occurred.")

# --- Code from testing 2.pdf ---

# Initialize recognizer and text-to-speech engine
r = sr.Recognizer()
engine = pyttsx3.init()

# Function to record audio, recognize speech, and save audio
def record_text():
    """
    Records audio from the microphone, performs speech recognition,
    saves the audio to WAV and MP3 files, and returns the recognized text.

    Returns:
        str: The recognized text, or None if an error occurred or audio was not understood.
    """
    while True:
        try:
            # use the microphone as source for input.
            with sr.Microphone() as source:
                print("Adjusting for ambient noise...")
                r.adjust_for_ambient_noise(source, duration=0.5) # Reduced duration slightly
                print("Listening... (say 'stop' to exit)")
                # listens for the user's input
                audio = r.listen(source) # Removed timeout and phrase_time_limit for continuous listening until pause

                # Use Google Speech Recognition
                print("Recognizing...")
                MyText = r.recognize_google(audio, language="en-US")

                # Save the recorded audio to a WAV file with timestamp
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                wav_filename = f"output_{timestamp}.wav"
                with open(wav_filename, "wb") as f_wav:
                    f_wav.write(audio.get_wav_data())
                print(f"Saved WAV: {wav_filename}")

                # Convert WAV to MP3
                try:
                    mp3_filename = f"output_{timestamp}.mp3"
                    sound = AudioSegment.from_wav(wav_filename)
                    sound.export(mp3_filename, format="mp3")
                    print(f"Saved MP3: {mp3_filename}")
                    os.remove(wav_filename) # Remove the temporary WAV file
                except Exception as e:
                    print(f"Could not convert WAV to MP3: {e}")
                    # Keep WAV if MP3 conversion fails

                return MyText

        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None # Indicate error

        except sr.UnknownValueError:
            print("Could not understand audio, please try again.")
            # Continue loop to listen again
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None # Indicate error

# Function to output text to files
def output_text_to_file(text):
    """
    Appends the given text to output.txt and output.csv files.

    Args:
        text (str): The text to save.
    """
    if text: # Only save if text is not None or empty
        try:
            # Save text to a plain text file
            with open("output.txt", "a") as f:
                f.write(text + "\n")

            # Save text to a CSV file
            with open("output.csv", "a", newline="", encoding='utf-8') as csvfile: # Added encoding
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([text])
            print("Wrote text to files.")
        except Exception as e:
            print(f"Error writing text to file: {e}")
    else:
        print("No text recognized to save.")


# --- Main Execution Logic (Example) ---
if __name__ == "__main__":
    # Example: Run the speech recognition part (comment out if not needed)
    print("\nStarting speech recognition loop...")
    while True:
        text_recognized = record_text()
        if text_recognized:
            print(f"Recognized: {text_recognized}")
            if text_recognized.lower() == "stop":
                print("Stopping the program...")
                break
            output_text_to_file(text_recognized)
        elif text_recognized is None:
             # Handle potential errors from record_text if needed, or just retry
             print("Retrying...")
             time.sleep(1) # Small delay before retrying

    # The BERT sentiment analysis parts were already run as examples above.
    # If you want to integrate them further (e.g., analyze recognized speech),
    # you would call get_bert_sentiment(text_recognized) within the loop.
    print("\nScript finished.")
