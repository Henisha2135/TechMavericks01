import cv2
import pytesseract
import pyttsx3
from PIL import Image
from TTS.api import TTS
import soundfile as sf
import simpleaudio as sa
import speech_recognition as sr
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Configure the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Initialize the TTS model
model_name = "tts_models/en/ljspeech/tacotron2-DDC_ph"  # Change this if needed
tts = TTS(model_name)

# Initialize the TTS engine for pyttsx3 (Not used in Colab)
# engine = pyttsx3.init()
# engine.setProperty('rate', 150)  # Speed of speech (words per minute)
# engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

# Load the BERT tokenizer and model for classification
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
labels = ['Easy', 'Medium', 'Tough']

def perform_ocr(image):
    """
    Perform OCR on the given image and return the extracted text.
    """
    pil_image = Image.fromarray(image)
    text = pytesseract.image_to_string(pil_image, lang='eng')
    return text.strip()

def text_to_speech(text):
    """
    Convert text to speech using Coqui TTS model.
    """
    audio = tts.tts(text)
    sf.write('output.wav', audio, 22050)  # Save audio with a sample rate of 22050 Hz
    return 'output.wav'

def play_audio(file_path):
    """
    Play the generated audio file.
    """
    wave_obj = sa.WaveObject.from_wave_file(file_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

def speech_to_text():
    """
    Converts live speech to text using Google's Speech Recognition API.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        
        while True:
            try:
                audio = recognizer.listen(source)
                text = recognizer.recognize_google(audio)
                print("You said:", text)
                return text  # Return recognized text
            except sr.UnknownValueError:
                print("Sorry, I did not understand that.")
            except sr.RequestError as e:
                print(f"Sorry, there was an error with the request; {e}")
                break

def speak_text(text):
    """
    Convert text to speech using pyttsx3.
    """
    # Use Coqui TTS instead of pyttsx3 in Colab
    audio_path = text_to_speech(text)
    play_audio(audio_path)

def load_sentences_from_csv(file_path):
    """Reads sentences from a CSV file and returns them as a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        if 'Word' not in df.columns or 'Difficulty Level' not in df.columns:
            raise ValueError("The CSV file must contain 'Word' and 'Difficulty Level' columns.")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def classify_sentence(sentence):
    """Classifies a sentence using DistilBERT."""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return labels[predictions.item()]

def extract_keywords(sentence):
    """Extracts keywords from a sentence. Simple tokenization for now."""
    return set(sentence.lower().split())

def map_keywords_to_difficulty(keywords, df):
    """Maps keywords to difficulty levels based on the DataFrame."""
    difficulty_map = {}
    for keyword in keywords:
        match = df[df['Word'].str.lower() == keyword]
        if not match.empty:
            difficulty = match['Difficulty Level'].values[0]
            difficulty_map[keyword] = difficulty
    return difficulty_map

def highlight_words(sentence, difficulty_map):
    """Highlights words in the sentence based on difficulty mapping."""
    words = sentence.split()
    highlighted_sentence = ' '.join(f"[{word}]({difficulty_map.get(word.lower(), 'Unknown')})" for word in words)
    return highlighted_sentence

def main():
    print("Live OCR and Speech-to-Text Application")
    print("Press 'q' to quit.")

    # Load sentences from the CSV file
    file_path = '/content/words.csv'  # Ensure this file exists in the correct location
    df = load_sentences_from_csv(file_path)
    
    if df is None:
        print("No sentences to process. Exiting.")
        return

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Perform OCR on the captured frame
        text = perform_ocr(frame)
        
        # Display the captured frame
        cv2.imshow('Live OCR', frame)
        
        # Overlay the extracted text on the video feed
        if text:
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # Convert text to speech and play the audio
            audio_path = text_to_speech(text)
            play_audio(audio_path)
            
            # Classify the difficulty level of the extracted text
            classification = classify_sentence(text)
            print(f"Text difficulty classification: {classification}")

            # Extract keywords and map to difficulties
            keywords = extract_keywords(text)
            difficulty_map = map_keywords_to_difficulty(keywords, df)
            
            if difficulty_map:
                # Highlight words and display the result
                highlighted_sentence = highlight_words(text, difficulty_map)
                print(f"Highlighted Sentence: {highlighted_sentence}")
            else:
                print("No matching keywords found in the CSV file.")
        
        # Handle speech-to-text in a separate thread
        if cv2.waitKey(1) & 0xFF == ord('s'):
            print("Listening for speech...")
            spoken_text = speech_to_text()
            if spoken_text:
                # Convert spoken text to speech and play it
                speak_text(spoken_text)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if _name_ == "_main_":
    main()