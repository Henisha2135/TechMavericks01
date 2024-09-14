import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn  # Import uvicorn to run the FastAPI app

# Initialize FastAPI app
app = FastAPI()

# Load the tokenizer and pre-trained model once
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Labels for the predicted classes
labels = ['Easy', 'Medium', 'Tough']

class SentenceRequest(BaseModel):
    sentence: str

class SentencesResponse(BaseModel):
    highlighted_sentence: str

# Load CSV data once at startup
def load_sentences_from_csv(file_path: str) -> pd.DataFrame:
    """Reads sentences from a CSV file and returns them as a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        if 'Word' not in df.columns or 'Difficulty Level' not in df.columns:
            raise ValueError("The CSV file must contain 'Word' and 'Difficulty Level' columns.")
        return df
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Error: File '{file_path}' not found.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {e}")

# Load the CSV file during FastAPI startup
df = load_sentences_from_csv('words.csv')

def extract_keywords(sentence: str) -> set:
    """Extracts keywords from a sentence. Simple tokenization for now."""
    return set(sentence.lower().split())

def map_keywords_to_difficulty(keywords: set, df: pd.DataFrame) -> dict:
    """Maps keywords to difficulty levels based on the DataFrame."""
    difficulty_map = {}
    for keyword in keywords:
        match = df[df['Word'].str.lower() == keyword]
        if not match.empty:
            difficulty = match['Difficulty Level'].values[0]
            difficulty_map[keyword] = difficulty
    return difficulty_map

def highlight_words(sentence: str, difficulty_map: dict) -> str:
    """Highlights words in the sentence based on difficulty mapping."""
    words = sentence.split()
    highlighted_sentence = ' '.join(f"[{word}]({difficulty_map.get(word.lower(), 'Unknown')})" for word in words)
    return highlighted_sentence

@app.post("/highlight", response_model=SentencesResponse)
def highlight_sentence(request: SentenceRequest):
    """API endpoint to highlight words in the sentence based on difficulty."""
    sentence = request.sentence
    
    # Extract keywords and map to difficulties
    keywords = extract_keywords(sentence)
    difficulty_map = map_keywords_to_difficulty(keywords, df)
    
    if difficulty_map:
        # Highlight words and return the result
        highlighted_sentence = highlight_words(sentence, difficulty_map)
        return SentencesResponse(highlighted_sentence=highlighted_sentence)
    else:
        raise HTTPException(status_code=404, detail="No matching keywords found in the CSV file.")

# Run the FastAPI app with uvicorn  
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
