from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import pandas as pd
import torch
import os

# Performance settings
torch.set_num_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create FastAPI app
app = FastAPI(title="Sentiment Analysis API")

# Global model
classifier = None


# Load model when application starts
@app.on_event("startup")
def load_model():
    global classifier

    print("Loading DistilBERT model...")

    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )

    print("Model loaded successfully!")


# Request Model
class Review(BaseModel):
    text: str


# Home Endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to Sentiment Analysis API"
    }


# Predict Single Review
@app.post("/predict")
def predict(review: Review):

    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    result = classifier(
        review.text,
        truncation=True,
        max_length=512
    )

    return {
        "review": review.text,
        "sentiment": result[0]["label"],
        "confidence": round(result[0]["score"] * 100, 2)
    }


# Predict CSV
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):

    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    df = pd.read_csv(file.file)

    if "reviewText" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain a 'reviewText' column."
        )

    sentiments = []
    confidences = []

    for review in df["reviewText"]:

        result = classifier(
            str(review),
            truncation=True,
            max_length=512
        )

        sentiments.append(result[0]["label"])
        confidences.append(round(result[0]["score"] * 100, 2))

    df["Sentiment"] = sentiments
    df["Confidence"] = confidences

    os.makedirs("data", exist_ok=True)

    output_file = "data/predicted_reviews.csv"
    df.to_csv(output_file, index=False)

    return {
        "message": "Prediction Completed",
        "total_reviews": len(df),
        "output_file": output_file
    }


# Dataset Statistics
@app.get("/stats")
def stats():

    output_file = "data/predicted_reviews.csv"

    if not os.path.exists(output_file):
        raise HTTPException(
            status_code=404,
            detail="No prediction file found. Upload a CSV first."
        )

    df = pd.read_csv(output_file)

    response = {
        "Total Reviews": len(df),
        "Positive": int((df["Sentiment"] == "POSITIVE").sum()),
        "Negative": int((df["Sentiment"] == "NEGATIVE").sum())
    }

    if "overall" in df.columns:
        response["Average Rating"] = round(df["overall"].mean(), 2)

    return response