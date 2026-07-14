from fastapi import FastAPI,UploadFile, File
from pydantic import BaseModel
from transformers import pipeline
import pandas as pd
import torch

torch.set_num_threads(1)
# Create FastAPI app
app = FastAPI(title="Sentiment Analysis API")

print("Loading DistilBERT model...")

classifier = pipeline(
    "sentiment-analysis",
   sentiment_model = pipeline(
    "sentiment-analysis",
    model="finiteautomata/bertweet-base-sentiment-analysis"
)
)

print("Model Loaded Successfully!")


# Request model
class Review(BaseModel):
    text: str


# Home endpoint
@app.get("/")
def home():
    return {"message": "Welcome to Sentiment Analysis API"}


# Prediction endpoint
@app.post("/predict")
def predict(review: Review):

    result = classifier(review.text)

    return {
        "review": review.text,
        "sentiment": result[0]["label"],
        "confidence": round(result[0]["score"] * 100, 2)
    }

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):

    df = pd.read_csv(file.file)

    sentiments = []
    confidences = []

    for review in df["reviewText"]:
        result = classifier(str(review))
        sentiments.append(result[0]["label"])
        confidences.append(round(result[0]["score"] * 100, 2))

    df["Sentiment"] = sentiments
    df["Confidence"] = confidences

    output_file = "data/predicted_reviews.csv"
    df.to_csv(output_file, index=False)

    return {
        "message": "Prediction Completed",
        "total_reviews": len(df),
        "output_file": output_file
    }

@app.get("/stats")
def stats():

    df = pd.read_csv("data/predicted_reviews.csv")

    return {
        "Total Reviews": len(df),
        "Positive": int((df["Sentiment"] == "POSITIVE").sum()),
        "Negative": int((df["Sentiment"] == "NEGATIVE").sum()),
        "Average Rating": round(df["overall"].mean(), 2)
    }