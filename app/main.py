from fastapi import FastAPI,UploadFile, File
from pydantic import BaseModel
from transformers import pipeline
import pandas as pd
import torch
import os
torch.set_num_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Create FastAPI app
app = FastAPI(title="Sentiment Analysis API")
import os
import torch

torch.set_num_threads(1)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


classifier = None


def get_model():
    global classifier

    if classifier is None:
        print("STEP 1")

        from transformers import pipeline
        print("STEP 2")

        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )

        print("STEP 3")

    return classifier

# Request model
class Review(BaseModel):
    text: str


# Home endpoint
@app.get("/")
def home():
    return {"message": "Welcome to Sentiment Analysis API"}


# Prediction endpoint
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):

    model = get_model()

    df = pd.read_csv(file.file)

    # Check that the CSV has the required column
    if "reviewText" not in df.columns:
        return {"error": "CSV must contain a 'reviewText' column"}

    sentiments = []
    confidences = []

    for review in df["reviewText"]:
        result = model(
            str(review),
            truncation=True,
            max_length=512
        )

        sentiments.append(result[0]["label"])
        confidences.append(round(result[0]["score"] * 100, 2))

    # ADD THESE TWO LINES
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
@app.get("/stats")
def stats():

    df = pd.read_csv("data/predicted_reviews.csv")

    return {
        "Total Reviews": len(df),
        "Positive": int((df["Sentiment"] == "POSITIVE").sum()),
        "Negative": int((df["Sentiment"] == "NEGATIVE").sum()),
        "Average Rating": round(df["overall"].mean(), 2)
    }