import pandas as pd
import re

# Load cleaned dataset
df = pd.read_csv("data/preprocess.csv")


# Function to clean text
def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply cleaning
df["clean_review"] = df["reviewText"].apply(clean_text)

print(df[["reviewText", "clean_review"]].head())

# Save cleaned data
df.to_csv("data/preprocess.csv", index=False)

print("\nPreprocessing Completed!")