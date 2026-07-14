import pandas as pd

# Load dataset
df = pd.read_csv("data/data.csv")

print("Original Shape:", df.shape)

# Keep only required columns
df = df[["reviewText", "overall"]]

# Remove rows where review text is missing
df = df.dropna(subset=["reviewText"])

print("After removing missing reviews:", df.shape)


# Convert ratings into sentiment labels
def sentiment_label(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"


df["sentiment"] = df["overall"].apply(sentiment_label)

print(df.head())

# Save cleaned dataset
df.to_csv("data/clean.csv", index=False)

print("\nCleaned dataset saved successfully!")