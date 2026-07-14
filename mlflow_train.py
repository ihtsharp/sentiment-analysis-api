import mlflow
import mlflow.transformers
import matplotlib.pyplot as plt
from transformers import pipeline
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Set experiment
mlflow.set_experiment("Sentiment Analysis")

print("Loading DistilBERT model...")

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512
)


print("Model Loaded!")

df = pd.read_csv("data/preprocess.csv")
# Remove Neutral reviews because DistilBERT SST-2 is binary
df = df[df["sentiment"] != "Neutral"]


df = df.head(500)

predictions = []

print("Running predictions...")
for review in df["clean_review"]:
    result = classifier(
    str(review),
    truncation=True,
    max_length=512
)[0]["label"]
    predictions.append(result)
print("\nUnique values in sentiment column:")
print(df["sentiment"].unique())

print("\nValue counts:")
print(df["sentiment"].value_counts(dropna=False))

label_map = {
    "Positive": "POSITIVE",
    "Negative": "NEGATIVE"
}

true_labels = df["sentiment"].map(label_map)

print("\nMapped labels:")
print(true_labels.head(20))
print(true_labels.isna().sum(), "missing labels after mapping")
print(true_labels.isna().sum())
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(
    true_labels,
    predictions,
    pos_label="POSITIVE"
)

recall = recall_score(
    true_labels,
    predictions,
    pos_label="POSITIVE"
)

f1 = f1_score(
    true_labels,
    predictions,
    pos_label="POSITIVE"
)

cm = confusion_matrix(true_labels, predictions)

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation="nearest")
plt.colorbar()

plt.xticks([0,1], ["NEGATIVE","POSITIVE"])
plt.yticks([0,1], ["NEGATIVE","POSITIVE"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.savefig("confusion_matrix.png")
plt.close()

report = classification_report(true_labels, predictions)

print(f"Accuracy: {accuracy:.4f}")

with mlflow.start_run():
    mlflow.log_param("Model", "DistilBERT")
    mlflow.log_param("Dataset", "Amazon Reviews")
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1 Score", f1)
    model_info = mlflow.transformers.log_model(
        transformers_model=classifier,
        artifact_path="model",
        registered_model_name="sentiment-analysis-model",
        pip_requirements=[
            "torch",
            "transformers",
            "torchvision",
            "pandas",
            "numpy"
        ]
    )

    mlflow.log_artifact("confusion_matrix.png")

    with open("classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")
  
print("Registered!")
