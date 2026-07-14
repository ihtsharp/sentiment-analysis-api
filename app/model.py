from transformers import pipeline

print("Loading DistilBERT model...")

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print(classifier("I love this product"))
print(classifier("Worst purchase ever"))
print(classifier("The guitar is okay"))

print("Model Loaded Successfully!")

text = "This guitar sounds amazing."

result = classifier(text)

print(result)