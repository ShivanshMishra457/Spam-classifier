"""
Spam Email Classifier
Author: Shivansh
Description: Naive Bayes classifier trained on the UCI SMS Spam Collection dataset.
"""

import os
import urllib.request
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = "data"
MODEL_DIR = "model"


def download_dataset():
    """Download the UCI SMS Spam Collection dataset if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "smsspamcollection.zip")
    tsv_path = os.path.join(DATA_DIR, "SMSSpamCollection")

    if not os.path.exists(tsv_path):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)
        os.remove(zip_path)
        print("Dataset downloaded.")
    else:
        print("Dataset already exists, skipping download.")

    return tsv_path


def load_data(path):
    """Load and return the dataset as a DataFrame."""
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    print(f"\nDataset loaded: {len(df)} samples")
    print(df["label"].value_counts().to_string())
    return df


def train(df):
    """Train the Naive Bayes classifier and return model components."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label_num"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, vectorizer


def save_model(model, vectorizer):
    """Save trained model and vectorizer to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
    print("\nModel saved to model/")


def load_model():
    """Load saved model and vectorizer from disk."""
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
    return model, vectorizer


def predict(text, model, vectorizer):
    """Predict whether a given text is spam or ham."""
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "SPAM" if pred == 1 else "HAM"
    confidence = prob[pred] * 100
    return label, confidence


def interactive_mode(model, vectorizer):
    """Run an interactive CLI loop for testing the classifier."""
    print("\n--- Interactive Mode ---")
    print("Type an email/SMS message to classify it. Type 'quit' to exit.\n")
    while True:
        text = input("Enter message: ").strip()
        if text.lower() == "quit":
            break
        if not text:
            continue
        label, confidence = predict(text, model, vectorizer)
        print(f"  → Prediction : {label}  ({confidence:.1f}% confidence)\n")


if __name__ == "__main__":
    path = download_dataset()
    df = load_data(path)
    model, vectorizer = train(df)
    save_model(model, vectorizer)

    samples = [
        "Congratulations! You've won a FREE iPhone. Click now to claim your prize!",
        "Hey, are we still meeting at 3pm today?",
        "URGENT: Your bank account has been suspended. Verify immediately.",
        "Thanks for sending the report, I'll review it tonight.",
    ]

    print("\n--- Sample Predictions ---")
    for s in samples:
        label, conf = predict(s, model, vectorizer)
        print(f"  [{label:4s}] ({conf:.1f}%)  {s[:60]}...")

    interactive_mode(model, vectorizer)
