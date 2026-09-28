"""
Experiment 1: Sentiment Analysis of Library Book Reviews
Performs TF-IDF feature extraction, model training (Logistic Regression / Naive Bayes),
comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix),
and live review inference.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "library_reviews.csv")

class SentimentAnalyzer:
    def __init__(self, model_type="logistic_regression"):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        if model_type == "naive_bayes":
            self.model = MultinomialNB()
        else:
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.is_trained = False
        self.evaluation_metrics = {}

    def load_dataset(self, filepath=None):
        if filepath is None:
            filepath = DATA_PATH
        return pd.read_csv(filepath)

    def train_and_evaluate(self, test_size=0.25, random_state=42):
        df = self.load_dataset()
        X = df["review"]
        y = df["sentiment"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.model.fit(X_train_vec, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test_vec)
        labels = sorted(list(set(y)))

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)

        self.evaluation_metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "labels": labels,
            "classification_report": report,
            "train_size": len(X_train),
            "test_size": len(X_test)
        }

        # Save artifacts
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, os.path.join(MODEL_DIR, f"sentiment_{self.model_type}.pkl"))
        joblib.dump(self.vectorizer, os.path.join(MODEL_DIR, "sentiment_vectorizer.pkl"))

        return self.evaluation_metrics

    def predict(self, text):
        if not self.is_trained:
            # Check if saved model exists
            model_path = os.path.join(MODEL_DIR, f"sentiment_{self.model_type}.pkl")
            vec_path = os.path.join(MODEL_DIR, "sentiment_vectorizer.pkl")
            if os.path.exists(model_path) and os.path.exists(vec_path):
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vec_path)
                self.is_trained = True
            else:
                self.train_and_evaluate()

        vec = self.vectorizer.transform([text])
        prediction = self.model.predict(vec)[0]
        
        # Calculate probabilities if supported
        confidence = 1.0
        probabilities = {}
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec)[0]
            classes = self.model.classes_
            for cls, prob in zip(classes, probs):
                probabilities[cls] = float(prob)
            confidence = float(max(probs))

        return {
            "sentiment": prediction,
            "confidence": confidence,
            "probabilities": probabilities
        }

# Singleton instance for quick access
sentiment_analyzer = SentimentAnalyzer()
