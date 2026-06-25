import joblib
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "best_model.jb")
vectorizer = joblib.load(MODEL_DIR / "vectorizer.jb")


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z ]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text


def predict_news(news_text):

    news_text = clean_text(news_text)

    vector = vectorizer.transform([news_text])

    prediction = model.predict(vector)[0]

    confidence = model.predict_proba(vector).max()

    return {
    "prediction": "REAL" if prediction == 1 else "FAKE",
    "confidence": round(float(confidence) * 100, 2)
    }