import pandas as pd
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Loading datasets...")

fake = pd.read_csv("../Datasets/Fake.csv")
true = pd.read_csv("../Datasets/True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true])

df = df.sample(frac=1, random_state=42)

print("Total Articles:", len(df))


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\\S+", "", text)

    text = re.sub(r"[^a-zA-Z ]", " ", text)

    text = re.sub(r"\\s+", " ", text)

    return text


df["content"] = (
    df["title"].fillna("")
    + " "
    + df["text"].fillna("")
)

df["content"] = df["content"].apply(clean_text)

X = df["content"]

y = df["label"]

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training model...")

model = LogisticRegression(max_iter=2000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"Accuracy: {accuracy:.4f}")

print(
    classification_report(
        y_test,
        predictions
    )
)

joblib.dump(
    model,
    "../Models/best_model.jb"
)

joblib.dump(
    vectorizer,
    "../Models/vectorizer.jb"
)

print("Model Saved Successfully")