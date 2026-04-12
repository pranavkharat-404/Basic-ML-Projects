# src/train.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_model(X, y):
    """
    Train TF-IDF + Logistic Regression
    """

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_tfidf = tfidf.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_tfidf, y)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")
    joblib.dump(tfidf, "model/tfidf.pkl")

    return model, tfidf