import os
import joblib

from sklearn.model_selection import train_test_split

from src.preprocessing import load_and_preprocess
from src.feature_engineering import create_tfidf_features
from src.train import train_models
from src.evaluate import evaluate_models


DATA_PATH = "data/spam.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    print("\n[1/5] Loading and preprocessing data...")
    df = load_and_preprocess(DATA_PATH)

    X = df['message']
    y = df['label']

    print("\n[2/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n[3/5] Creating TF-IDF features...")
    X_train_tfidf, X_test_tfidf, tfidf = create_tfidf_features(X_train, X_test)

    print("\n[4/5] Training models...")
    models = train_models(X_train_tfidf, y_train)

    print("\n[5/5] Evaluating models...")
    results = evaluate_models(models, X_test_tfidf, y_test)

    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_model = models[best_model_name]

    print(f"\n🔥 Best Model: {best_model_name}")

    model_bundle = {
        "model": best_model,
        "tfidf": tfidf
    }

    joblib.dump(model_bundle, os.path.join(MODEL_DIR, "spam_model.pkl"))

    print("✅ Model saved to models/spam_model.pkl")


if __name__ == "__main__":
    main()