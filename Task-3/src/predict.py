import joblib
from preprocessing import clean_text

# Load model
MODEL_PATH = "models/spam_model.pkl"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
tfidf = bundle["tfidf"]


def predict_message(message):
    # Clean input
    cleaned = clean_text(message)

    # Convert to TF-IDF
    vector = tfidf.transform([cleaned])

    # Predict
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0][1]

    return prediction, probability


if __name__ == "__main__":
    print("📩 Enter a message to check if it's Spam:\n")

    user_input = input("Message: ")

    pred, prob = predict_message(user_input)

    if pred == 1:
        print(f"\n🚨 SPAM detected! (Confidence: {prob:.2f})")
    else:
        print(f"\n✅ Legit message (Confidence: {1 - prob:.2f})")