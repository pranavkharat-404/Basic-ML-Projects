# src/evaluate.py

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, tfidf, X_test, y_test):
    """
    Evaluate model performance
    """

    X_test_tfidf = tfidf.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    print("\n📊 Evaluation Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return y_pred