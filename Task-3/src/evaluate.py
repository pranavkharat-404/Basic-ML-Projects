import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

OUTPUT_DIR = "outputs/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_models(models, X_test, y_test):
    
    results = {}

    for name, model in models.items():
        
        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }

        print(f"\n{name}")
        print("-" * 40)
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")

        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"])

        disp.plot()
        plt.title(f"Confusion Matrix - {name}")

        filename = name.lower().replace(" ", "_")
        plt.savefig(f"{OUTPUT_DIR}/cm_{filename}.png")
        plt.close()

    return results