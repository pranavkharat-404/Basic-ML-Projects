import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # headless rendering

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

TRAIN_CSV = os.path.join(DATA_DIR, "fraudTrain.csv")
TEST_CSV  = os.path.join(DATA_DIR, "fraudTest.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

CATEGORICAL_COLS = ["category", "gender", "state", "job"]
DROP_COLS = [
    "Unnamed: 0", "trans_num", "first", "last",
    "street", "city", "zip", "dob",
    "merchant",
    "cc_num",
]

def load_and_engineer(path: str) -> pd.DataFrame:
    print(f"  Loading {os.path.basename(path)} …")
    df = pd.read_csv(path, parse_dates=["trans_date_trans_time"])

    df["hour"]       = df["trans_date_trans_time"].dt.hour
    df["dayofweek"]  = df["trans_date_trans_time"].dt.dayofweek
    df["month"]      = df["trans_date_trans_time"].dt.month
    df["is_night"]   = df["hour"].between(0, 5).astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    df["lat_diff"]  = (df["lat"]  - df["merch_lat"]).abs()
    df["long_diff"] = (df["long"] - df["merch_long"]).abs()
    df["geo_dist"]  = np.sqrt(df["lat_diff"]**2 + df["long_diff"]**2)

    df["log_amt"] = np.log1p(df["amt"])

    df["dob_parsed"] = pd.to_datetime(df["dob"], errors="coerce")
    df["age"] = (df["trans_date_trans_time"] - df["dob_parsed"]).dt.days // 365

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    cols_to_drop = [c for c in DROP_COLS + ["trans_date_trans_time", "dob_parsed",
                                              "lat", "long", "merch_lat", "merch_long",
                                              "lat_diff", "long_diff"]
                    if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    df.dropna(inplace=True)
    return df


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]
    return X, y

MODELS = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            C=0.1,
            solver="lbfgs",
        )),
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
}

def evaluate(name, model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}  (key metric for imbalanced data)")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    return y_pred, y_prob, roc_auc, pr_auc

def save_confusion_matrix(name, y_test, y_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Legit", "Fraud"],
        cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix — {name}")
    fig.tight_layout()
    slug = name.lower().replace(" ", "_")
    fig.savefig(os.path.join(OUTPUT_DIR, f"cm_{slug}.png"), dpi=120)
    plt.close(fig)

def save_roc_curves(roc_data):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (y_test, y_prob) in roc_data.items():
        RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=120)
    plt.close(fig)
    print(f"\n  ROC curve saved → outputs/roc_curves.png")

def save_feature_importance(name, model, feature_names):
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
    else:
        return
    idx = np.argsort(imp)[-20:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[idx], imp[idx], color="#3b82f6")
    ax.set_title(f"Feature Importance — {name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    slug = name.lower().replace(" ", "_")
    fig.savefig(os.path.join(OUTPUT_DIR, f"importance_{slug}.png"), dpi=120)
    plt.close(fig)
    print(f"  Feature importance saved → outputs/importance_{slug}.png")

def main():
    # ── 1. Load data
    print("\n[1/4] Loading & engineering features …")
    train_df = load_and_engineer(TRAIN_CSV)
    test_df  = load_and_engineer(TEST_CSV)

    X_train, y_train = split_xy(train_df)
    X_test,  y_test  = split_xy(test_df)

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(f"  Train shape : {X_train.shape}  | Fraud rate: {y_train.mean()*100:.2f}%")
    print(f"  Test  shape : {X_test.shape}   | Fraud rate: {y_test.mean()*100:.2f}%")
    print(f"  Features    : {list(X_train.columns)}")

    print("\n[2/4] Training models …")
    results  = {}
    roc_data = {}

    for name, model in MODELS.items():
        print(f"\n  ▶ {name}")
        model.fit(X_train, y_train)

        y_pred, y_prob, roc_auc, pr_auc = evaluate(name, model, X_test, y_test)
        results[name] = {"roc_auc": roc_auc, "pr_auc": pr_auc, "model": model}
        roc_data[name] = (y_test, y_prob)

        save_confusion_matrix(name, y_test, y_pred)
        save_feature_importance(name, model, X_train.columns)

    print("\n[3/4] Saving ROC curves …")
    save_roc_curves(roc_data)

    print("\n[4/4] Saving best model …")
    best_name = max(results, key=lambda n: results[n]["pr_auc"])
    best_model = results[best_name]["model"]
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump({"model": best_model, "feature_names": list(X_train.columns)}, model_path)
    print(f"  Best model  : {best_name}  (PR-AUC {results[best_name]['pr_auc']:.4f})")
    print(f"  Saved to    : models/best_model.pkl")

    print("\n" + "="*55)
    print("  SUMMARY")
    print("="*55)
    print(f"  {'Model':<22} {'ROC-AUC':>10} {'PR-AUC':>10}")
    print(f"  {'-'*42}")
    for n, r in sorted(results.items(), key=lambda x: -x[1]["pr_auc"]):
        tag = " ← best" if n == best_name else ""
        print(f"  {n:<22} {r['roc_auc']:>10.4f} {r['pr_auc']:>10.4f}{tag}")
    print("\nDone! Check the outputs/ folder for plots.\n")
    
if __name__ == "__main__":
    main()
