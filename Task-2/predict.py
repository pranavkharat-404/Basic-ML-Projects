"""
predict.py — Run the saved best model on new transactions.

Usage:
    python predict.py --input data/fraudTest.csv --output outputs/predictions.csv
    python predict.py --input my_new_transactions.csv
"""

import argparse
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

CATEGORICAL_COLS = ["category", "gender", "state", "job"]
DROP_COLS = [
    "Unnamed: 0", "trans_num", "first", "last",
    "street", "city", "zip", "dob", "merchant", "cc_num",
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")

    df["hour"]       = df["trans_date_trans_time"].dt.hour
    df["dayofweek"]  = df["trans_date_trans_time"].dt.dayofweek
    df["month"]      = df["trans_date_trans_time"].dt.month
    df["is_night"]   = df["hour"].between(0, 5).astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    df["lat_diff"]  = (df["lat"]  - df["merch_lat"]).abs()
    df["long_diff"] = (df["long"] - df["merch_long"]).abs()
    df["geo_dist"]  = np.sqrt(df["lat_diff"]**2 + df["long_diff"]**2)
    df["log_amt"]   = np.log1p(df["amt"])

    df["dob_parsed"] = pd.to_datetime(df.get("dob", pd.NaT), errors="coerce")
    df["age"] = ((df["trans_date_trans_time"] - df["dob_parsed"]).dt.days // 365).fillna(0).astype(int)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    cols_to_drop = [c for c in DROP_COLS + ["trans_date_trans_time", "dob_parsed",
                                              "lat", "long", "merch_lat", "merch_long",
                                              "lat_diff", "long_diff", "is_fraud"]
                    if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    df.dropna(inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Fraud Detection — Predict")
    parser.add_argument("--input",  required=True,  help="Path to input CSV")
    parser.add_argument("--output", default="outputs/predictions.csv", help="Path to output CSV")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for flagging fraud (default 0.5)")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        sys.exit("❌  No saved model found. Run train.py first.")

    print(f"Loading model from {MODEL_PATH} …")
    bundle        = joblib.load(MODEL_PATH)
    model         = bundle["model"]
    feature_names = bundle["feature_names"]

    print(f"Loading input: {args.input} …")
    raw = pd.read_csv(args.input)
    df  = engineer(raw)
    X   = df.reindex(columns=feature_names, fill_value=0)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= args.threshold).astype(int)

    out = raw.iloc[df.index].copy()
    out["fraud_probability"] = probs.round(4)
    out["fraud_flag"]        = preds

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)

    n_flagged = preds.sum()
    print(f"\nResults:")
    print(f"  Transactions analysed : {len(preds)}")
    print(f"  Flagged as fraud      : {n_flagged}  ({n_flagged/len(preds)*100:.2f}%)")
    print(f"  Output saved to       : {args.output}")


if __name__ == "__main__":
    main()
