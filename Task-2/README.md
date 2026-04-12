# 🔍 Credit Card Fraud Detection
**Dataset:** [Kaggle — kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

Trains and compares three classifiers — Logistic Regression, Decision Tree, and Random Forest —
on the Sparkov-generated credit card transaction dataset.

---

## 📁 Folder Structure

```
fraud_detection_project/
│
├── data/                        ← 📥 PUT YOUR CSV FILES HERE
│   ├── fraudTrain.csv           ← training data  (1.2 M rows)
│   └── fraudTest.csv            ← test data      (555 K rows)
│
├── outputs/                     ← auto-created when you run train.py
│   ├── roc_curves.png
│   ├── cm_logistic_regression.png
│   ├── cm_decision_tree.png
│   ├── cm_random_forest.png
│   ├── importance_decision_tree.png
│   ├── importance_random_forest.png
│   └── predictions.csv          ← created by predict.py
│
├── models/                      ← auto-created when you run train.py
│   └── best_model.pkl           ← best model saved here
│
├── train.py                     ← main training script
├── predict.py                   ← inference on new data
├── requirements.txt             ← Python dependencies
└── README.md                    ← this file
```

---

## 📥 Step 1 — Download the Dataset

1. Go to → https://www.kaggle.com/datasets/kartik2112/fraud-detection
2. Click **Download** (you need a free Kaggle account)
3. Unzip the downloaded file — you'll get two CSVs:
   - `fraudTrain.csv`
   - `fraudTest.csv`
4. **Copy both files into the `data/` folder** of this project.

> ⚠️ The filenames must be exactly `fraudTrain.csv` and `fraudTest.csv`.

---

## 🐍 Step 2 — Set Up Python Environment

You need **Python 3.9 or higher**.

### Option A — Using a virtual environment (recommended)

```bash
# Navigate to the project folder
cd fraud_detection_project

# Create virtual environment
python -m venv venv

# Activate it
# macOS / Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option B — Using conda

```bash
conda create -n fraud python=3.10 -y
conda activate fraud
pip install -r requirements.txt
```

### Option C — Install globally (quick & dirty)

```bash
pip install -r requirements.txt
```

---

## 🚀 Step 3 — Train the Models

```bash
python train.py
```

This will:
1. Load and engineer features from both CSVs
2. Train all three models (Logistic Regression, Decision Tree, Random Forest)
3. Print classification reports and ROC-AUC / PR-AUC scores
4. Save confusion matrix plots → `outputs/`
5. Save ROC curve comparison → `outputs/roc_curves.png`
6. Save feature importance charts → `outputs/`
7. Save the best model (by PR-AUC) → `models/best_model.pkl`

**Expected runtime:** ~3–8 minutes (Random Forest is the slowest step)

---

## 🔮 Step 4 — Run Predictions on New Data

After training, you can score any CSV that has the same columns as the Kaggle dataset:

```bash
# Score the test set and save predictions
python predict.py --input data/fraudTest.csv --output outputs/predictions.csv

# Score your own file
python predict.py --input /path/to/my_transactions.csv --output outputs/my_predictions.csv

# Adjust the fraud flagging threshold (default 0.5)
python predict.py --input data/fraudTest.csv --threshold 0.3
```

The output CSV will have two new columns added:
- `fraud_probability` — model's confidence score (0–1)
- `fraud_flag` — 1 = flagged as fraud, 0 = legitimate

---

## 📊 Features Used

| Feature | Description |
|---|---|
| `amt` / `log_amt` | Transaction amount (raw + log-scaled) |
| `hour` | Hour of day (0–23) |
| `dayofweek` | Day of week (0=Mon, 6=Sun) |
| `month` | Month of year |
| `is_night` | 1 if transaction is between midnight–5 AM |
| `is_weekend` | 1 if Saturday or Sunday |
| `geo_dist` | Euclidean distance between cardholder home and merchant |
| `age` | Cardholder age at transaction time |
| `category` | Merchant category (label-encoded) |
| `gender` | Cardholder gender (label-encoded) |
| `state` | Cardholder state (label-encoded) |
| `job` | Cardholder job (label-encoded) |
| `city_pop` | Population of cardholder's city |

---

## 🧠 Models & Key Choices

| Model | Key Settings | Notes |
|---|---|---|
| Logistic Regression | `class_weight=balanced`, `C=0.1` | Good baseline; needs feature scaling |
| Decision Tree | `max_depth=8`, `class_weight=balanced` | Interpretable; prone to overfitting |
| Random Forest | `100 trees`, `max_depth=10`, `class_weight=balanced` | Best overall; use PR-AUC to compare |

> **Why PR-AUC and not accuracy?**
> The dataset is highly imbalanced (~0.58% fraud). A model that predicts "legit" 100% of the time
> gets 99.4% accuracy but catches zero fraud. PR-AUC rewards models that rank frauds highly
> among the top predictions — a much more meaningful metric here.

---

## 🛠 Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: data/fraudTrain.csv` | Put the CSVs in the `data/` folder |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Training is slow | Normal — Random Forest on 1.2M rows takes a few minutes |
| Memory error | Reduce dataset size: open `train.py`, add `train_df = train_df.sample(200_000)` after the load line |
| Plots don't open | They're saved to `outputs/` — open the PNG files directly |

---

## 📈 Expected Results (approximate)

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | ~0.97 | ~0.60 |
| Decision Tree | ~0.90 | ~0.55 |
| Random Forest | ~0.99 | ~0.85 |
