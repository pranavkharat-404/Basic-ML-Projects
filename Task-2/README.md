#Credit Card Fraud Detection
**Dataset:** [Kaggle — kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

Trains and compares three classifiers. Logistic Regression, Decision Tree, and Random Forest on the Sparkov-generated credit card transaction dataset.


## Features Used

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

## Models & Key Choices

| Model | Key Settings | Notes |
|---|---|---|
| Logistic Regression | `class_weight=balanced`, `C=0.1` | Good baseline; needs feature scaling |
| Decision Tree | `max_depth=8`, `class_weight=balanced` | Interpretable; prone to overfitting |
| Random Forest | `100 trees`, `max_depth=10`, `class_weight=balanced` | Best overall; use PR-AUC to compare |

> **Why PR-AUC and not accuracy?**
> The dataset is highly imbalanced (~0.58% fraud). A model that predicts "legit" 100% of the time
> gets 99.4% accuracy but catches zero fraud. PR-AUC rewards models that rank frauds highly
> among the top predictions a much more meaningful metric here.
