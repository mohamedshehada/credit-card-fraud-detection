# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions in a highly imbalanced dataset, built end-to-end: EDA, correct handling of class imbalance, model comparison, threshold tuning, and explainability.

## Problem

Credit card companies need to catch fraudulent transactions without wrongly blocking legitimate ones. The core challenge is **extreme class imbalance**: fraud makes up only **0.172%** of transactions in this dataset. A model that never predicts fraud would score ~99.8% accuracy while being completely useless — so this project is built specifically around metrics and techniques that hold up under that imbalance.

## Dataset

[Credit Card Fraud Detection (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions made by European cardholders over two days in September 2013, with 492 labeled as fraud.

- Features `V1`–`V28` are the output of a PCA transformation on the original transaction features, released this way by the data providers to protect confidentiality. They carry predictive signal but cannot be interpreted individually (e.g. "V14" cannot be read as "transaction location").
- `Time` (seconds since the first transaction) and `Amount` are the only two features in their original, interpretable form.
- `Class` is the target: `1` = fraud, `0` = legitimate.

Dataset collected by Worldline and the Machine Learning Group at Université Libre de Bruxelles. Full citation details are on the [Kaggle page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

**The raw CSV is not included in this repo** (large file, redistribution restricted by Kaggle's terms). Download it from the link above and place it as `creditcard.csv` — see [How to Run](#how-to-run).

## Approach

1. **Stratified train/test split first** — before any scaling or resampling, to prevent data leakage. `stratify=y` keeps the 0.172% fraud ratio consistent in both sets.
2. **Minimal preprocessing** — only `Amount` and `Time` are scaled; `V1`–`V28` are already PCA outputs and pre-scaled. No imputation, since the dataset has zero missing values (verified, not assumed).
3. **Imbalance handling via `class_weight` / `scale_pos_weight`**, benchmarked directly against SMOTE rather than assumed to be better.
4. **PR-AUC (Average Precision) as the primary metric**, not accuracy — this is also what the dataset's own documentation recommends, since ROC-AUC and accuracy are both misleadingly high on this kind of imbalance.
5. **Threshold tuned explicitly** against the precision-recall curve, rather than left at the default 0.5.
6. **SHAP** for model explainability, with an explicit acknowledgment of what SHAP can and can't tell us given the anonymized features.

## EDA

EDA here is intentionally scoped to what's actually interpretable:
- Class distribution confirms the 0.172% imbalance
- `Amount` and `Time` are the only two features examined individually, since `V1`–`V28` have no standalone real-world meaning
- No attempt is made to explain *why* any individual `V` column matters — only *that* it does (via SHAP later), since claiming a business explanation for a PCA component would be fabricated insight, not analysis

## Results

Three models were trained and compared with `class_weight`/`scale_pos_weight` for imbalance handling:

| Model | Train PR-AUC | Test PR-AUC | Test ROC-AUC | Train/Test Gap |
|---|---|---|---|---|
| Logistic Regression | 0.748 | 0.716 | 0.972 | 0.032 |
| Random Forest | 0.980 | 0.845 | 0.980 | 0.135 |
| XGBoost (regularized) | 0.999 | 0.874 | 0.987 | 0.125 |
| **XGBoost (tuned)** | 1.000 | **0.877** | 0.982 | 0.123 |

**XGBoost (tuned)** was selected as the final model, using hyperparameters found via `GridSearchCV` (`n_estimators=500, max_depth=3, learning_rate=0.2, subsample=0.8, colsample_bytree=0.8`), scored on PR-AUC with 3-fold CV, plus additional `reg_lambda`/`min_child_weight` regularization.

### Threshold selection

The default 0.5 cutoff is rarely optimal for rare-event detection. A full precision/recall sweep across thresholds showed:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.35 | 0.874 | 0.847 | **0.860** |
| 0.50 (default) | 0.880 | 0.827 | 0.853 |

**Threshold 0.35 was chosen** — it improves both recall and F1 over the default, with only a marginal precision trade-off. In fraud detection, a missed fraud case (false negative) is typically more costly than a false alarm that gets manually reviewed, so favoring recall is the right call here as long as precision doesn't collapse — which it doesn't.

At threshold 0.35 on the test set:
- **Recall ≈ 85%** of fraud cases caught
- **Precision ≈ 87%** of flagged transactions are actually fraud

### Explainability (SHAP)

Top features by mean absolute SHAP value: `V5`, `V15`, `Amount`, `V11`, `V13`, `V9`, `V4`, `V8`, `V27`, `V1`.

Because `V1`–`V28` are anonymized PCA components, SHAP can identify **which components drive predictions most**, but not **why** in business terms — this is a real constraint of the dataset, not a gap in the analysis.

## Limitations

- **Overfitting gap**: even after regularization (capped tree depth, L2 regularization, minimum child weight), the tuned XGBoost still reaches Train PR-AUC = 1.0000. With only 394 fraud cases in the training set and 500 boosting rounds, the model can still fit training residuals to zero despite shallow trees. The ~0.12 train/test gap should be read as a real limitation of working with this few positive examples, not something fully resolved.
- **Anonymized features**: `V1`–`V28` cannot be interpreted in business terms, which limits how actionable the SHAP results are for an actual fraud team, even though they're statistically valid.
- **Static snapshot**: the data covers only a two-day window from September 2013 — real fraud patterns shift over time, so a production system would need periodic retraining and drift monitoring rather than a one-time model.

## How to Run

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Update the `file_path` variable in the notebook to point to the file (e.g. your Google Drive path if running in Colab, or a local path otherwise).
3. Run `fraud_detection.ipynb` top to bottom.

## Project Structure

```
credit-card-fraud-detection/
├── README.md
└── Credit_Card_Fraud_Detection.ipynb
```

## Author

Mohammed Shehada — [GitHub](https://github.com/mohamedshehada)
