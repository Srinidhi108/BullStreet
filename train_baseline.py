"""
train_baseline.py  -  first ML run on the BullStreet dataset.

Put this file in the BullStreet-main folder (next to main.py) and run:
    python train_baseline.py

What it does:
  1. Loads the date-based train / validation splits.
  2. Preprocesses (fit on train only, so there is no leakage).
  3. Trains: "always positive" baseline, Logistic Regression, Random Forest, XGBoost.
  4. Compares them on the VALIDATION set (the test set is left untouched on purpose).
  5. Runs SHAP on the best tree model and saves a plot to reports/shap_summary.png.
"""
import warnings
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from src.preprocessor import (
    LeakageFreePreprocessor,
    prepare_modeling_dataset,
    NUMERICAL_FEATURE_COLS,
    CATEGORICAL_FEATURE_COLS,
)

warnings.filterwarnings("ignore")

# ---------- 1. Load data ----------
train = pd.read_csv("data/splits/train.csv")
val = pd.read_csv("data/splits/validation.csv")

X_tr_raw, y_tr, _ = prepare_modeling_dataset(train, task="classification")
X_va_raw, y_va, _ = prepare_modeling_dataset(val, task="classification")
print(f"Train: {len(y_tr)} IPOs ({y_tr.mean():.1%} positive)")
print(f"Validation: {len(y_va)} IPOs ({y_va.mean():.1%} positive)\n")

# ---------- 2. Preprocess (fit on train only) ----------
pre = LeakageFreePreprocessor()
pre.fit(X_tr_raw)
X_tr = pre.transform(X_tr_raw)
X_va = pre.transform(X_va_raw)
feature_names = NUMERICAL_FEATURE_COLS + CATEGORICAL_FEATURE_COLS

# ---------- 3. Models ----------
models = {
    "Always-positive baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        class_weight="balanced", random_state=42,
    ),
}

try:
    from xgboost import XGBClassifier
    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    models["XGBoost"] = XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg / pos, eval_metric="logloss", random_state=42,
    )
except ImportError:
    print("(xgboost not installed - skipping it. Run: pip install xgboost)\n")

# ---------- 4. Train and compare ----------
rows = []
fitted = {}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    pred = model.predict(X_va)
    proba = model.predict_proba(X_va)[:, 1]
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_va, pred),
        "AUC": roc_auc_score(y_va, proba),
        "Precision": precision_score(y_va, pred, zero_division=0),
        "Recall": recall_score(y_va, pred, zero_division=0),
        "F1": f1_score(y_va, pred, zero_division=0),
    })
    fitted[name] = model

results = pd.DataFrame(rows).set_index("Model").round(3)
print("=== Validation results ===")
print(results.to_string())
print("\nTip: compare against the 'Always-positive baseline'. AUC of 0.5 = random guessing.\n")

# ---------- 5. SHAP on the best tree model ----------
tree_models = [m for m in ("XGBoost", "Random Forest") if m in fitted]
if tree_models:
    best = max(tree_models, key=lambda m: results.loc[m, "AUC"])
    try:
        import os
        import numpy as np
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        print(f"Running SHAP on: {best}")
        explainer = shap.TreeExplainer(fitted[best])
        sv = explainer.shap_values(X_va)
        if isinstance(sv, list):          # older shap: list of arrays per class
            sv = sv[1]
        elif getattr(sv, "ndim", 2) == 3:  # newer shap: (rows, features, classes)
            sv = sv[:, :, 1]

        os.makedirs("reports", exist_ok=True)
        shap.summary_plot(sv, X_va, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig("reports/shap_summary.png", dpi=150)
        plt.close()

        importance = pd.Series(np.abs(sv).mean(axis=0), index=feature_names)
        print("Saved reports/shap_summary.png")
        print("\nTop 8 features by mean |SHAP|:")
        print(importance.sort_values(ascending=False).head(8).round(4).to_string())
    except ImportError:
        print("SHAP/matplotlib not installed. Run: pip install shap matplotlib")
