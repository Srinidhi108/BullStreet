"""
train_and_predict.py - Multi-horizon training and inference utility.

Usage:
    python train_and_predict.py
    python train_and_predict.py --ipo IPO_0700
    python train_and_predict.py --ipo "cargosol"

Evaluates models on train and validation partitions across Listing Day,
7-session, and 30-session horizons.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.preprocessor import (
    CATEGORICAL_FEATURE_COLS,
    NUMERICAL_FEATURE_COLS,
    LeakageFreePreprocessor,
)

warnings.filterwarnings("ignore")

FEATURES = NUMERICAL_FEATURE_COLS + CATEGORICAL_FEATURE_COLS
HORIZONS = {
    "Listing day": "positive_listing_return",
    "7 sessions": "positive_7_session_return",
    "30 sessions": "positive_30_session_return",
}


def make_models(y_train):
    """The models we compare. XGBoost is used only if it is installed."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5,
            class_weight="balanced", random_state=42,
        ),
    }
    try:
        from xgboost import XGBClassifier
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=neg / max(pos, 1), eval_metric="logloss", random_state=42,
        )
    except ImportError:
        print("(xgboost not installed, skipping it. Run: pip install xgboost)")
    return models


def train_horizon(train, val, target):
    """Train all models for one time point. Returns (results table, best model, preprocessor)."""
    tr = train.dropna(subset=[target])
    va = val.dropna(subset=[target])
    y_tr = tr[target].astype(int)
    y_va = va[target].astype(int)

    info = {"train_rows": len(tr), "val_rows": len(va),
            "train_pos": float(y_tr.mean()), "val_pos": float(y_va.mean())}
    if len(tr) < 30 or len(va) < 10 or y_tr.nunique() < 2 or y_va.nunique() < 2:
        return None, None, None, info

    pre = LeakageFreePreprocessor()
    pre.fit(tr[FEATURES])
    X_tr = pre.transform(tr[FEATURES])
    X_va = pre.transform(va[FEATURES])

    rows, fitted = [], {}
    base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    rows.append({"Model": "Always-positive baseline",
                 "Accuracy": accuracy_score(y_va, base.predict(X_va)), "AUC": 0.5,
                 "Precision": precision_score(y_va, base.predict(X_va), zero_division=0),
                 "Recall": recall_score(y_va, base.predict(X_va), zero_division=0),
                 "F1": f1_score(y_va, base.predict(X_va), zero_division=0)})

    for name, model in make_models(y_tr).items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        proba = model.predict_proba(X_va)[:, 1]
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_va, pred), "AUC": roc_auc_score(y_va, proba),
                     "Precision": precision_score(y_va, pred, zero_division=0),
                     "Recall": recall_score(y_va, pred, zero_division=0),
                     "F1": f1_score(y_va, pred, zero_division=0)})
        fitted[name] = model

    results = pd.DataFrame(rows).set_index("Model").round(3)
    best_name = results.drop(index="Always-positive baseline")["AUC"].idxmax()
    info["best"] = best_name
    return results, fitted[best_name], pre, info


def verdict(p):
    return "UP" if p >= 0.5 else "DOWN"


def actual_text(row, target):
    v = row.get(target)
    if pd.isna(v):
        return "n/a"
    return "UP" if int(v) == 1 else "DOWN"


def show_ipo(row, split_name, bundles):
    """Print the model's prediction for one IPO next to what really happened."""
    print("-" * 72)
    print(f"{row['ipo_id']}  |  {str(row['company_name']).title()}  |  {row['listing_segment']}")
    print(f"Listed: {row['listing_date']}   Issue price: {row['issue_price']}   "
          f"Total subscription: {row['total_subscription']}x   (split: {split_name})")
    if split_name == "train":
        print("Note: In-sample training record.")
    elif split_name == "test":
        print("Note: Held-out test split record.")
    X = pd.DataFrame([row[FEATURES]])
    print(f"  {'Time point':<13}{'Model score':>12}  {'Predicted':<10}{'Actual':<8}")
    for label, target in HORIZONS.items():
        bundle = bundles.get(label)
        if bundle is None:
            print(f"  {label:<13}{'(no model)':>12}")
            continue
        model, pre = bundle
        p = float(model.predict_proba(pre.transform(X))[:, 1][0])
        act = actual_text(row, target)
        tag = ""
        if act != "n/a":
            tag = "  OK" if verdict(p) == act else "  MISS"
        print(f"  {label:<13}{p * 100:>11.0f}%  {verdict(p):<10}{act:<8}{tag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipo", type=str, default=None, help="IPO id or part of company name")
    parser.add_argument("--demo-count", type=int, default=8)
    args = parser.parse_args()

    train = pd.read_csv("data/splits/train.csv")
    val = pd.read_csv("data/splits/validation.csv")
    test = pd.read_csv("data/splits/test.csv")

    bundles, summary_rows = {}, []
    print("=" * 72)
    print("TRAINING (train split) AND COMPARING (validation split)")
    print("=" * 72)
    for label, target in HORIZONS.items():
        results, model, pre, info = train_horizon(train, val, target)
        print(f"\n--- {label} ---")
        print(f"IPOs with this outcome: train {info['train_rows']} "
              f"({info['train_pos']:.0%} up), validation {info['val_rows']} "
              f"({info['val_pos']:.0%} up)")
        if results is None:
            print("Not enough data for this time point. Did you run the extended fetch?")
            print("  python main.py --stage clean --fetch-extended")
            continue
        print(results.to_string())
        print(f"Best model by AUC: {info['best']}")
        bundles[label] = (model, pre)
        best_row = results.loc[info["best"]]
        summary_rows.append({"Time point": label, "Best model": info["best"],
                             "Val IPOs": info["val_rows"], "Baseline accuracy": results.loc["Always-positive baseline", "Accuracy"],
                             "Accuracy": best_row["Accuracy"], "AUC": best_row["AUC"],
                             "Precision": best_row["Precision"], "Recall": best_row["Recall"]})

    if summary_rows:
        summary = pd.DataFrame(summary_rows).round(3)
        print("\n" + "=" * 72)
        print("SUMMARY (best model per time point, validation split)")
        print("=" * 72)
        print(summary.to_string(index=False))
        print("\nAUC: 0.5 = coin flip, 1.0 = perfect. Compare accuracy with the baseline column.")
        try:
            os.makedirs("reports", exist_ok=True)
            summary.to_csv("reports/horizon_results.csv", index=False)
            print("Saved reports/horizon_results.csv")
        except PermissionError:
            print("(could not save reports/horizon_results.csv - close it if it is open in Excel)")

    if not bundles:
        return

    print("\n" + "=" * 72)
    print("PREDICTIONS ON REAL IPOs")
    print("=" * 72)
    print("'Model score' is the model's confidence that the stock ends ABOVE its issue price.")
    print("It is a ranking score, not a guaranteed probability.\n")

    pool = pd.concat([train.assign(_split="train"), val.assign(_split="validation"),
                      test.assign(_split="test")], ignore_index=True)

    if args.ipo:
        q = args.ipo.strip().lower()
        hit = pool[(pool["ipo_id"].str.lower() == q) |
                   (pool["company_name"].astype(str).str.lower().str.contains(q, regex=False))]
        if hit.empty:
            print(f"No IPO found matching '{args.ipo}'.")
            return
        if len(hit) > 5:
            print(f"{len(hit)} matches, showing the first 5. Use the exact IPO id to pick one.")
        for _, row in hit.head(5).iterrows():
            show_ipo(row, row["_split"], bundles)
    else:
        needed = [t for t in HORIZONS.values()]
        demo = val.dropna(subset=needed).tail(args.demo_count)
        if demo.empty:
            demo = val.tail(args.demo_count)
        print(f"Showing the {len(demo)} most recent validation IPOs "
              f"(the model did NOT train on these).")
        for _, row in demo.iterrows():
            show_ipo(row, "validation", bundles)
    print("-" * 72)


if __name__ == "__main__":
    main()
