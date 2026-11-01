"""Controlled old-vs-current comparison on the same train/validation/test files.

The old architecture is recreated with its original fixed RF/XGBoost/LightGBM
models and validation-tuned blend/threshold. The test set is used only once for
the final comparison. The current model is loaded from its saved bundle.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier

from src.model_engine import engine
from src.preprocessor import LeakageFreePreprocessor, NUMERICAL_FEATURE_COLS, CATEGORICAL_FEATURE_COLS

FEATURES = NUMERICAL_FEATURE_COLS + CATEGORICAL_FEATURE_COLS


def cls_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {"auc": round(float(roc_auc_score(y, p)), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
            "brier": round(float(brier_score_loss(y, p)), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "threshold": threshold, "n": int(len(y))}


def reg_metrics(y, p):
    return {"mae": round(float(mean_absolute_error(y, p)), 2),
            "median_ae": round(float(np.median(np.abs(y - p))), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 2),
            "r2": round(float(r2_score(y, p)), 4),
            "directional_accuracy": round(float(np.mean((y > 0) == (p > 0))), 4), "n": int(len(y))}


def tune_weights(y, p1, p2, p3):
    best = (-np.inf, (0.4, 0.35, 0.25))
    for a in np.arange(0, 1.01, 0.05):
        for b in np.arange(0, 1.01 - a, 0.05):
            c = round(1 - a - b, 10)
            score = roc_auc_score(y, a * p1 + b * p2 + c * p3)
            if score > best[0]: best = (score, (a, b, c))
    return best[1]


def tune_reg_weight(y, p_rf, p_lgb):
    return min(np.arange(0, 1.01, 0.05), key=lambda w: mean_absolute_error(y, w * p_rf + (1 - w) * p_lgb))


def tune_threshold(y, p):
    return max(np.arange(0.05, 0.96, 0.01), key=lambda t: accuracy_score(y, p >= t))


def main():
    train = pd.read_csv("data/splits/train.csv")
    val = pd.read_csv("data/splits/validation.csv")
    test = pd.read_csv("data/splits/test.csv")
    prep = LeakageFreePreprocessor().fit(train[FEATURES])

    def X(frame): return prep.transform(frame[FEATURES])
    old = {}
    for prefix, target in [("listing", "listing_day_close_return_pct"), ("7", "return_7th_session_pct"), ("30", "return_30th_session_pct")]:
        tr = train.dropna(subset=[target]); va = val.dropna(subset=[target]); te = test.dropna(subset=[target])
        ytr = (tr[target].to_numpy(float) > 0).astype(int); yva = (va[target].to_numpy(float) > 0).astype(int); yte = (te[target].to_numpy(float) > 0).astype(int)
        neg, pos = max(1, int((ytr == 0).sum())), max(1, int((ytr == 1).sum()))
        models = [
            RandomForestClassifier(n_estimators=350, max_depth=6, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1),
            XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=neg / pos, eval_metric="logloss", random_state=42, n_jobs=-1),
            LGBMClassifier(n_estimators=120, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1),
        ]
        fitted = [m.fit(X(tr), ytr) for m in models]
        cal = CalibratedClassifierCV(fitted[0], method="sigmoid", cv=3).fit(X(tr), ytr)
        pv = [cal.predict_proba(X(va))[:, 1], fitted[1].predict_proba(X(va))[:, 1], fitted[2].predict_proba(X(va))[:, 1]]
        pt = [cal.predict_proba(X(te))[:, 1], fitted[1].predict_proba(X(te))[:, 1], fitted[2].predict_proba(X(te))[:, 1]]
        weights = tune_weights(yva, *pv)
        val_prob = sum(w * p for w, p in zip(weights, pv)); test_prob = sum(w * p for w, p in zip(weights, pt))
        threshold = 0.5 if prefix == "listing" else tune_threshold(yva, val_prob)
        old[prefix] = {"classification": cls_metrics(yte, test_prob, threshold), "validation_threshold": threshold, "validation_weights": weights}

        trr = train.dropna(subset=[target]); var = val.dropna(subset=[target]); ter = test.dropna(subset=[target])
        rf = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=4, random_state=42, n_jobs=-1).fit(X(trr), trr[target].to_numpy(float))
        lgb = LGBMRegressor(n_estimators=150, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1).fit(X(trr), trr[target].to_numpy(float))
        w = tune_reg_weight(var[target].to_numpy(float), rf.predict(X(varr := var)), lgb.predict(X(varr)))
        old[prefix]["regression"] = reg_metrics(ter[target].to_numpy(float), w * rf.predict(X(ter)) + (1 - w) * lgb.predict(X(ter)))

    with open("reports/test_holdout_report.json", encoding="utf-8") as h: current = json.load(h)
    comparison = {"protocol": "same current train/validation/test; old weights and thresholds selected on validation only", "old_architecture": old, "current_model": current}
    os.makedirs("reports", exist_ok=True)
    with open("reports/old_model_same_split_comparison.json", "w", encoding="utf-8") as h: json.dump(comparison, h, indent=2)

    print("=" * 88); print("OLD VS CURRENT MODEL — SAME SPLITS"); print("=" * 88)
    print(f"{'Horizon':<15} {'Metric':<18} {'Old':>10} {'Current':>10} {'Winner':>10}")
    for label, key in [("Listing day", "listing_day"), ("7 sessions", "7_session"), ("30 sessions", "30_session")]:
        for group, metric, lower in [("classification", "roc_auc", False), ("regression", "mae", True)]:
            old_key = "listing" if key == "listing_day" else key.split("_")[0]
            old_metric = "auc" if metric == "roc_auc" else metric
            ov = old[old_key][group][old_metric]
            cv = current[key][group]["roc_auc" if metric == "roc_auc" else "mae"]
            winner = "Old" if (ov > cv if not lower else ov < cv) else "Current"
            print(f"{label:<15} {metric:<18} {ov:>10} {cv:>10} {winner:>10}")
    print("\nDetailed JSON: reports/old_model_same_split_comparison.json")


if __name__ == "__main__": main()
