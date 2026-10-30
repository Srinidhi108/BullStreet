"""Evaluate the saved BullStreet model once on the untouched chronological test set."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.model_engine import engine, FEATURE_NAMES
from src.preprocessor import prepare_modeling_dataset


TEST_PATH = "data/splits/test.csv"
REPORT_PATH = "reports/test_holdout_report.json"


def classification_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "threshold": threshold,
        "always_positive_baseline_accuracy": round(float((y_true == 1).mean()), 4),
        "sample_size": int(len(y_true)),
    }


def regression_metrics(y_true, predictions):
    return {
        "mae": round(float(mean_absolute_error(y_true, predictions)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, predictions))), 2),
        "r2": round(float(r2_score(y_true, predictions)), 4),
        "directional_accuracy": round(float(np.mean((predictions > 0) == (y_true > 0))), 4),
        "sample_size": int(len(y_true)),
    }


def evaluate_horizon(df, prefix, return_target, classification_threshold):
    X_reg, y_reg, _ = prepare_modeling_dataset(df, task={"7": "7_session", "30": "30_session"}[prefix])
    X_cls = X_reg
    y_cls = (y_reg > 0).astype(int)
    X_reg_t = engine.preprocessor.transform(X_reg)

    if prefix == "7":
        cal = engine.clf_7_cal
        xgb = engine.clf_7_xgb
        lgb = engine.clf_7_lgb
        reg_rf = engine.reg_7_rf
        reg_lgb = engine.reg_7_lgb
        cls_weights = engine.ensemble_weights["7_session"]
        reg_weight = engine.ensemble_weights["7_regression_rf"]
    else:
        cal = engine.clf_30_cal
        xgb = engine.clf_30_xgb
        lgb = engine.clf_30_lgb
        reg_rf = engine.reg_30_rf
        reg_lgb = engine.reg_30_lgb
        cls_weights = engine.ensemble_weights["30_session"]
        reg_weight = engine.ensemble_weights["30_regression_rf"]

    probabilities = (
        cls_weights["cal"] * cal.predict_proba(X_reg_t)[:, 1]
        + cls_weights["xgb"] * xgb.predict_proba(X_reg_t)[:, 1]
        + cls_weights["lgb"] * lgb.predict_proba(X_reg_t)[:, 1]
    )
    predictions = reg_weight * reg_rf.predict(X_reg_t) + (1.0 - reg_weight) * reg_lgb.predict(X_reg_t)
    return {
        "classification": classification_metrics(y_cls, probabilities, classification_threshold),
        "regression": regression_metrics(y_reg.to_numpy(), predictions),
    }


def main():
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(TEST_PATH)
    if not engine.load_artifacts():
        raise RuntimeError("Saved model bundle could not be loaded")

    test = pd.read_csv(TEST_PATH)
    result = {
        "test_file": TEST_PATH,
        "test_rows": len(test),
        "listing_day": {},
        "7_session": {},
        "30_session": {},
    }

    X_cls, y_cls, _ = prepare_modeling_dataset(test, task="classification")
    X_reg, y_reg, _ = prepare_modeling_dataset(test, task="regression")
    X_cls_t = engine.preprocessor.transform(X_cls)
    X_reg_t = engine.preprocessor.transform(X_reg)
    weights = engine.ensemble_weights["listing"]
    listing_prob = (
        weights["cal"] * engine.clf_rf.predict_proba(X_cls_t)[:, 1]
        + weights["xgb"] * engine.clf_xgb.predict_proba(X_cls_t)[:, 1]
        + weights["lgb"] * engine.clf_lgb.predict_proba(X_cls_t)[:, 1]
    )
    listing_pred = engine.ensemble_weights["listing_regression_rf"] * engine.reg_rf.predict(X_reg_t)
    listing_pred += (1.0 - engine.ensemble_weights["listing_regression_rf"]) * engine.reg_lgb.predict(X_reg_t)
    result["listing_day"] = {
        "classification": classification_metrics(y_cls, listing_prob, engine.classification_thresholds["listing"]),
        "regression": regression_metrics(y_reg.to_numpy(), listing_pred),
    }

    for prefix, target in [("7", "7_session"), ("30", "30_session")]:
        result[target] = evaluate_horizon(
            test,
            prefix,
            target,
            engine.classification_thresholds[f"{prefix}_session"],
        )

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
