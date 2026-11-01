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
from src.generalization import classification_metrics as rich_classification_metrics
from src.generalization import regression_metrics as rich_regression_metrics


TEST_PATH = "data/splits/test.csv"
REPORT_PATH = "reports/test_holdout_report.json"


def classification_metrics(y_true, probabilities, threshold):
    result = rich_classification_metrics(y_true, probabilities, threshold)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()} | {"threshold": threshold}


def regression_metrics(y_true, predictions):
    result = rich_regression_metrics(y_true, predictions)
    return {k: (round(v, 4) if k in {"r2", "directional_accuracy"} else round(v, 2) if isinstance(v, float) else v) for k, v in result.items()}


def evaluate_horizon(df, prefix, return_target, classification_threshold):
    X_reg, y_reg, _ = prepare_modeling_dataset(df, task={"7": "7_session", "30": "30_session"}[prefix])
    X_cls = X_reg
    y_cls = (y_reg > 0).astype(int)
    prep = engine.horizon_preprocessors.get(prefix, engine.preprocessor)
    X_reg_t = prep.transform(X_reg[prep.num_cols + prep.cat_cols])

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
        weights["cal"] * engine.clf_cal.predict_proba(X_cls_t)[:, 1]
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
