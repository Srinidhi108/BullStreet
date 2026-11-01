"""Leakage-safe rolling validation and model selection for IPO horizons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingRegressor,
    HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, mean_absolute_error, mean_squared_error,
    median_absolute_error, precision_score, r2_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from src.preprocessor import LeakageFreePreprocessor


def _safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None


def prior_adjust_probability(p, source_prior, target_prior):
    """Adjust calibrated probabilities for a time-varying class prior."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    source_prior = float(np.clip(source_prior, 1e-6, 1.0 - 1e-6))
    target_prior = float(np.clip(target_prior, 1e-6, 1.0 - 1e-6))
    odds = p / (1.0 - p)
    adjusted_odds = odds * (target_prior / (1.0 - target_prior)) / (source_prior / (1.0 - source_prior))
    return adjusted_odds / (1.0 + adjusted_odds)


def classification_metrics(y, p, threshold=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    prevalence = float(y.mean()) if len(y) else 0.0
    calibration_bins = []
    for low, high in zip(np.linspace(0.0, 1.0, 6)[:-1], np.linspace(0.0, 1.0, 6)[1:]):
        mask = (p >= low) & ((p < high) if high < 1.0 else (p <= high))
        if np.any(mask):
            calibration_bins.append({
                "mean_predicted": float(np.mean(p[mask])),
                "observed_rate": float(np.mean(y[mask])),
                "count": int(np.sum(mask)),
            })
    return {
        "roc_auc": _safe_auc(y, p),
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y, p)),
        "positive_prevalence": prevalence,
        "always_positive_accuracy": prevalence,
        "majority_accuracy": max(prevalence, 1.0 - prevalence),
        "predicted_positive_rate": float(pred.mean()) if len(pred) else 0.0,
        "calibration_bins": calibration_bins,
        "sample_size": int(len(y)),
    }


def regression_metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mean_pred = np.repeat(np.mean(y), len(y))
    median_pred = np.repeat(np.median(y), len(y))
    return {
        "mae": float(mean_absolute_error(y, p)),
        "median_absolute_error": float(median_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(y) > 1 else None,
        "directional_accuracy": float(np.mean((p > 0) == (y > 0))),
        "sample_size": int(len(y)),
        "baseline_mean_mae": float(mean_absolute_error(y, mean_pred)),
        "baseline_median_mae": float(mean_absolute_error(y, median_pred)),
        "beats_mean_baseline": bool(mean_absolute_error(y, p) < mean_absolute_error(y, mean_pred)),
        "beats_median_baseline": bool(mean_absolute_error(y, p) < mean_absolute_error(y, median_pred)),
    }


def _clf_candidates(y):
    pos = max(int(np.sum(y == 1)), 1)
    neg = max(int(np.sum(y == 0)), 1)
    scale = neg / pos
    return {
        "logistic_regression": LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=6,
            max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=180, max_depth=2, learning_rate=0.03,
            min_child_weight=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
            reg_alpha=0.1, scale_pos_weight=scale, eval_metric="logloss", random_state=42,
            n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=300, max_depth=5, min_samples_leaf=6,
            max_features=0.7, class_weight="balanced", random_state=42, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=10,
            learning_rate=0.03, l2_regularization=5.0, random_state=42),
        "lightgbm": LGBMClassifier(n_estimators=180, max_depth=3, num_leaves=10,
            learning_rate=0.03, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=5.0, reg_alpha=0.1, random_state=42, verbose=-1, n_jobs=-1),
    }


def _reg_candidates():
    return {
        "median_baseline": DummyRegressor(strategy="median"),
        "mean_baseline": DummyRegressor(strategy="mean"),
        "huber": HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=1000),
        "random_forest": RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=6,
            max_features=0.8, random_state=42, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(n_estimators=300, max_depth=6, min_samples_leaf=6,
            max_features=0.8, random_state=42, n_jobs=-1),
        "gradient_boosting_huber": GradientBoostingRegressor(n_estimators=150, max_depth=2,
            min_samples_leaf=8, learning_rate=0.03, loss="huber", random_state=42),
        "xgboost_huber": XGBRegressor(n_estimators=180, max_depth=2, learning_rate=0.03,
            min_child_weight=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=8.0,
            reg_alpha=0.2, objective="reg:pseudohubererror", eval_metric="mae",
            random_state=42, n_jobs=-1),
        "lightgbm_huber": LGBMRegressor(n_estimators=180, max_depth=3, num_leaves=10,
            learning_rate=0.03, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=8.0, reg_alpha=0.2, objective="huber", random_state=42,
            verbose=-1, n_jobs=-1),
    }


def _threshold(y, p, expected_prevalence=None):
    """Choose a stable operating threshold, not an all-positive shortcut."""
    values = np.linspace(0.20, 0.80, 61)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    prevalence = float(expected_prevalence if expected_prevalence is not None else y.mean())
    candidates = []
    for threshold in values:
        pred = p >= threshold
        predicted_rate = float(pred.mean())
        if predicted_rate in (0.0, 1.0):
            continue
        bal = balanced_accuracy_score(y, pred)
        f1 = f1_score(y, pred, zero_division=0)
        # Keep the operating point near the expected class prior while still
        # rewarding useful separation. The penalty is soft, not a hard prior.
        prior_penalty = abs(predicted_rate - prevalence)
        score = 0.60 * bal + 0.30 * f1 - 0.10 * prior_penalty
        candidates.append((score, bal, f1, -prior_penalty, float(threshold)))
    if not candidates:
        return 0.50
    return max(candidates)[-1]


def select_with_rolling_cv(df, target, task, n_splits=4, feature_cols=None, recency_half_life=180.0,
                           prefer_simpler=False):
    """Select a model using expanding, chronological folds inside train only."""
    work = df.dropna(subset=[target]).copy()
    dates = pd.to_datetime(work.get("listing_date"), dayfirst=True, errors="coerce")
    work["__date"] = dates
    work = work.sort_values(["__date", "ipo_id"], na_position="last").reset_index(drop=True)
    features = [c for c in work.columns if c in (
        ["issue_price", "total_issue_size_cr", "fresh_issue_amount_cr", "ofs_amount_cr",
         "fresh_issue_pct", "ofs_pct", "lot_size", "face_value", "qib_subscription",
         "nii_subscription", "retail_subscription", "total_subscription", "revenue_cr",
         "revenue_growth_pct", "pat_cr", "pat_growth_pct", "pat_margin_pct", "total_assets_cr",
         "total_borrowings_cr", "return_on_assets_pct", "debt_to_assets_ratio", "profitable_before_ipo",
         "nifty_return_5d", "nifty_return_20d", "nifty_return_60d", "nifty_volatility_20d",
         "india_vix", "usd_inr_rate", "listing_segment", "exchange", "sector"]
    )]
    # Keep this local list synchronized with the project's preprocessor.
    from src.preprocessor import NUMERICAL_FEATURE_COLS, CATEGORICAL_FEATURE_COLS
    features = feature_cols or (NUMERICAL_FEATURE_COLS + CATEGORICAL_FEATURE_COLS)
    profile_num = [c for c in features if c in NUMERICAL_FEATURE_COLS]
    profile_cat = [c for c in features if c in CATEGORICAL_FEATURE_COLS]
    y = work[target].astype(float).to_numpy()
    is_class = task == "classification"
    y_class = (y > 0).astype(int) if is_class else None
    candidates = _clf_candidates(y_class) if is_class else _reg_candidates()
    splitter = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(work) // 70)))
    folds = list(splitter.split(work))
    scores = {}
    oof_by_model = {}
    for name, prototype in candidates.items():
        fold_scores, oof_y, oof_p = [], [], []
        for train_idx, val_idx in folds:
            if is_class and len(np.unique(y_class[train_idx])) < 2:
                continue
            prep = LeakageFreePreprocessor(num_cols=profile_num, cat_cols=profile_cat).fit(work.iloc[train_idx][features])
            Xtr = prep.transform(work.iloc[train_idx][features])
            Xva = prep.transform(work.iloc[val_idx][features])
            model = clone_estimator(prototype)
            if is_class and recency_half_life:
                age_days = (work.iloc[train_idx]["__date"].max() - work.iloc[train_idx]["__date"]).dt.days.to_numpy()
                sample_weight = np.power(0.5, np.maximum(age_days, 0) / recency_half_life)
                model.fit(Xtr, y_class[train_idx], sample_weight=sample_weight)
            elif is_class:
                model.fit(Xtr, y_class[train_idx])
            else:
                model.fit(Xtr, y[train_idx])
            p = model.predict_proba(Xva)[:, 1] if is_class else model.predict(Xva)
            oof_y.extend((y_class[val_idx] if is_class else y[val_idx]).tolist())
            oof_p.extend(np.asarray(p).tolist())
            fold_scores.append(_safe_auc(y_class[val_idx], p) if is_class else mean_absolute_error(y[val_idx], p))
        if fold_scores:
            scores[name] = {
                "fold_scores": [None if s is None else float(s) for s in fold_scores],
                "mean_primary": float(np.nanmean(fold_scores)),
                "median_primary": float(np.nanmedian(fold_scores)),
                "worst_fold": float(min(fold_scores) if is_class else max(fold_scores)),
                "fold_std": float(np.nanstd(fold_scores)),
                "fold_count": len(fold_scores),
            }
            oof_by_model[name] = (np.asarray(oof_y), np.asarray(oof_p))
    if not scores:
        raise ValueError(f"No valid rolling folds for {target}")
    if is_class:
        # AUC is primary, with stability and balanced classification as
        # tie-breakers. Avoid selecting a volatile model for a tiny horizon.
        selected = max(scores, key=lambda n: (
            scores[n]["mean_primary"] - 0.05 * scores[n]["fold_std"],
            scores[n]["worst_fold"],
        ))
        if prefer_simpler:
            best_mean = scores[selected]["mean_primary"]
            best_recent = scores[selected]["fold_scores"][-1]
            simple_order = ["logistic_regression", "random_forest", "extra_trees",
                            "hist_gradient_boosting", "lightgbm", "xgboost"]
            eligible = [name for name in simple_order if name in scores and
                        scores[name]["mean_primary"] >= best_mean - 0.07 and
                        scores[name]["fold_scores"][-1] >= best_recent - 0.03]
            if eligible:
                selected = eligible[0]
    else:
        selected = min(scores, key=lambda n: scores[n]["mean_primary"])
    oy, op = oof_by_model[selected]
    if is_class:
        # The production model is calibrated. Recreate that calibration inside
        # each rolling fold before choosing a threshold; selecting a threshold
        # from raw RF scores and applying it to sigmoid probabilities is not
        # calibration-consistent and caused the 7-session all-negative output.
        calibrated_oof = []
        calibrated_y = []
        prototype = _clf_candidates(y_class)[selected]
        for train_idx, val_idx in folds:
            if len(np.unique(y_class[train_idx])) < 2:
                continue
            prep = LeakageFreePreprocessor(num_cols=profile_num, cat_cols=profile_cat).fit(work.iloc[train_idx][features])
            Xtr = prep.transform(work.iloc[train_idx][features])
            Xva = prep.transform(work.iloc[val_idx][features])
            base = clone_estimator(prototype)
            if recency_half_life:
                age_days = (work.iloc[train_idx]["__date"].max() - work.iloc[train_idx]["__date"]).dt.days.to_numpy()
                sample_weight = np.power(0.5, np.maximum(age_days, 0) / recency_half_life)
                try:
                    base.fit(Xtr, y_class[train_idx], sample_weight=sample_weight)
                except TypeError:
                    base.fit(Xtr, y_class[train_idx])
            else:
                base.fit(Xtr, y_class[train_idx])
            calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(Xtr, y_class[train_idx])
            calibrated_oof.extend(calibrated.predict_proba(Xva)[:, 1].tolist())
            calibrated_y.extend(y_class[val_idx].tolist())
        oy = np.asarray(calibrated_y)
        op = np.asarray(calibrated_oof)
        source_prior = float(y_class.mean())
        recent_window = max(40, len(y_class) // 4)
        recent_train_prior = float(y_class[-recent_window:].mean())
        op = prior_adjust_probability(op, source_prior, recent_train_prior)
        # Use the most recent fold prior as the expected deployment prior.
        threshold = _threshold(oy, op, expected_prevalence=recent_train_prior)
    else:
        threshold = None
    scores[selected]["oof_metrics"] = classification_metrics(oy, op, threshold) if is_class else regression_metrics(oy, op)
    # Explicit baseline references are always reported, even if a baseline wins.
    if is_class:
        scores["baselines"] = {
            "always_positive": {"accuracy": float(oy.mean()), "roc_auc": 0.5, "pr_auc": float(oy.mean())},
            "majority_class": {"accuracy": float(max(oy.mean(), 1 - oy.mean()))},
        }
    else:
        scores["baselines"] = {
            "mean_return": regression_metrics(oy, np.repeat(np.mean(oy), len(oy))),
            "median_return": regression_metrics(oy, np.repeat(np.median(oy), len(oy))),
        }
    return {"selected_model": selected, "threshold": threshold, "scores": scores,
            "target_rows": int(len(work)), "oof_y": oy, "oof_p": op, "features": features,
            "predicted_positive_rate": float(np.mean(op >= threshold)) if is_class else None,
            "actual_positive_rate": float(np.mean(oy)) if is_class else None,
            "source_positive_rate": float(y_class.mean()) if is_class else None,
            "deployment_positive_rate": float(y_class[folds[-1][0]].mean()) if is_class else None,
            "prior_correction": {
                "source": float(y_class.mean()),
                "target": float(y_class[-max(40, len(y_class) // 4):].mean()),
            } if is_class else None,
            "recency_half_life": recency_half_life}


def clone_estimator(estimator):
    from sklearn.base import clone
    return clone(estimator)


def final_fit(df, target, selection, task):
    from src.preprocessor import LeakageFreePreprocessor
    from src.preprocessor import NUMERICAL_FEATURE_COLS, CATEGORICAL_FEATURE_COLS
    features = NUMERICAL_FEATURE_COLS + CATEGORICAL_FEATURE_COLS
    work = df.dropna(subset=[target]).copy()
    prep = LeakageFreePreprocessor().fit(work[features])
    model = (_clf_candidates((work[target].astype(float).to_numpy() > 0).astype(int))
             if task == "classification" else _reg_candidates())[selection["selected_model"]]
    model = clone_estimator(model)
    yy = (work[target].astype(float).to_numpy() > 0).astype(int) if task == "classification" else work[target].astype(float).to_numpy()
    model.fit(prep.transform(work[features]), yy)
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3).fit(prep.transform(work[features]), yy) if task == "classification" else None
    return prep, model, calibrated
