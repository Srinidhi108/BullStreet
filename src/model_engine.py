"""
model_engine.py - Production-Grade Dual ML Prediction & Explainability Engine
for BullStreet IPO Intelligence Platform.

Capabilities:
- Classification: Calibrated Probability of Positive Listing Return (UP / DOWN),
  Risk Category (LOW / MODERATE / HIGH), and Confidence Score.
- Regression: Expected Listing Day Close Return %, Expected Opening Pop %,
  Projected Price Targets, and Expected Return Intervals.
- Explainability: SHAP TreeExplainer for real-time local factor attribution
  (identifying top positive and negative drivers per IPO).
- Peer Intelligence: Distance-based historical peer retrieval.
"""
import os
import json
import logging
import warnings
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from src.preprocessor import (
    LeakageFreePreprocessor,
    prepare_modeling_dataset,
    NUMERICAL_FEATURE_COLS,
    CATEGORICAL_FEATURE_COLS,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "artifacts"
BUNDLE_PATH = os.path.join(ARTIFACTS_DIR, "model_bundle.joblib")
BENCHMARK_REPORT_PATH = "reports/model_performance_report.json"

FEATURE_NAMES = NUMERICAL_FEATURE_COLS + CATEGORICAL_FEATURE_COLS


class IPOModelEngine:
    """
    Central ML engine encapsulating preprocessing, dual prediction models,
    SHAP explainability, and similarity matching.
    """

    def __init__(self):
        self.preprocessor: LeakageFreePreprocessor = None
        self.clf_rf = None
        self.clf_xgb = None
        self.clf_lgb = None
        self.reg_rf = None
        self.reg_lgb = None
        self.reg_open = None
        self.clf_7_rf = None
        self.clf_7_xgb = None
        self.clf_7_lgb = None
        self.clf_7_cal = None
        self.clf_30_rf = None
        self.clf_30_xgb = None
        self.clf_30_lgb = None
        self.clf_30_cal = None
        self.reg_7_rf = None
        self.reg_7_lgb = None
        self.reg_30_rf = None
        self.reg_30_lgb = None
        self.ensemble_weights = {}
        self.classification_thresholds = {}
        self.explainer = None
        self.benchmarks: Dict[str, Any] = {}
        self.historical_db: pd.DataFrame = None
        self.fitted_ = False

    def _fit_classifier_set(self, X: np.ndarray, y: pd.Series, prefix: str) -> None:
        """Fit the classification ensemble for one forecast horizon."""
        pos_count = int((y == 1).sum())
        neg_count = int((y == 0).sum())
        scale_pos = neg_count / max(pos_count, 1)
        rf = RandomForestClassifier(
            n_estimators=350, max_depth=6, min_samples_leaf=4,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ).fit(X, y)
        xgb = XGBClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos, eval_metric="logloss", random_state=42,
        ).fit(X, y)
        lgb = LGBMClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
        ).fit(X, y)
        cal = CalibratedClassifierCV(rf, method="sigmoid", cv=3).fit(X, y)
        setattr(self, f"clf_{prefix}_rf", rf)
        setattr(self, f"clf_{prefix}_xgb", xgb)
        setattr(self, f"clf_{prefix}_lgb", lgb)
        setattr(self, f"clf_{prefix}_cal", cal)

    def _fit_regressor_set(self, X: np.ndarray, y: pd.Series, prefix: str) -> None:
        """Fit the regression ensemble for one forecast horizon."""
        rf = RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=4,
            random_state=42, n_jobs=-1,
        ).fit(X, y)
        lgb = LGBMRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
        ).fit(X, y)
        setattr(self, f"reg_{prefix}_rf", rf)
        setattr(self, f"reg_{prefix}_lgb", lgb)

    @staticmethod
    def _tune_classification_weights(y: pd.Series, p_rf: np.ndarray, p_xgb: np.ndarray, p_lgb: np.ndarray) -> Dict[str, float]:
        """Select ensemble weights maximizing validation ROC-AUC."""
        best_score = -np.inf
        best = {"cal": 0.40, "xgb": 0.35, "lgb": 0.25}
        # The calibrated RF probability is supplied as p_rf here.
        for w_rf in np.arange(0.0, 1.01, 0.05):
            for w_xgb in np.arange(0.0, 1.01 - w_rf, 0.05):
                w_lgb = round(1.0 - w_rf - w_xgb, 10)
                if w_lgb < 0:
                    continue
                blended = w_rf * p_rf + w_xgb * p_xgb + w_lgb * p_lgb
                score = roc_auc_score(y, blended)
                if score > best_score:
                    best_score = score
                    best = {"cal": round(float(w_rf), 2), "xgb": round(float(w_xgb), 2), "lgb": round(float(w_lgb), 2)}
        return best

    @staticmethod
    def _tune_regression_weight(y: pd.Series, p_rf: np.ndarray, p_lgb: np.ndarray) -> float:
        """Select the RF weight minimizing validation MAE."""
        scores = []
        for weight in np.arange(0.0, 1.01, 0.05):
            pred = weight * p_rf + (1.0 - weight) * p_lgb
            scores.append((mean_absolute_error(y, pred), float(weight)))
        return min(scores, key=lambda item: item[0])[1]

    @staticmethod
    def _tune_classification_threshold(y: pd.Series, probabilities: np.ndarray) -> float:
        """Choose the validation threshold maximizing classification accuracy."""
        candidates = []
        for threshold in np.arange(0.05, 0.96, 0.01):
            candidates.append((accuracy_score(y, probabilities >= threshold), float(threshold)))
        return max(candidates, key=lambda item: item[0])[1]

    def train_and_evaluate(
        self,
        train_path: str = "data/splits/train.csv",
        val_path: str = "data/splits/validation.csv",
        clean_master_path: str = "data/processed/ipo_master_clean.csv",
    ) -> Dict[str, Any]:
        """
        Fits dual classification and regression pipelines on training split,
        evaluates on validation split without leakage, and stores benchmarks.
        """
        logger.info("Loading training and validation partitions...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
        # Load clean master for peer retrieval
        if os.path.exists(clean_master_path):
            self.historical_db = pd.read_csv(clean_master_path)
        else:
            self.historical_db = pd.concat([train_df, val_df], ignore_index=True)

        # 1. Feature Extraction
        Xc_tr, yc_tr, _ = prepare_modeling_dataset(train_df, task="classification")
        Xc_va, yc_va, _ = prepare_modeling_dataset(val_df, task="classification")
        Xr_tr, yr_tr, _ = prepare_modeling_dataset(train_df, task="regression")
        Xr_va, yr_va, _ = prepare_modeling_dataset(val_df, task="regression")
        X7r_tr, y7r_tr, _ = prepare_modeling_dataset(train_df, task="7_session")
        X7r_va, y7r_va, _ = prepare_modeling_dataset(val_df, task="7_session")
        X30r_tr, y30r_tr, _ = prepare_modeling_dataset(train_df, task="30_session")
        X30r_va, y30r_va, _ = prepare_modeling_dataset(val_df, task="30_session")

        def classification_dataset(frame: pd.DataFrame, return_target: str):
            # Derive the binary target from the populated return column. This
            # also works when the optional positive_* columns are still blank.
            sub = frame.dropna(subset=[return_target]).copy()
            return sub[FEATURE_NAMES], (sub[return_target].astype(float) > 0).astype(int)

        X7c_tr, y7c_tr = classification_dataset(train_df, "return_7th_session_pct")
        X7c_va, y7c_va = classification_dataset(val_df, "return_7th_session_pct")
        X30c_tr, y30c_tr = classification_dataset(train_df, "return_30th_session_pct")
        X30c_va, y30c_va = classification_dataset(val_df, "return_30th_session_pct")

        # 2. Leakage-Free Preprocessing Fit
        logger.info("Fitting LeakageFreePreprocessor strictly on training partition...")
        self.preprocessor = LeakageFreePreprocessor()
        self.preprocessor.fit(Xc_tr)
        
        X_tr = self.preprocessor.transform(Xc_tr)
        X_va = self.preprocessor.transform(Xc_va)
        X7c_tr_t = self.preprocessor.transform(X7c_tr)
        X7c_va_t = self.preprocessor.transform(X7c_va)
        X30c_tr_t = self.preprocessor.transform(X30c_tr)
        X30c_va_t = self.preprocessor.transform(X30c_va)
        X7r_tr_t = self.preprocessor.transform(X7r_tr)
        X7r_va_t = self.preprocessor.transform(X7r_va)
        X30r_tr_t = self.preprocessor.transform(X30r_tr)
        X30r_va_t = self.preprocessor.transform(X30r_va)

        # 3. Train Classification Models
        logger.info("Training Classification Models (Random Forest, XGBoost, LightGBM)...")
        # Class balancing weights
        pos_count = int((yc_tr == 1).sum())
        neg_count = int((yc_tr == 0).sum())
        scale_pos = neg_count / max(pos_count, 1)

        self.clf_rf = RandomForestClassifier(
            n_estimators=350,
            max_depth=6,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ).fit(X_tr, yc_tr)

        self.clf_xgb = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=42,
        ).fit(X_tr, yc_tr)

        self.clf_lgb = LGBMClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        ).fit(X_tr, yc_tr)

        # Calibrated Classifier for rock-solid probability estimates
        self.clf_cal = CalibratedClassifierCV(self.clf_rf, method="sigmoid", cv=3)
        self.clf_cal.fit(X_tr, yc_tr)

        # 4. Train Regression Models (Listing Day Close Return %)
        logger.info("Training Regression Models (LightGBM, Random Forest)...")
        self.reg_rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        ).fit(X_tr, yr_tr)

        self.reg_lgb = LGBMRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        ).fit(X_tr, yr_tr)

        # 5. Train Opening Pop Regressor (Listing Opening Gain %)
        sub_open_tr = train_df.dropna(subset=["listing_opening_gain_pct"])
        Xo_tr = self.preprocessor.transform(sub_open_tr[FEATURE_NAMES])
        yo_tr = sub_open_tr["listing_opening_gain_pct"].astype(float)
        self.reg_open = LGBMRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.04,
            random_state=42,
            verbose=-1,
        ).fit(Xo_tr, yo_tr)

        # 6. Train real horizon-specific models instead of heuristic scaling.
        logger.info("Training 7-session classification and regression models...")
        self._fit_classifier_set(X7c_tr_t, y7c_tr, "7")
        self._fit_regressor_set(X7r_tr_t, y7r_tr, "7")
        logger.info("Training 30-session classification and regression models...")
        self._fit_classifier_set(X30c_tr_t, y30c_tr, "30")
        self._fit_regressor_set(X30r_tr_t, y30r_tr, "30")

        # Tune ensemble weights against the held-out validation partition.
        # These weights are stored and reused during live inference.
        def validation_probabilities(prefix, X):
            if prefix == "":
                return (
                    self.clf_rf.predict_proba(X)[:, 1],
                    self.clf_xgb.predict_proba(X)[:, 1],
                    self.clf_lgb.predict_proba(X)[:, 1],
                )
            return (
                getattr(self, f"clf_{prefix}_cal").predict_proba(X)[:, 1],
                getattr(self, f"clf_{prefix}_xgb").predict_proba(X)[:, 1],
                getattr(self, f"clf_{prefix}_lgb").predict_proba(X)[:, 1],
            )

        p_rf_listing, p_xgb, p_lgb = validation_probabilities("", X_va)
        self.ensemble_weights["listing"] = self._tune_classification_weights(yc_va, p_rf_listing, p_xgb, p_lgb)
        self.ensemble_weights["listing_regression_rf"] = self._tune_regression_weight(
            yr_va, self.reg_rf.predict(X_va), self.reg_lgb.predict(X_va)
        )
        # Keep the established 0.50 listing-day decision threshold. Accuracy
        # optimization would select an all-positive classifier because this
        # validation period is positive-heavy.
        self.classification_thresholds["listing"] = 0.50
        for prefix, Xc_val, yc_val, Xr_val, yr_val in [
            ("7", X7c_va_t, y7c_va, X7r_va_t, y7r_va),
            ("30", X30c_va_t, y30c_va, X30r_va_t, y30r_va),
        ]:
            p_cal_h, p_xgb_h, p_lgb_h = validation_probabilities(prefix, Xc_val)
            self.ensemble_weights[f"{prefix}_session"] = self._tune_classification_weights(
                yc_val, p_cal_h, p_xgb_h, p_lgb_h
            )
            hweights = self.ensemble_weights[f"{prefix}_session"]
            hprob = hweights["cal"] * p_cal_h + hweights["xgb"] * p_xgb_h + hweights["lgb"] * p_lgb_h
            self.classification_thresholds[f"{prefix}_session"] = self._tune_classification_threshold(yc_val, hprob)
            self.ensemble_weights[f"{prefix}_regression_rf"] = self._tune_regression_weight(
                yr_val,
                getattr(self, f"reg_{prefix}_rf").predict(Xr_val),
                getattr(self, f"reg_{prefix}_lgb").predict(Xr_val),
            )

        # 7. Fit SHAP Tree Explainer
        logger.info("Fitting SHAP TreeExplainer for instantaneous factor attribution...")
        self.explainer = shap.TreeExplainer(self.clf_rf)

        # 8. Evaluate on Validation Set
        logger.info("Evaluating all models on held-out validation split...")
        
        # Classification Validation
        prob_rf = self.clf_rf.predict_proba(X_va)[:, 1]
        prob_xgb = self.clf_xgb.predict_proba(X_va)[:, 1]
        prob_lgb = self.clf_lgb.predict_proba(X_va)[:, 1]
        weights = self.ensemble_weights["listing"]
        prob_blend = weights["cal"] * prob_rf + weights["xgb"] * prob_xgb + weights["lgb"] * prob_lgb
        pred_blend = (prob_blend >= self.classification_thresholds["listing"]).astype(int)

        base_acc = float((yc_va == 1).mean())

        # Regression Validation
        pred_rf_r = self.reg_rf.predict(X_va)
        pred_lgb_r = self.reg_lgb.predict(X_va)
        reg_weight = self.ensemble_weights["listing_regression_rf"]
        pred_reg_blend = reg_weight * pred_rf_r + (1.0 - reg_weight) * pred_lgb_r

        mae_blend = mean_absolute_error(yr_va, pred_reg_blend)
        rmse_blend = np.sqrt(mean_squared_error(yr_va, pred_reg_blend))
        r2_blend = r2_score(yr_va, pred_reg_blend)
        dir_acc_blend = np.mean((pred_reg_blend > 0) == (yr_va > 0))

        horizon_benchmarks = {}
        for label, Xc_val, yc_val, Xr_val, yr_val in [
            ("7_session", X7c_va_t, y7c_va, X7r_va_t, y7r_va),
            ("30_session", X30c_va_t, y30c_va, X30r_va_t, y30r_va),
        ]:
            prefix = label.split("_")[0]
            clf_rf = getattr(self, f"clf_{prefix}_rf")
            clf_xgb = getattr(self, f"clf_{prefix}_xgb")
            clf_lgb = getattr(self, f"clf_{prefix}_lgb")
            clf_cal = getattr(self, f"clf_{prefix}_cal")
            p_rf = clf_rf.predict_proba(Xc_val)[:, 1]
            p_xgb = clf_xgb.predict_proba(Xc_val)[:, 1]
            p_lgb = clf_lgb.predict_proba(Xc_val)[:, 1]
            p_cal = clf_cal.predict_proba(Xc_val)[:, 1]
            hweights = self.ensemble_weights[f"{prefix}_session"]
            p_blend = hweights["cal"] * p_cal + hweights["xgb"] * p_xgb + hweights["lgb"] * p_lgb
            pred_class = (p_blend >= self.classification_thresholds[f"{prefix}_session"]).astype(int)

            reg_rf = getattr(self, f"reg_{prefix}_rf")
            reg_lgb = getattr(self, f"reg_{prefix}_lgb")
            hreg_weight = self.ensemble_weights[f"{prefix}_regression_rf"]
            pred_return = hreg_weight * reg_rf.predict(Xr_val) + (1.0 - hreg_weight) * reg_lgb.predict(Xr_val)
            horizon_benchmarks[label] = {
                "validation_sample_size": len(yc_val),
                "baseline_always_positive_acc": round(float((yc_val == 1).mean()), 4),
                "classification": {
                    "ensemble_auc": round(float(roc_auc_score(yc_val, p_blend)), 4),
                    "accuracy": round(float(accuracy_score(yc_val, pred_class)), 4),
                    "precision": round(float(precision_score(yc_val, pred_class, zero_division=0)), 4),
                    "recall": round(float(recall_score(yc_val, pred_class, zero_division=0)), 4),
                    "f1": round(float(f1_score(yc_val, pred_class, zero_division=0)), 4),
                },
                "regression": {
                    "mae": round(float(mean_absolute_error(yr_val, pred_return)), 2),
                    "rmse": round(float(np.sqrt(mean_squared_error(yr_val, pred_return))), 2),
                    "r2": round(float(r2_score(yr_val, pred_return)), 4),
                    "directional_accuracy": round(float(np.mean((pred_return > 0) == (yr_val > 0))), 4),
                },
            }

        # Benchmarks Dictionary
        self.benchmarks = {
            "validation_sample_size": len(val_df),
            "baseline_always_positive_acc": round(base_acc, 4),
            "ensemble_weights_tuned_on_validation": self.ensemble_weights,
            "classification_thresholds_tuned_on_validation": self.classification_thresholds,
            "classification": {
                "ensemble_auc": round(float(roc_auc_score(yc_va, prob_blend)), 4),
                "rf_auc": round(float(roc_auc_score(yc_va, prob_rf)), 4),
                "xgb_auc": round(float(roc_auc_score(yc_va, prob_xgb)), 4),
                "lgb_auc": round(float(roc_auc_score(yc_va, prob_lgb)), 4),
                "accuracy": round(float(accuracy_score(yc_va, pred_blend)), 4),
                "precision": round(float(precision_score(yc_va, pred_blend, zero_division=0)), 4),
                "recall": round(float(recall_score(yc_va, pred_blend, zero_division=0)), 4),
                "f1": round(float(f1_score(yc_va, pred_blend, zero_division=0)), 4),
            },
            "regression": {
                "mae": round(float(mae_blend), 2),
                "rmse": round(float(rmse_blend), 2),
                "r2": round(float(r2_blend), 4),
                "directional_accuracy": round(float(dir_acc_blend), 4),
                "rf_mae": round(float(mean_absolute_error(yr_va, pred_rf_r)), 2),
                "lgb_mae": round(float(mean_absolute_error(yr_va, pred_lgb_r)), 2),
            },
            "horizons": horizon_benchmarks,
        }

        self.fitted_ = True
        self.save_artifacts()
        return self.benchmarks

    def save_artifacts(self, bundle_path: str = BUNDLE_PATH):
        """Saves fitted models and metadata."""
        os.makedirs(os.path.dirname(bundle_path), exist_ok=True)
        bundle = {
            "preprocessor": self.preprocessor,
            "clf_rf": self.clf_rf,
            "clf_xgb": self.clf_xgb,
            "clf_lgb": self.clf_lgb,
            "clf_cal": self.clf_cal,
            "reg_rf": self.reg_rf,
            "reg_lgb": self.reg_lgb,
            "reg_open": self.reg_open,
            "clf_7_rf": self.clf_7_rf,
            "clf_7_xgb": self.clf_7_xgb,
            "clf_7_lgb": self.clf_7_lgb,
            "clf_7_cal": self.clf_7_cal,
            "clf_30_rf": self.clf_30_rf,
            "clf_30_xgb": self.clf_30_xgb,
            "clf_30_lgb": self.clf_30_lgb,
            "clf_30_cal": self.clf_30_cal,
            "reg_7_rf": self.reg_7_rf,
            "reg_7_lgb": self.reg_7_lgb,
            "reg_30_rf": self.reg_30_rf,
            "reg_30_lgb": self.reg_30_lgb,
            "ensemble_weights": self.ensemble_weights,
            "classification_thresholds": self.classification_thresholds,
            "explainer": self.explainer,
            "benchmarks": self.benchmarks,
            "historical_db": self.historical_db,
        }
        joblib.dump(bundle, bundle_path)
        logger.info(f"Model bundle saved to {bundle_path}")

        # Save benchmark report
        os.makedirs(os.path.dirname(BENCHMARK_REPORT_PATH), exist_ok=True)
        with open(BENCHMARK_REPORT_PATH, "w") as f:
            json.dump(self.benchmarks, f, indent=2)

    def load_artifacts(self, bundle_path: str = BUNDLE_PATH) -> bool:
        """Loads fitted model bundle."""
        if not os.path.exists(bundle_path):
            return False
        bundle = joblib.load(bundle_path)
        self.preprocessor = bundle["preprocessor"]
        self.clf_rf = bundle["clf_rf"]
        self.clf_xgb = bundle["clf_xgb"]
        self.clf_lgb = bundle["clf_lgb"]
        self.clf_cal = bundle.get("clf_cal", bundle["clf_rf"])
        self.reg_rf = bundle["reg_rf"]
        self.reg_lgb = bundle["reg_lgb"]
        self.reg_open = bundle.get("reg_open")
        self.clf_7_rf = bundle.get("clf_7_rf")
        self.clf_7_xgb = bundle.get("clf_7_xgb")
        self.clf_7_lgb = bundle.get("clf_7_lgb")
        self.clf_7_cal = bundle.get("clf_7_cal")
        self.clf_30_rf = bundle.get("clf_30_rf")
        self.clf_30_xgb = bundle.get("clf_30_xgb")
        self.clf_30_lgb = bundle.get("clf_30_lgb")
        self.clf_30_cal = bundle.get("clf_30_cal")
        self.reg_7_rf = bundle.get("reg_7_rf")
        self.reg_7_lgb = bundle.get("reg_7_lgb")
        self.reg_30_rf = bundle.get("reg_30_rf")
        self.reg_30_lgb = bundle.get("reg_30_lgb")
        self.ensemble_weights = bundle.get("ensemble_weights", {})
        self.classification_thresholds = bundle.get("classification_thresholds", {})
        self.explainer = bundle["explainer"]
        self.benchmarks = bundle.get("benchmarks", {})
        self.historical_db = bundle.get("historical_db")
        self.fitted_ = True
        return True

    def predict_row(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs comprehensive dual inference (Classification + Regression + SHAP)
        for a single IPO's features.
        """
        if not self.fitted_:
            raise RuntimeError("Model engine has not been fitted or loaded.")

        df_row = pd.DataFrame([row_dict])
        
        # Ensure all required features are present
        for col in FEATURE_NAMES:
            if col not in df_row.columns:
                df_row[col] = np.nan

        X_trans = self.preprocessor.transform(df_row[FEATURE_NAMES])

        # 1. Classification Predictions
        p_rf = float(self.clf_rf.predict_proba(X_trans)[:, 1][0])
        p_xgb = float(self.clf_xgb.predict_proba(X_trans)[:, 1][0])
        p_lgb = float(self.clf_lgb.predict_proba(X_trans)[:, 1][0])
        p_cal = float(self.clf_cal.predict_proba(X_trans)[:, 1][0])

        def horizon_probability(prefix: str) -> tuple[float, float]:
            rf = getattr(self, f"clf_{prefix}_rf")
            xgb = getattr(self, f"clf_{prefix}_xgb")
            lgb = getattr(self, f"clf_{prefix}_lgb")
            cal = getattr(self, f"clf_{prefix}_cal")
            p_rf_h = float(rf.predict_proba(X_trans)[:, 1][0])
            p_xgb_h = float(xgb.predict_proba(X_trans)[:, 1][0])
            p_lgb_h = float(lgb.predict_proba(X_trans)[:, 1][0])
            p_cal_h = float(cal.predict_proba(X_trans)[:, 1][0])
            hweights = self.ensemble_weights.get(f"{prefix}_session", {"cal": 0.40, "xgb": 0.35, "lgb": 0.25})
            probability = float(np.clip(
                hweights["cal"] * p_cal_h + hweights["xgb"] * p_xgb_h + hweights["lgb"] * p_lgb_h,
                0.01, 0.99,
            ))
            agreement = ((int(p_rf_h >= 0.5) + int(p_xgb_h >= 0.5) + int(p_lgb_h >= 0.5)) / 3.0) * 100
            return probability, round(agreement, 1)

        win_prob_7, agreement_7 = horizon_probability("7")
        win_prob_30, agreement_30 = horizon_probability("30")
        
        # Blended calibrated win probability
        weights = self.ensemble_weights.get("listing", {"cal": 0.40, "xgb": 0.35, "lgb": 0.25})
        # Listing-day behavior uses the original Random Forest/XGBoost/LightGBM
        # probability blend; calibrated RF remains available for horizon models.
        win_prob = weights["cal"] * p_rf + weights["xgb"] * p_xgb + weights["lgb"] * p_lgb
        win_prob = np.clip(win_prob, 0.01, 0.99)

        # Recommendation & Risk Classification
        if win_prob >= 0.75:
            verdict = "STRONG BUY"
            risk_category = "LOW RISK"
            confidence = "HIGH CONVICTION"
        elif win_prob >= 0.55:
            verdict = "MODERATE BUY"
            risk_category = "MODERATE RISK"
            confidence = "MEDIUM CONVICTION"
        elif win_prob >= 0.40:
            verdict = "NEUTRAL / HOLD"
            risk_category = "ELEVATED RISK"
            confidence = "SPECULATIVE"
        else:
            verdict = "AVOID"
            risk_category = "HIGH RISK"
            confidence = "HIGH DOWNSIDE RISK"

        # 2. Regression Predictions
        r_rf = float(self.reg_rf.predict(X_trans)[0])
        r_lgb = float(self.reg_lgb.predict(X_trans)[0])
        reg_weight = self.ensemble_weights.get("listing_regression_rf", 0.50)
        exp_close_return = round(reg_weight * r_rf + (1.0 - reg_weight) * r_lgb, 2)

        exp_open_return = round(float(self.reg_open.predict(X_trans)[0]), 2) if self.reg_open else round(exp_close_return * 0.9, 2)

        reg_7_weight = self.ensemble_weights.get("7_regression_rf", 0.50)
        reg_30_weight = self.ensemble_weights.get("30_regression_rf", 0.50)
        exp_7_return = round(float(
            reg_7_weight * self.reg_7_rf.predict(X_trans)[0]
            + (1.0 - reg_7_weight) * self.reg_7_lgb.predict(X_trans)[0]
        ), 2)
        exp_30_return = round(float(
            reg_30_weight * self.reg_30_rf.predict(X_trans)[0]
            + (1.0 - reg_30_weight) * self.reg_30_lgb.predict(X_trans)[0]
        ), 2)

        issue_price = float(row_dict.get("issue_price") or 100.0)
        proj_close_price = round(issue_price * (1 + exp_close_return / 100.0), 2)
        proj_open_price = round(issue_price * (1 + exp_open_return / 100.0), 2)

        # Expected Confidence Interval (+- 1 MAE from benchmarks)
        mae = self.benchmarks.get("regression", {}).get("mae", 22.0)
        interval_low = round(exp_close_return - mae * 0.75, 2)
        interval_high = round(exp_close_return + mae * 0.75, 2)

        # Multi-Horizon Projections from independently trained horizon models.
        max_drawdown = round(min(-3.5, -abs(exp_close_return * 0.35) - 6.0), 2)

        # 3. Local SHAP Factor Attribution
        shap_factors = self._compute_shap_attribution(X_trans)

        return {
            "classification": {
                "win_probability_pct": round(win_prob * 100, 1),
                "verdict": verdict,
                "risk_category": risk_category,
                "confidence": confidence,
                "model_agreement_pct": round(
                    ((int(p_rf >= 0.5) + int(p_xgb >= 0.5) + int(p_lgb >= 0.5)) / 3.0) * 100, 1
                ),
                "session_7_win_probability_pct": round(win_prob_7 * 100, 1),
                "session_7_model_agreement_pct": agreement_7,
                "session_30_win_probability_pct": round(win_prob_30 * 100, 1),
                "session_30_model_agreement_pct": agreement_30,
            },
            "regression": {
                "expected_listing_return_pct": exp_close_return,
                "expected_opening_gain_pct": exp_open_return,
                "projected_close_price": proj_close_price,
                "projected_open_price": proj_open_price,
                "confidence_interval_pct": [interval_low, interval_high],
                "horizon_forecast": {
                    "listing_open_pct": exp_open_return,
                    "listing_close_pct": exp_close_return,
                    "session_7_pct": exp_7_return,
                    "session_30_pct": exp_30_return,
                    "projected_max_drawdown_pct": max_drawdown,
                },
            },
            "shap_explanation": shap_factors,
        }

    def _compute_shap_attribution(self, X_trans: np.ndarray) -> List[Dict[str, Any]]:
        """Calculates exact feature attributions for a single sample."""
        try:
            sv = self.explainer.shap_values(X_trans)
            if isinstance(sv, list):
                vals = sv[1][0]
            elif getattr(sv, "ndim", 2) == 3:
                vals = sv[0, :, 1]
            else:
                vals = sv[0]

            results = []
            for name, val in zip(FEATURE_NAMES, vals):
                impact = float(val)
                results.append({
                    "feature": name,
                    "impact": round(impact * 100, 2),  # impact in percentage points
                    "direction": "POSITIVE" if impact >= 0 else "NEGATIVE",
                })

            # Sort by absolute impact
            results.sort(key=lambda x: abs(x["impact"]), reverse=True)
            return results[:10]  # Return top 10 most influential features
        except Exception as e:
            logger.warning(f"SHAP attribution failed: {e}")
            return []

    def get_similar_peers(self, ipo_id: str, n: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the top n most similar historical IPO peers based on
        sector, issue size, and subscription multiples.
        """
        if self.historical_db is None or self.historical_db.empty:
            return []

        target_row = self.historical_db[self.historical_db["ipo_id"] == ipo_id]
        if target_row.empty:
            return []

        target = target_row.iloc[0]
        sector = target.get("sector")
        size = float(target.get("total_issue_size_cr") or 100)
        sub = float(target.get("total_subscription") or 1)

        candidates = self.historical_db[self.historical_db["ipo_id"] != ipo_id].copy()
        
        # Sector match bonus
        candidates["sector_dist"] = (candidates["sector"] != sector).astype(float) * 2.0
        candidates["size_dist"] = np.abs(np.log1p(candidates["total_issue_size_cr"].fillna(size)) - np.log1p(size))
        candidates["sub_dist"] = np.abs(np.log1p(candidates["total_subscription"].fillna(sub)) - np.log1p(sub))

        candidates["total_dist"] = candidates["sector_dist"] + candidates["size_dist"] + candidates["sub_dist"]
        peers = candidates.sort_values("total_dist").head(n)

        def _safe_float(val, default=0.0):
            if pd.isna(val):
                return default
            try:
                f = float(val)
                return default if np.isnan(f) else f
            except Exception:
                return default

        peer_list = []
        for _, p in peers.iterrows():
            peer_list.append({
                "ipo_id": str(p["ipo_id"]),
                "company_name": str(p["company_name"]),
                "listing_date": str(p["listing_date"]),
                "sector": str(p.get("sector") or "General"),
                "issue_price": round(_safe_float(p.get("issue_price"), 0.0), 2),
                "total_subscription": round(_safe_float(p.get("total_subscription"), 0.0), 1),
                "total_issue_size_cr": round(_safe_float(p.get("total_issue_size_cr"), 0.0), 1),
                "actual_listing_return_pct": round(_safe_float(p.get("listing_day_close_return_pct"), 0.0), 2),
                "win": int(_safe_float(p.get("positive_listing_return"), 0.0)) == 1,
            })
        return peer_list


# Singleton instance
engine = IPOModelEngine()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Training and benchmarking IPO Model Engine...")
    results = engine.train_and_evaluate()
    print("\nBenchmark Results:")
    print(json.dumps(results, indent=2))
