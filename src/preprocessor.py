"""
Preprocessing and modeling dataset preparation without data leakage.
Fits imputers, scalers, and encoders strictly on the training partition.
Provides scikit-learn compatible transformation pipeline and dataset extractors
for Dataset A (classification), B (regression), C (7-session), D (30-session).
"""
import os
import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

NUMERICAL_FEATURE_COLS = [
    "issue_price",
    "total_issue_size_cr",
    "fresh_issue_amount_cr",
    "ofs_amount_cr",
    "fresh_issue_pct",
    "ofs_pct",
    "lot_size",
    "face_value",
    "qib_subscription",
    "nii_subscription",
    "retail_subscription",
    "total_subscription",
    "revenue_cr",
    "revenue_growth_pct",
    "pat_cr",
    "pat_growth_pct",
    "pat_margin_pct",
    "total_assets_cr",
    "total_borrowings_cr",
    "return_on_assets_pct",
    "debt_to_assets_ratio",
    "profitable_before_ipo",
    "nifty_return_5d",
    "nifty_return_20d",
    "nifty_return_60d",
    "nifty_volatility_20d",
    "india_vix",
    "usd_inr_rate",
]

CATEGORICAL_FEATURE_COLS = [
    "listing_segment",
    "exchange",
    "sector"
]

# Forecast horizons emphasize different information available before listing.
# These are deliberately conservative profiles; each still retains the core
# issue and subscription signals needed for a usable fallback.
HORIZON_FEATURE_COLS = {
    "listing": NUMERICAL_FEATURE_COLS + CATEGORICAL_FEATURE_COLS,
    "7": [
        "issue_price", "total_issue_size_cr", "fresh_issue_pct", "ofs_pct",
        "qib_subscription", "nii_subscription", "retail_subscription",
        "total_subscription", "profitable_before_ipo", "nifty_return_5d",
        "nifty_return_20d", "nifty_volatility_20d", "india_vix", "usd_inr_rate",
        "listing_segment", "exchange", "sector",
    ],
    "30": NUMERICAL_FEATURE_COLS + CATEGORICAL_FEATURE_COLS,
}

class LeakageFreePreprocessor(BaseEstimator, TransformerMixin):
    """
    Fits imputers, scalers, and frequency encodings on training data only.
    Never uses validation or test information.
    Handles completely missing features in the training partition with constant fallback.
    """
    def __init__(self, num_cols=None, cat_cols=None):
        self.num_cols = num_cols or NUMERICAL_FEATURE_COLS
        self.cat_cols = cat_cols or CATEGORICAL_FEATURE_COLS
        self.num_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()
        self.cat_encodings = {}
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y=None):
        # 1. Fit numerical imputer and scaler on train
        X_num = self._engineer_numeric(X[self.num_cols].copy())
        X_num_imp = self.num_imputer.fit_transform(X_num)
        
        # In case some feature had 0 observations in train, imputer fills with 0
        X_num_imp = np.nan_to_num(X_num_imp, nan=0.0)
        self.scaler.fit(X_num_imp)

        # 2. Fit categorical encodings (frequency encoding on train)
        for col in self.cat_cols:
            if col in X.columns:
                counts = X[col].fillna("Unknown").value_counts(normalize=True).to_dict()
                self.cat_encodings[col] = counts

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise ValueError("Preprocessor must be fitted before transforming data.")
        
        # Transform numerical
        X_num = self._engineer_numeric(X[self.num_cols].copy())
        X_num_imp = self.num_imputer.transform(X_num)
        X_num_imp = np.nan_to_num(X_num_imp, nan=0.0)
        X_num_scaled = self.scaler.transform(X_num_imp)

        # Transform categorical to frequency-encoded floats
        cat_arrays = []
        for col in self.cat_cols:
            if col in X.columns:
                encoding = self.cat_encodings.get(col, {})
                s = X[col].fillna("Unknown").map(encoding).fillna(0.0).values.reshape(-1, 1)
                cat_arrays.append(s)
            else:
                cat_arrays.append(np.zeros((len(X), 1)))

        X_cat = np.hstack(cat_arrays) if cat_arrays else np.empty((len(X), 0))
        return np.hstack([X_num_scaled, X_cat])

    @staticmethod
    def _engineer_numeric(X: pd.DataFrame) -> pd.DataFrame:
        """Build leakage-safe, pre-listing features for small/regime-sensitive data."""
        out = X.copy()

        for name in (
            "total_issue_size_cr", "fresh_issue_amount_cr", "ofs_amount_cr",
            "revenue_cr", "pat_cr", "total_assets_cr", "total_borrowings_cr",
            "qib_subscription", "nii_subscription", "retail_subscription",
            "total_subscription",
        ):
            if name in X:
                values = pd.to_numeric(X[name], errors="coerce")
                out[f"log_{name}"] = np.log1p(np.clip(values, 0, None))

        def safe_div(a, b, name):
            if a in X and b in X:
                den = pd.to_numeric(X[b], errors="coerce").replace(0, np.nan)
                out[name] = pd.to_numeric(X[a], errors="coerce") / den

        safe_div("qib_subscription", "total_subscription", "qib_share_of_subscription")
        safe_div("nii_subscription", "total_subscription", "nii_share_of_subscription")
        safe_div("retail_subscription", "total_subscription", "retail_share_of_subscription")
        safe_div("pat_cr", "revenue_cr", "pat_to_revenue_ratio")
        safe_div("total_borrowings_cr", "total_assets_cr", "borrowings_to_assets_ratio")
        safe_div("total_issue_size_cr", "total_assets_cr", "issue_to_assets_ratio")

        if "qib_subscription" in X and "retail_subscription" in X:
            out["institutional_retail_gap"] = (
                np.log1p(np.clip(pd.to_numeric(X["qib_subscription"], errors="coerce"), 0, None))
                - np.log1p(np.clip(pd.to_numeric(X["retail_subscription"], errors="coerce"), 0, None))
            )
        if "nifty_return_5d" in X and "nifty_return_20d" in X:
            out["market_momentum_acceleration"] = X["nifty_return_5d"] - X["nifty_return_20d"] / 4.0
        if "nifty_volatility_20d" in X and "india_vix" in X:
            out["market_stress_score"] = X["nifty_volatility_20d"] + X["india_vix"]
        if "total_subscription" in X and "retail_subscription" in X:
            out["subscription_dispersion"] = (
                np.log1p(np.clip(pd.to_numeric(X["total_subscription"], errors="coerce"), 0, None))
                - np.log1p(np.clip(pd.to_numeric(X["retail_subscription"], errors="coerce"), 0, None))
            )
        return out.replace([np.inf, -np.inf], np.nan)


def prepare_modeling_dataset(df: pd.DataFrame, task: str = "classification"):
    """
    Extracts features and target for a given modeling task:
    - 'classification': Dataset A (positive_listing_return)
    - 'regression': Dataset B (listing_day_close_return_pct)
    - '7_session': Dataset C (return_7th_session_pct or positive_7_session_return)
    - '30_session': Dataset D (return_30th_session_pct or positive_30_session_return)
    """
    feature_cols = NUMERICAL_FEATURE_COLS + CATEGORICAL_FEATURE_COLS
    
    if task == "classification":
        sub = df.dropna(subset=["positive_listing_return"]).copy()
        X = sub[feature_cols]
        y = sub["positive_listing_return"].astype(int)
        return X, y, sub["ipo_id"]
        
    elif task == "regression":
        sub = df.dropna(subset=["listing_day_close_return_pct"]).copy()
        X = sub[feature_cols]
        y = sub["listing_day_close_return_pct"].astype(float)
        return X, y, sub["ipo_id"]
        
    elif task == "7_session":
        sub = df.dropna(subset=["return_7th_session_pct"]).copy()
        X = sub[feature_cols]
        y = sub["return_7th_session_pct"].astype(float)
        return X, y, sub["ipo_id"]
        
    elif task == "30_session":
        sub = df.dropna(subset=["return_30th_session_pct"]).copy()
        X = sub[feature_cols]
        y = sub["return_30th_session_pct"].astype(float)
        return X, y, sub["ipo_id"]
        
    else:
        raise ValueError(f"Unknown task: {task}")
