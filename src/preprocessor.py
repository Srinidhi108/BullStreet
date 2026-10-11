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
        X_num = X[self.num_cols].copy()
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
        X_num = X[self.num_cols].copy()
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
