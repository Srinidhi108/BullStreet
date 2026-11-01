import os
import pytest
import pandas as pd
import numpy as np

from src.preprocessor import (
    LeakageFreePreprocessor,
    prepare_modeling_dataset,
    NUMERICAL_FEATURE_COLS,
    CATEGORICAL_FEATURE_COLS
)
from src.preprocessor import HORIZON_FEATURE_COLS
from src.leakage_auditor import run_leakage_audit, TARGET_COLUMNS

def test_seven_csv_files_exist():
    expected_files = [
        "data/raw/ipo_master_raw.csv",
        "data/processed/ipo_master_clean.csv",
        "data/splits/train.csv",
        "data/splits/validation.csv",
        "data/splits/test.csv",
        "data_dictionary.csv",
        "split_manifest.csv"
    ]
    for f in expected_files:
        assert os.path.exists(f), f"Required file {f} does not exist"
        assert os.path.getsize(f) > 0, f"Required file {f} is empty"

def test_reports_exist():
    reports = [
        "reports/source_coverage_report.md",
        "reports/data_quality_report.md",
        "reports/split_report.md",
        "reports/leakage_audit_report.md"
    ]
    for r in reports:
        assert os.path.exists(r), f"Report {r} does not exist"
        assert os.path.getsize(r) > 0, f"Report {r} is empty"

def test_target_calculation_formulas():
    df = pd.read_csv("data/processed/ipo_master_clean.csv")
    
    # Check primary regression target: listing_day_close_return_pct
    expected_ret = ((df["listing_day_close_price"] - df["issue_price"]) / df["issue_price"]) * 100
    np.testing.assert_allclose(df["listing_day_close_return_pct"], expected_ret, atol=1e-2)
    
    # Check primary classification target: positive_listing_return
    expected_pos = (df["listing_day_close_return_pct"] > 0).astype(int)
    assert (df["positive_listing_return"] == expected_pos).all()

def test_deduplication_and_unique_ids():
    df = pd.read_csv("data/processed/ipo_master_clean.csv")
    assert df["ipo_id"].is_unique, "ipo_id must be unique across all records"
    assert df["company_name"].str.lower().duplicated().sum() == 0, "Company names must not have duplicates"

def test_date_chronology_within_records():
    df = pd.read_csv("data/processed/ipo_master_clean.csv")
    
    # Listing date must be on or after IPO close date
    close_dates = pd.to_datetime(df["ipo_close_date"])
    list_dates = pd.to_datetime(df["listing_date"])
    assert (list_dates >= close_dates).all(), "Listing date must be >= subscription close date"

def test_chronological_splits_integrity():
    train_df = pd.read_csv("data/splits/train.csv")
    val_df = pd.read_csv("data/splits/validation.csv")
    test_df = pd.read_csv("data/splits/test.csv")
    
    # Check chronological monotonicity
    train_max = train_df["listing_date"].max()
    val_min = val_df["listing_date"].min()
    val_max = val_df["listing_date"].max()
    test_min = test_df["listing_date"].min()
    
    assert train_max <= val_min, f"Train max date ({train_max}) > Val min date ({val_min})"
    assert val_max <= test_min, f"Val max date ({val_max}) > Test min date ({test_min})"
    
    # Disjoint IDs
    train_ids = set(train_df["ipo_id"])
    val_ids = set(val_df["ipo_id"])
    test_ids = set(test_df["ipo_id"])
    
    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val IDs overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test IDs overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test IDs overlap!"

def test_no_target_leakage_in_preprocessor():
    feature_cols = NUMERICAL_FEATURE_COLS + CATEGORICAL_FEATURE_COLS
    for target in TARGET_COLUMNS:
        assert target not in feature_cols, f"Target column {target} leaked into feature matrix!"

def test_preprocessor_training_only_fit():
    train_df = pd.read_csv("data/splits/train.csv")
    val_df = pd.read_csv("data/splits/validation.csv")
    
    X_train, y_train, _ = prepare_modeling_dataset(train_df, "classification")
    X_val, y_val, _ = prepare_modeling_dataset(val_df, "classification")
    
    preproc = LeakageFreePreprocessor()
    preproc.fit(X_train)
    
    X_train_trans = preproc.transform(X_train)
    X_val_trans = preproc.transform(X_val)
    
    assert not np.isnan(X_train_trans).any(), "NaN found in transformed training data"
    assert not np.isnan(X_val_trans).any(), "NaN found in transformed validation data"
    assert X_train_trans.shape[1] == X_val_trans.shape[1]

def test_leakage_audit_passes():
    passed = run_leakage_audit()
    assert passed is True, "Dataset failed automated leakage audit"

def test_horizon_feature_profiles_are_valid():
    feature_set = set(NUMERICAL_FEATURE_COLS + CATEGORICAL_FEATURE_COLS)
    for horizon, profile in HORIZON_FEATURE_COLS.items():
        assert set(profile).issubset(feature_set), f"Unknown feature in {horizon} profile"
        assert not set(profile).intersection(set(TARGET_COLUMNS)), f"Target leaked into {horizon} profile"

def test_holdout_horizon_predictions_are_not_degenerate():
    import json
    with open("reports/test_holdout_report.json", "r") as handle:
        report = json.load(handle)
    for horizon in ("7_session", "30_session"):
        metrics = report[horizon]["classification"]
        assert 0.0 < metrics["predicted_positive_rate"] < 0.9
        assert metrics["balanced_accuracy"] > 0.5
