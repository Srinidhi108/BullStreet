"""
Chronological dataset splitter and manifest generator for IPO Intelligence Platform.
Splits data chronologically by listing_date:
- Train: earliest 70%
- Validation: next 15%
- Test: latest 15%
Guarantees zero data leakage and generates split_manifest.csv.
"""
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def create_chronological_splits(
    clean_csv_path: str = "data/processed/ipo_master_clean.csv",
    splits_dir: str = "data/splits",
    manifest_path: str = "split_manifest.csv",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> dict:
    """
    Creates chronological train, validation, and test datasets and saves split_manifest.csv.
    """
    os.makedirs(splits_dir, exist_ok=True)
    df = pd.read_csv(clean_csv_path)
    
    # Exclude rows without verified primary target for modeling splits
    df_eligible = df[df["listing_day_close_return_pct"].notna() & df["positive_listing_return"].notna()].copy()
    
    # Sort chronologically by listing_date with company_name as secondary tie-breaker
    df_eligible = df_eligible.sort_values(by=["listing_date", "company_name"]).reset_index(drop=True)
    
    n_total = len(df_eligible)
    n_train = int(np.floor(n_total * train_ratio))
    n_val = int(np.floor(n_total * val_ratio))
    n_test = n_total - n_train - n_val
    
    train_df = df_eligible.iloc[:n_train].copy()
    val_df = df_eligible.iloc[n_train : n_train + n_val].copy()
    test_df = df_eligible.iloc[n_train + n_val:].copy()
    
    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"
    
    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "validation.csv")
    test_path = os.path.join(splits_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Saved splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Generate split manifest
    manifest_rows = []
    splits = [("train", train_df), ("validation", val_df), ("test", test_df)]
    now_str = datetime.now().isoformat()
    
    for name, s_df in splits:
        pos_count = int((s_df["positive_listing_return"] == 1).sum())
        neg_count = int((s_df["positive_listing_return"] == 0).sum())
        missing_count = int(s_df["positive_listing_return"].isna().sum())
        
        manifest_rows.append({
            "split_name": name,
            "first_listing_date": s_df["listing_date"].min(),
            "last_listing_date": s_df["listing_date"].max(),
            "number_of_ipos": len(s_df),
            "positive_target_count": pos_count,
            "negative_target_count": neg_count,
            "missing_target_count": missing_count,
            "feature_count": len([c for c in s_df.columns if c not in ["split"]]),
            "dataset_version": "v1.0.0",
            "creation_timestamp": now_str,
            "ipo_ids": ";".join(s_df["ipo_id"].tolist())
        })
        
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False)
    logger.info(f"Saved split manifest to {manifest_path}")
    
    return {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
        "manifest": manifest_df
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_chronological_splits()
