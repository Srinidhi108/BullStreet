"""
Outcome enrichment worker that fetches and caches session outcomes
in parallel for all valid symbols without blocking the main workflow.
"""
import os
import logging
import concurrent.futures
import pandas as pd
import numpy as np
from src.outcome_manager import OutcomeManager

logger = logging.getLogger(__name__)

def enrich_outcomes_dataset(clean_csv_path: str = "data/processed/ipo_master_clean.csv", max_workers: int = 8, limit: int = None):
    if not os.path.exists(clean_csv_path):
        logger.error(f"{clean_csv_path} does not exist.")
        return
        
    df = pd.read_csv(clean_csv_path)
    mgr = OutcomeManager("data/raw")
    
    rows_to_process = df[df["company_symbol"].notna() & (df["company_symbol"] != "")].copy()
    if limit:
        rows_to_process = rows_to_process.head(limit)
        
    logger.info(f"Enriching session outcomes for {len(rows_to_process)} symbols...")
    
    def process_row(idx_row):
        idx, row = idx_row
        sym = row["company_symbol"]
        ldate = pd.to_datetime(row["listing_date"])
        iprice = float(row["issue_price"])
        outcomes = mgr.calculate_session_outcomes(sym, ldate, iprice)
        return idx, outcomes

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_row, (idx, row)) for idx, row in rows_to_process.iterrows()]
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                logger.debug(f"Row enrichment error: {e}")
                
    # Update dataframe
    for idx, outcomes in results:
        for k, v in outcomes.items():
            if k in df.columns and pd.notna(v):
                df.at[idx, k] = v
                
    df.to_csv(clean_csv_path, index=False)
    logger.info(f"Updated {clean_csv_path} with session outcomes. Non-null return_7th_session_pct: {df['return_7th_session_pct'].notna().sum()}, return_30th_session_pct: {df['return_30th_session_pct'].notna().sum()}")
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    enrich_outcomes_dataset()
