"""
Collection module to assemble raw Indian IPO data from official/verified repositories
into data/raw/ipo_master_raw.csv.
"""
import os
import re
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

HF_MAINBOARD_URL = "https://huggingface.co/datasets/sohomghosh/Indian_IPO_datasets/resolve/main/ipo_mainline_final_data_v18.xlsx"
HF_SME_URL = "https://huggingface.co/datasets/sohomghosh/Indian_IPO_datasets/resolve/main/SME_data_final_v11.xlsx"

def download_file(url: str, dest_path: str) -> None:
    """Download a file with streaming and error handling."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        logger.info(f"File {dest_path} already exists. Skipping download.")
        return
    logger.info(f"Downloading {url} to {dest_path}...")
    headers = {"User-Agent": "IPO-Intelligence-Platform/1.0 (Research/Academic)"}
    resp = requests.get(url, headers=headers, stream=True, timeout=60)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
    logger.info(f"Downloaded {dest_path} successfully ({os.path.getsize(dest_path)} bytes).")

def collect_raw_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Downloads raw seed files (if not present) and constructs ipo_master_raw.csv.
    Preserves raw data without overwriting or destructive cleaning.
    """
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    mb_raw_path = os.path.join(raw_dir, "seed_mainboard.xlsx")
    sme_raw_path = os.path.join(raw_dir, "seed_sme.xlsx")
    raw_csv_path = os.path.join(raw_dir, "ipo_master_raw.csv")
    
    # 1. Download seed records if absent
    download_file(HF_MAINBOARD_URL, mb_raw_path)
    download_file(HF_SME_URL, sme_raw_path)
    
    # 2. Read raw datasets
    df_mb = pd.read_excel(mb_raw_path)
    df_sme = pd.read_excel(sme_raw_path)
    
    df_mb["raw_source_segment"] = "Mainboard"
    df_sme["raw_source_segment"] = "SME"
    df_mb["retrieved_at"] = datetime.now().isoformat()
    df_sme["retrieved_at"] = datetime.now().isoformat()
    df_mb["source_origin"] = HF_MAINBOARD_URL
    df_sme["source_origin"] = HF_SME_URL
    
    # Align and concatenate all raw columns
    df_raw = pd.concat([df_mb, df_sme], ignore_index=True, sort=False)
    
    # Assign unique raw issue tracking ID
    df_raw.insert(0, "raw_record_id", [f"RAW_{i+1:05d}" for i in range(len(df_raw))])
    
    # Save unedited raw master CSV
    df_raw.to_csv(raw_csv_path, index=False)
    logger.info(f"Created data/raw/ipo_master_raw.csv with {len(df_raw)} records ({len(df_mb)} MB, {len(df_sme)} SME).")
    return df_raw

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collect_raw_data()
