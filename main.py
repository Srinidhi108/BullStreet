"""
IPO Intelligence Platform - Master Pipeline Orchestrator.

Executes the end-to-end dataset engineering pipeline:
1. Data Collection -> data/raw/ipo_master_raw.csv
2. Normalization & Cleaning -> data/processed/ipo_master_clean.csv
3. Chronological Splitting -> data/splits/{train,validation,test}.csv + split_manifest.csv
4. Reporting -> reports/
5. Leakage Auditing -> reports/leakage_audit_report.md
"""
import sys
import os
import argparse
import logging
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.collector import collect_raw_data
from src.cleaner import build_clean_master
from src.splitter import create_chronological_splits
from src.reporter import generate_all_reports
from src.leakage_auditor import run_leakage_audit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("IPO-Pipeline")

def run_pipeline(
    data_dir: str = "data",
    reports_dir: str = "reports",
    stage: str = "all",
    fetch_extended: bool = False
):
    logger.info(f"=== Starting IPO Intelligence Pipeline (Stage: {stage}) ===")
    
    # Stage 1: Collection
    if stage in ["all", "collect"]:
        logger.info("--> Executing Stage 1: Raw Data Collection...")
        collect_raw_data(data_dir=data_dir)
        
    # Stage 2: Cleaning & Target Engineering
    if stage in ["all", "clean"]:
        logger.info("--> Executing Stage 2: Data Cleaning & Feature Engineering...")
        build_clean_master(data_dir=data_dir, fetch_extended_sessions=fetch_extended)
        
    # Stage 3: Chronological Splitting
    if stage in ["all", "split"]:
        logger.info("--> Executing Stage 3: Chronological Dataset Splitting...")
        create_chronological_splits(
            clean_csv_path=os.path.join(data_dir, "processed", "ipo_master_clean.csv"),
            splits_dir=os.path.join(data_dir, "splits"),
            manifest_path="split_manifest.csv"
        )
        
    # Stage 4: Reporting
    if stage in ["all", "report"]:
        logger.info("--> Executing Stage 4: Generating Markdown Quality & Coverage Reports...")
        generate_all_reports(data_dir=data_dir, reports_dir=reports_dir)
        
    # Stage 5: Leakage Audit
    if stage in ["all", "audit"]:
        logger.info("--> Executing Stage 5: Running Automated Leakage Audit...")
        passed = run_leakage_audit(
            clean_csv_path=os.path.join(data_dir, "processed", "ipo_master_clean.csv"),
            splits_dir=os.path.join(data_dir, "splits"),
            report_path=os.path.join(reports_dir, "leakage_audit_report.md")
        )
        if not passed:
            logger.error("Leakage audit FAILED. Dataset marked as NOT READY.")
            sys.exit(1)
            
    logger.info("=== Pipeline Execution Successfully Completed! ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPO Intelligence Dataset Engineering Pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "collect", "clean", "split", "report", "audit"],
        help="Pipeline stage to execute (supports rerun of individual stages)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing raw, processed, and split data"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory to store markdown reports"
    )
    parser.add_argument(
        "--fetch-extended",
        action="store_true",
        help="Fetch extended 7th and 30th session outcomes from yfinance"
    )
    args = parser.parse_args()
    
    run_pipeline(
        data_dir=args.data_dir,
        reports_dir=args.reports_dir,
        stage=args.stage,
        fetch_extended=args.fetch_extended
    )
