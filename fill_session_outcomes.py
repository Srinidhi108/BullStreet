"""
Fill 7-session and 30-session IPO outcomes in the cleaned dataset.

For each IPO, this script:
  1. Reads the company symbol, listing date, and issue price.
  2. Loads cached Yahoo Finance history or downloads it through yfinance.
  3. Selects the 7th and 30th actual trading-session closes.
  4. Calculates returns relative to the IPO issue price.
  5. Writes only the outcome columns back to the CSV.

Examples:
    python fill_session_outcomes.py --limit 10
    python fill_session_outcomes.py
    python fill_session_outcomes.py --force

The default mode skips rows whose four requested values are already filled.
Use --force to recalculate them.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.outcome_manager import OutcomeManager


DEFAULT_CSV = "data/processed/ipo_master_clean.csv"
OUTCOME_COLUMNS = [
    "close_price_7th_session",
    "close_price_30th_session",
    "return_7th_session_pct",
    "return_30th_session_pct",
]

logger = logging.getLogger("fill_session_outcomes")


def parse_listing_date(value: Any) -> pd.Timestamp | None:
    """Parse the project's date formats without crashing the whole run."""
    if pd.isna(value) or str(value).strip() == "":
        return None

    parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed


def is_complete(row: pd.Series) -> bool:
    """Return True when all four requested outcome values are present."""
    return all(pd.notna(row.get(column)) for column in OUTCOME_COLUMNS)


def process_row(item: Tuple[int, pd.Series]) -> Tuple[int, Dict[str, Any], str | None]:
    """Calculate outcomes for one IPO and return a structured status."""
    index, row = item
    symbol = str(row.get("company_symbol", "")).strip()
    listing_date = parse_listing_date(row.get("listing_date"))

    try:
        issue_price = float(row.get("issue_price"))
    except (TypeError, ValueError):
        issue_price = np.nan

    if not symbol:
        return index, {}, "missing company_symbol"
    if listing_date is None:
        return index, {}, "invalid listing_date"
    if not np.isfinite(issue_price) or issue_price <= 0:
        return index, {}, "invalid issue_price"

    try:
        manager = OutcomeManager("data/raw")
        outcomes = manager.calculate_session_outcomes(
            symbol=symbol,
            listing_date=listing_date,
            issue_price=issue_price,
        )
    except Exception as exc:  # Keep one bad symbol from stopping the batch.
        return index, {}, f"{type(exc).__name__}: {exc}"

    available = [column for column in OUTCOME_COLUMNS if pd.notna(outcomes.get(column))]
    if not available:
        return index, {}, "no usable price history or fewer than 30 sessions"

    return index, outcomes, None


def fill_outcomes(
    csv_path: str = DEFAULT_CSV,
    workers: int = 4,
    limit: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Fill requested outcomes and save the updated dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_columns = [column for column in OUTCOME_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    if not force:
        candidates = df[~df.apply(is_complete, axis=1)]
    else:
        candidates = df

    if limit is not None:
        candidates = candidates.head(limit)

    logger.info("Rows in dataset: %d", len(df))
    logger.info("Rows to process: %d", len(candidates))
    logger.info("Using %d worker(s); cached files are reused when available.", workers)

    success_count = 0
    partial_count = 0
    failure_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        jobs = [executor.submit(process_row, item) for item in candidates.iterrows()]
        for job in concurrent.futures.as_completed(jobs):
            index, outcomes, error = job.result()
            if error:
                failure_count += 1
                logger.warning("Row %s (%s) skipped: %s", index, df.at[index, "company_symbol"], error)
                continue

            for column in OUTCOME_COLUMNS:
                value = outcomes.get(column)
                if pd.notna(value):
                    df.at[index, column] = value

            if all(pd.notna(outcomes.get(column)) for column in OUTCOME_COLUMNS):
                success_count += 1
            else:
                partial_count += 1

    # Write only after processing completes, so an interrupted run does not
    # leave a half-written CSV behind.
    temp_path = f"{csv_path}.tmp"
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, csv_path)

    logger.info("Fully populated rows: %d", success_count)
    logger.info("Partially populated rows: %d", partial_count)
    logger.info("Skipped/failed rows: %d", failure_count)
    logger.info(
        "Final non-null counts: 7th close=%d, 30th close=%d, 7th return=%d, 30th return=%d",
        df["close_price_7th_session"].notna().sum(),
        df["close_price_30th_session"].notna().sum(),
        df["return_7th_session_pct"].notna().sum(),
        df["return_30th_session_pct"].notna().sum(),
    )
    logger.info("Updated %s", csv_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to the cleaned IPO CSV")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent download workers")
    parser.add_argument("--limit", type=int, help="Process only the first N incomplete rows")
    parser.add_argument("--force", action="store_true", help="Recalculate already populated rows")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fill_outcomes(
        csv_path=args.csv,
        workers=args.workers,
        limit=args.limit,
        force=args.force,
    )


if __name__ == "__main__":
    main()
