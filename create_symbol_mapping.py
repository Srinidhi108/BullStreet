"""Create a reviewable symbol-mapping template for IPOs missing tickers."""

from __future__ import annotations

import argparse
import os

import pandas as pd


DEFAULT_INPUT = "data/processed/ipo_master_clean.csv"
DEFAULT_OUTPUT = "data/processed/symbol_mapping.csv"


def create_mapping_file(input_path: str = DEFAULT_INPUT, output_path: str = DEFAULT_OUTPUT) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    missing = df[
        df["company_symbol"].isna()
        | df["company_symbol"].astype(str).str.strip().isin(["", "nan", "None"])
    ].copy()

    mapping = missing[
        [
            "ipo_id",
            "company_name",
            "company_symbol",
            "listing_date",
            "exchange",
            "listing_segment",
            "source_url",
        ]
    ].copy()

    mapping = mapping.rename(columns={"company_symbol": "current_symbol"})
    mapping["mapped_symbol"] = ""
    mapping["symbol_exchange"] = ""
    mapping["mapping_source"] = ""
    mapping["mapping_status"] = "PENDING"
    mapping["notes"] = ""

    mapping = mapping[
        [
            "ipo_id",
            "company_name",
            "current_symbol",
            "mapped_symbol",
            "symbol_exchange",
            "listing_date",
            "exchange",
            "listing_segment",
            "source_url",
            "mapping_source",
            "mapping_status",
            "notes",
        ]
    ].sort_values(["listing_date", "company_name"], na_position="last")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mapping.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(mapping)} rows.")
    print("Fill mapped_symbol, symbol_exchange, mapping_source, and mapping_status before importing.")
    return mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    create_mapping_file(args.input, args.output)
