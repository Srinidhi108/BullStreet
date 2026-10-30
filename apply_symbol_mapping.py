"""Apply verified ticker mappings to the cleaned IPO dataset."""

from __future__ import annotations

import argparse
import os

import pandas as pd


DEFAULT_MASTER = "data/processed/ipo_master_clean.csv"
DEFAULT_MAPPING = "data/processed/symbol_mapping.csv"


def apply_mapping(master_path: str, mapping_path: str, dry_run: bool = False) -> None:
    master = pd.read_csv(master_path)
    mapping = pd.read_csv(mapping_path).fillna("")

    required = {"ipo_id", "mapped_symbol", "mapping_status"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Mapping file is missing columns: {sorted(missing)}")

    verified = mapping[
        mapping["mapped_symbol"].astype(str).str.strip().ne("")
        & mapping["mapping_status"].astype(str).str.upper().isin(["VERIFIED", "CONFIRMED"])
    ].copy()

    updates = 0
    unknown_ids = []
    for _, row in verified.iterrows():
        matches = master.index[master["ipo_id"].astype(str) == str(row["ipo_id"])]
        if len(matches) == 0:
            unknown_ids.append(str(row["ipo_id"]))
            continue
        master.loc[matches, "company_symbol"] = str(row["mapped_symbol"]).strip().upper()
        updates += len(matches)

    print(f"Verified mappings found: {len(verified)}")
    print(f"Master rows to update: {updates}")
    if unknown_ids:
        print(f"Unknown IPO IDs skipped: {', '.join(unknown_ids)}")

    if dry_run:
        print("Dry run only; master CSV was not changed.")
        return

    temp_path = f"{master_path}.tmp"
    master.to_csv(temp_path, index=False)
    os.replace(temp_path, master_path)
    print(f"Updated {master_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", default=DEFAULT_MASTER)
    parser.add_argument("--mapping", default=DEFAULT_MAPPING)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_mapping(args.master, args.mapping, args.dry_run)
