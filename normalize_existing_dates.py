"""One-time repair for legacy clean-master date strings.

The original CSV was written with ambiguous DD-MM-YY values. This preserves
all rows/targets and only rewrites the three date columns as ISO dates.
"""
import pandas as pd

PATHS = [
    "data/processed/ipo_master_clean.csv",
    "data/splits/train.csv",
    "data/splits/validation.csv",
    "data/splits/test.csv",
]

for path in PATHS:
    df = pd.read_csv(path)
    for column in ("ipo_open_date", "ipo_close_date", "listing_date"):
        if column in df:
            parsed = pd.to_datetime(df[column], dayfirst=True, errors="coerce")
            df[column] = parsed.dt.strftime("%Y-%m-%d")
    df.to_csv(path, index=False)
    print(f"normalized {path}")
