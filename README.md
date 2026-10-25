# IPO Intelligence Platform

A robust, reproducible dataset engineering pipeline for Indian Initial Public Offerings (IPOs) listed on the National Stock Exchange (NSE) and Bombay Stock Exchange (BSE) from **1 January 2015 to 31 December 2025**.

The pipeline ingests raw public IPO records, standardizes and validates attributes, isolates predictor features available strictly on or before the **subscription close date** (the prediction timestamp), derives primary and secondary prediction targets, creates chronological training, validation, and test partitions, and conducts an automated leakage audit.

---

## 1. Project Directory Structure

Project layout and dataset organization:

```text
af_project/
├── data/
│   ├── raw/
│   │   └── ipo_master_raw.csv         # Raw collected records preserved without modification
│   ├── processed/
│   │   └── ipo_master_clean.csv       # Consolidated, cleaned, validated master dataset (974 IPOs)
│   └── splits/
│       ├── train.csv                  # Chronological training split (earliest 70%, 681 IPOs)
│       ├── validation.csv             # Chronological validation split (next 15%, 146 IPOs)
│       └── test.csv                   # Chronological held-out testing split (latest 15%, 147 IPOs)
├── reports/
│   ├── source_coverage_report.md      # Yearly, segment, exchange, and verification counts
│   ├── data_quality_report.md         # Column-level missingness, validity, and imputation policies
│   ├── split_report.md                # Partition boundaries, row counts, and target win rates
│   └── leakage_audit_report.md        # Comprehensive 8-point automated leakage audit report
├── src/
│   ├── schema.py                      # Master schema definitions and required columns
│   ├── collector.py                   # Ingestion and raw master preservation engine
│   ├── cleaner.py                     # Normalizer, deduplicator, and target calculation engine
│   ├── market_data.py                 # Nifty 50, India VIX, and USD/INR historical series manager
│   ├── outcome_manager.py             # Listing day and post-listing trading session outcome calculator
│   ├── outcome_enricher.py            # Multi-threaded session outcome enricher
│   ├── splitter.py                    # Chronological partitioner and split manifest builder
│   ├── preprocessor.py                # Leakage-free preprocessing (imputation, scaling, encoding)
│   ├── reporter.py                    # Markdown report generation engine
│   └── leakage_auditor.py             # Automated leakage verification engine
├── tests/
│   └── test_pipeline.py               # Comprehensive pytest test suite (formulas, splits, leakage)
├── data_dictionary.csv                # Complete column metadata, definitions, units, and sources
├── split_manifest.csv                 # Split boundaries, row counts, class balances, and IPO ID lists
├── requirements.txt                   # Frozen Python environment dependencies
├── README.md                          # Pipeline documentation and execution guide
└── main.py                            # Central entrypoint orchestrating pipeline stages
```

---

## 2. Core Project Deliverables

1. **`data/raw/ipo_master_raw.csv`**: Unfiltered, unmodified raw records combining Mainboard and SME seed datasets (1,099 raw issue filings).
2. **`data/processed/ipo_master_clean.csv`**: Cleaned, verified, and deduplicated master dataset of 974 Indian IPOs (294 Mainboard, 680 SME) with 56 structured attributes.
3. **`data/splits/train.csv`**: Earliest 70% of chronological issues (681 records: 2015-03-19 to 2022-10-06).
4. **`data/splits/validation.csv`**: Next 15% of chronological issues (146 records: 2022-10-10 to 2023-07-21).
5. **`data/splits/test.csv`**: Latest 15% of chronological issues (147 records: 2023-07-26 to 2024-01-03).
6. **`data_dictionary.csv`**: Exhaustive specification of all 56 columns, their formulas, sources, units, and availability timestamps.
7. **`split_manifest.csv`**: Partition metadata documenting exact date boundaries, target win rates, and semicolon-delimited lists of IPO IDs.

---

## 3. Methodology & Design Principles

### Prediction Timestamp Enforcement
The prediction timestamp is set to **after public subscription closes (`ipo_close_date`) but before secondary market listing begins (`listing_date`)**.
- **Allowed Features:** Issue price band, total capital raised, fresh issue vs. OFS breakdown, lot size, face value, sector/industry, QIB / NII / Retail final subscription multiples, pre-IPO audited financials (Revenue, PAT, Net Worth, Total Assets, Borrowings), trailing NIFTY 50 returns (5-day, 20-day, 60-day), trailing NIFTY volatility, India VIX, and USD/INR exchange rate.
- **Strictly Excluded from Predictors:** Any listing morning opening prices, intraday highs/lows, closing prices, trading volume, or subsequent financial quarters.

### Target Definitions
1. **Primary Regression Target:**  
   $$\text{listing\_day\_close\_return\_pct} = \frac{\text{listing\_day\_close\_price} - \text{issue\_price}}{\text{issue\_price}} \times 100$$
2. **Primary Classification Target:**  
   $$\text{positive\_listing\_return} = \begin{cases} 1 & \text{if } \text{listing\_day\_close\_return\_pct} > 0 \\ 0 & \text{otherwise} \end{cases}$$
3. **Secondary Targets:**
   - $\text{listing\_opening\_gain\_pct}$: Morning listing open return.
   - $\text{return\_7th\_session\_pct}$: Close price after 7 actual trading sessions.
   - $\text{return\_30th\_session\_pct}$: Close price after 30 actual trading sessions.
   - $\text{max\_drawdown\_30\_sessions\_pct}$: Peak-to-trough percentage drawdown over the first 30 trading sessions.

### Leakage-Free Preprocessing (`src/preprocessor.py`)
- Numerical imputation (median) and feature standardization (`StandardScaler`) are fit **strictly on `train.csv`**.
- Categorical encodings (frequency distributions) are derived **only from `train.csv`**.
- The fitted transformer handles unseen categories and preserves test isolation without information leakage.

---

## 4. Setup and Execution

### Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the End-to-End Pipeline
Execute the full pipeline through `main.py`:
```bash
source .venv/bin/activate
python3 main.py
```

### Running Individual Stages
The pipeline supports rerunning specific stages without re-fetching network data:
```bash
# Rerun data cleaning and target calculation only
python3 main.py --stage clean

# Rerun chronological splitting and manifest generation only
python3 main.py --stage split

# Regenerate quality and coverage reports
python3 main.py --stage report

# Run automated leakage audit
python3 main.py --stage audit
```

### Running Unit & Integrity Tests
```bash
source .venv/bin/activate
pytest tests/test_pipeline.py -v
```
All 9 automated unit tests verify formula accuracy, date ordering, disjoint ID splits, and leakage protection.

---

## 5. Machine Learning Modeling Usage

To train models using `train.csv`, `validation.csv`, and `test.csv`:

```python
import pandas as pd
from src.preprocessor import LeakageFreePreprocessor, prepare_modeling_dataset

train_df = pd.read_csv("data/splits/train.csv")
val_df = pd.read_csv("data/splits/validation.csv")
test_df = pd.read_csv("data/splits/test.csv")

# Extract features and target for Binary Classification (Dataset A)
X_train, y_train, ids_train = prepare_modeling_dataset(train_df, task="classification")
X_val, y_val, ids_val = prepare_modeling_dataset(val_df, task="classification")
X_test, y_test, ids_test = prepare_modeling_dataset(test_df, task="classification")

# Fit preprocessor strictly on training set
preprocessor = LeakageFreePreprocessor()
preprocessor.fit(X_train)

# Transform partitions without data leakage
X_train_trans = preprocessor.transform(X_train)
X_val_trans = preprocessor.transform(X_val)
X_test_trans = preprocessor.transform(X_test)
```
