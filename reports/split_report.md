# Dataset Partition & Split Report

**Generated:** 2026-10-11 15:25:15  
**Splitting Strategy:** Chronological split by `listing_date` (70% Train / 15% Validation / 15% Test) with company name tie-breaking.

## 1. Split Partition Summary

| Partition | First Listing Date | Last Listing Date | Number of IPOs | Positive Return (1) | Negative Return (0) | Win Rate (%) |
|---|---|---|---|---|---|---|
| **Train** | 2015-03-19 | 2022-10-06 | 681 | 486 | 195 | 71.4% |
| **Validation** | 2022-10-10 | 2023-07-21 | 146 | 114 | 32 | 78.1% |
| **Test** | 2023-07-26 | 2024-01-03 | 147 | 126 | 21 | 85.7% |

## 2. Segment Distribution across Splits

| Partition | Mainboard IPOs | SME IPOs | Mainboard Share (%) |
|---|---|---|---|
| Train | 220 | 461 | 32.3% |
| Validation | 29 | 117 | 19.9% |
| Test | 45 | 102 | 30.6% |

## 3. Strict Chronological Isolation

- **Train to Validation Boundary:** No listing date in the validation split precedes any listing date in the training split.
- **Validation to Test Boundary:** The held-out test split consists solely of the latest IPO listings and remains unaccessed during model feature selection and hyperparameter tuning.
- **Zero Identifier Contamination:** Each IPO ID appears in exactly one partition.
