# Leakage Audit Report

**Audit Status:** `PASSED (DATASET READY)`  
**Execution Timestamp:** 2026-10-25 14:45:39  
**Master Dataset Records:** 974  
**Partition Sizes:** Train=681, Validation=146, Test=147  

## Summary of Leakage Checks

| ID | Audit Requirement | Result | Observations |
|---|---|---|---|
| L01 | Target Separation from Feature Matrix | **PASSED** | All target columns strictly isolated from predictor features. |
| L02 | No Post-Listing Attributes in Predictors | **PASSED** | Predictor features contain zero post-listing metrics. |
| L03 | Temporal Chronology (Listing >= Subscription Close) | **PASSED** | All 974 records verified: listing_date >= ipo_close_date. |
| L04 | Chronological Split Boundaries | **PASSED** | Train max (2022-10-06) <= Val min (2022-10-10) and Val max (2023-07-21) <= Test min (2023-07-26). |
| L05 | Partition Identity Isolation (Disjoint IPO IDs) | **PASSED** | Zero ID overlap across partitions. Counts: Train=681, Val=146, Test=147. |
| L06 | Target Formula Mathematical Consistency | **PASSED** | Max return deviation = 0.0050%, binary mismatch count = 0. |
| L07 | Independent Training-Only Preprocessing Fit | **PASSED** | Imputation and scaling parameters strictly derived from training partition statistics. |
| L08 | Dual-Exchange Deduplication Isolation | **PASSED** | Zero duplicate issuer listings found (0 duplicates). |

## Prediction Timestamp Enforcement

The prediction timestamp for every IPO is defined strictly as the **subscription close date** (`ipo_close_date`), before trading commences on `listing_date`.
- All predictor variables (issue structure, investor bidding multiples, audited financial statements, and trailing macroeconomic indicators) use information available on or before subscription close.
- No market price from the listing day or secondary market sessions is accessible to feature engineering or model training.

## Chronological Split Validation

- **Train Partition:** 2015-03-19 to 2022-10-06 (681 IPOs)
- **Validation Partition:** 2022-10-10 to 2023-07-21 (146 IPOs)
- **Test Partition:** 2023-07-26 to 2024-01-03 (147 IPOs)

All partition boundaries are strictly ordered chronologically with zero cross-split record contamination.
