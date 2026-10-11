# Data Quality & Completeness Report

**Generated:** 2026-10-11 15:25:15  
**Total Records Evaluated:** 974  
**Total Attributes:** 56  

## 1. Column-Level Completeness and Missingness

| Column Name | Non-Null Count | Missing Count | Missingness (%) | Data Category |
|---|---|---|---|---|
| `ipo_id` | 974 | 0 | 0.0% | Feature / Metadata |
| `company_name` | 974 | 0 | 0.0% | Feature / Metadata |
| `company_symbol` | 689 | 285 | 29.3% | Feature / Metadata |
| `listing_segment` | 974 | 0 | 0.0% | Feature / Metadata |
| `exchange` | 974 | 0 | 0.0% | Feature / Metadata |
| `ipo_open_date` | 974 | 0 | 0.0% | Feature / Metadata |
| `ipo_close_date` | 974 | 0 | 0.0% | Feature / Metadata |
| `listing_date` | 974 | 0 | 0.0% | Feature / Metadata |
| `issue_price_min` | 442 | 532 | 54.6% | Feature / Metadata |
| `issue_price_max` | 442 | 532 | 54.6% | Feature / Metadata |
| `issue_price` | 974 | 0 | 0.0% | Feature / Metadata |
| `total_issue_size_cr` | 974 | 0 | 0.0% | Feature / Metadata |
| `fresh_issue_amount_cr` | 519 | 455 | 46.7% | Feature / Metadata |
| `ofs_amount_cr` | 285 | 689 | 70.7% | Feature / Metadata |
| `fresh_issue_pct` | 519 | 455 | 46.7% | Feature / Metadata |
| `ofs_pct` | 285 | 689 | 70.7% | Feature / Metadata |
| `lot_size` | 968 | 6 | 0.6% | Feature / Metadata |
| `face_value` | 974 | 0 | 0.0% | Feature / Metadata |
| `sector` | 923 | 51 | 5.2% | Feature / Metadata |
| `industry` | 923 | 51 | 5.2% | Feature / Metadata |
| `qib_subscription` | 392 | 582 | 59.8% | Feature / Metadata |
| `nii_subscription` | 433 | 541 | 55.5% | Feature / Metadata |
| `retail_subscription` | 864 | 110 | 11.3% | Feature / Metadata |
| `total_subscription` | 865 | 109 | 11.2% | Feature / Metadata |
| `revenue_cr` | 614 | 360 | 37.0% | Feature / Metadata |
| `revenue_growth_pct` | 598 | 376 | 38.6% | Feature / Metadata |
| `pat_cr` | 635 | 339 | 34.8% | Feature / Metadata |
| `pat_growth_pct` | 620 | 354 | 36.3% | Feature / Metadata |
| `pat_margin_pct` | 612 | 362 | 37.2% | Feature / Metadata |
| `total_assets_cr` | 616 | 358 | 36.8% | Feature / Metadata |
| `total_borrowings_cr` | 122 | 852 | 87.5% | Feature / Metadata |
| `return_on_assets_pct` | 616 | 358 | 36.8% | Target |
| `debt_to_assets_ratio` | 122 | 852 | 87.5% | Feature / Metadata |
| `profitable_before_ipo` | 635 | 339 | 34.8% | Feature / Metadata |
| `nifty_return_5d` | 974 | 0 | 0.0% | Target |
| `nifty_return_20d` | 974 | 0 | 0.0% | Target |
| `nifty_return_60d` | 974 | 0 | 0.0% | Target |
| `nifty_volatility_20d` | 974 | 0 | 0.0% | Feature / Metadata |
| `india_vix` | 974 | 0 | 0.0% | Feature / Metadata |
| `usd_inr_rate` | 974 | 0 | 0.0% | Feature / Metadata |
| `source_url` | 974 | 0 | 0.0% | Feature / Metadata |
| `verification_status` | 974 | 0 | 0.0% | Feature / Metadata |
| `listing_open_price` | 974 | 0 | 0.0% | Target |
| `listing_day_high` | 974 | 0 | 0.0% | Feature / Metadata |
| `listing_day_low` | 974 | 0 | 0.0% | Feature / Metadata |
| `listing_day_close_price` | 974 | 0 | 0.0% | Target |
| `close_price_7th_session` | 0 | 974 | 100.0% | Target |
| `close_price_30th_session` | 0 | 974 | 100.0% | Target |
| `max_drawdown_30_sessions_pct` | 0 | 974 | 100.0% | Feature / Metadata |
| `listing_day_close_return_pct` | 974 | 0 | 0.0% | Target |
| `positive_listing_return` | 974 | 0 | 0.0% | Target |
| `listing_opening_gain_pct` | 974 | 0 | 0.0% | Feature / Metadata |
| `return_7th_session_pct` | 0 | 974 | 100.0% | Target |
| `return_30th_session_pct` | 0 | 974 | 100.0% | Target |
| `positive_7_session_return` | 0 | 974 | 100.0% | Target |
| `positive_30_session_return` | 0 | 974 | 100.0% | Target |

## 2. Automated Quality Validations

- **Zero Negative Prices:** Final issue prices and listing day prices are strictly positive (`issue_price > 0`).
- **Deduplication Check:** Duplicate check on company names and ISIN/symbol yields 0 duplicates across the unified master.
- **Date Integrity Check:** `listing_date >= ipo_close_date >= ipo_open_date` verified with 100% adherence.
- **Target Boundary Validation:** Primary classification target `positive_listing_return` has 0% missingness and perfectly aligns with `listing_day_close_return_pct > 0`.

## 3. Missing Value Handling Policy

- **Primary Modeling Targets:** Strictly excluded if missing; in our scope, all 974 records have fully verified primary targets.
- **Numerical Predictors:** Preprocessor fits median imputation strictly on the training split (`train.csv`) and applies it downstream.
- **Categorical Predictors:** Frequency encoding with unseen-category and missingness fallback handling.
- **Financial Fields:** Unreported/missing DRHP financial values are imputed without data fabrication.
