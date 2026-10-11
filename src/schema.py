"""
Data models and schemas for IPO Intelligence Platform.
"""
from dataclasses import dataclass
from typing import Optional

# Required columns for the clean master dataset
CLEAN_MASTER_COLUMNS = [
    # Primary Identifiers & Metadata
    "ipo_id",
    "company_name",
    "company_symbol",
    "listing_segment",      # Mainboard / SME
    "exchange",            # NSE, BSE, or NSE, BSE
    "ipo_open_date",
    "ipo_close_date",
    "listing_date",
    
    # Issue Structure Features (Available before listing)
    "issue_price_min",
    "issue_price_max",
    "issue_price",
    "total_issue_size_cr",
    "fresh_issue_amount_cr",
    "ofs_amount_cr",
    "fresh_issue_pct",
    "ofs_pct",
    "lot_size",
    "face_value",
    "sector",
    "industry",
    
    # Subscription Features (Available at close before listing)
    "qib_subscription",
    "nii_subscription",
    "retail_subscription",
    "total_subscription",
    
    # Pre-IPO Financial Attributes (Audited prior to prediction timestamp)
    "revenue_cr",
    "revenue_growth_pct",
    "pat_cr",
    "pat_growth_pct",
    "pat_margin_pct",
    "total_assets_cr",
    "total_borrowings_cr",
    "return_on_assets_pct",
    "debt_to_assets_ratio",
    "profitable_before_ipo",
    
    # Market & Macro Features (Up to subscription close / prediction timestamp)
    "nifty_return_5d",
    "nifty_return_20d",
    "nifty_return_60d",
    "nifty_volatility_20d",
    "india_vix",
    "usd_inr_rate",
    
    # Provenance and Validation Metadata
    "source_url",
    "verification_status",
    
    # Historical Outcomes & Targets
    "listing_open_price",
    "listing_day_high",
    "listing_day_low",
    "listing_day_close_price",
    "close_price_7th_session",
    "close_price_30th_session",
    "max_drawdown_30_sessions_pct",
    
    # Evaluated Targets
    "listing_day_close_return_pct",  # Primary regression target
    "positive_listing_return",        # Primary classification target (1 if return > 0 else 0)
    "listing_opening_gain_pct",      # Secondary target: open return
    "return_7th_session_pct",         # Secondary target: 7th session return
    "return_30th_session_pct",        # Secondary target: 30th session return
    "positive_7_session_return",     # Secondary target: positive 7-session return
    "positive_30_session_return"     # Secondary target: positive 30-session return
]
