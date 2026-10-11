"""
Cleaner and normalizer module for IPO Intelligence Platform.
Parses, cleans, validates and standardizes raw IPO data into ipo_master_clean.csv.
"""
import os
import re
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from dateutil import parser

from src.schema import CLEAN_MASTER_COLUMNS
from src.market_data import MarketDataManager
from src.outcome_manager import OutcomeManager

logger = logging.getLogger(__name__)

def parse_date_safe(val):
    if pd.isna(val) or str(val).strip() in ["", ".", "[.]", "None", "nan", "NaT"]:
        return None
    try:
        return parser.parse(str(val)).strftime("%Y-%m-%d")
    except Exception:
        return None

def parse_open_from_range(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if " to " in s:
        parts = s.split(" to ")
        first, second = parts[0].strip(), parts[1].strip()
        try:
            m_year = re.search(r"\b(20\d\d)\b", second)
            if not re.search(r"\b(20\d\d)\b", first) and m_year:
                first = f"{first}, {m_year.group(1)}"
            return parser.parse(first).strftime("%Y-%m-%d")
        except Exception:
            return None
    return parse_date_safe(val)

def parse_float(val):
    if pd.isna(val) or str(val).strip() in ["", ".", "[.]", "None", "nan"]:
        return np.nan
    s = str(val).replace(",", "").strip()
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group(0)) if m else np.nan

def parse_issue_size_cr(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace(",", "")
    m = re.search(r"aggregating up to\s+([\d\.]+)\s*Cr", s, flags=re.I)
    if m:
        return float(m.group(1))
    m2 = re.search(r"([\d\.]+)\s*Cr", s, flags=re.I)
    if m2:
        return float(m2.group(1))
    return parse_float(val)

def parse_price_band(val):
    if pd.isna(val):
        return np.nan, np.nan
    s = str(val).replace(",", "")
    m = re.findall(r"[-+]?\d*\.?\d+", s)
    if len(m) >= 2:
        return float(m[0]), float(m[1])
    elif len(m) == 1:
        return float(m[0]), float(m[0])
    return np.nan, np.nan

def clean_company_name(name):
    if pd.isna(name):
        return ""
    n = re.sub(r"\s+IPO$", "", str(name).strip(), flags=re.I)
    return n.strip()

def extract_latest_subscription(row, is_sme=False):
    max_day = 11 if is_sme else 6
    for day in range(max_day, 0, -1):
        qib_c = f"day_{day}_qib"
        nii_c = f"day_{day}_nii"
        nii_star = f"day_{day}_nii*"
        ret_c = f"day_{day}_retail"
        tot_c = f"day_{day}_total"
        
        has_val = any(c in row and pd.notna(row[c]) for c in [qib_c, nii_c, nii_star, ret_c, tot_c])
        if has_val:
            qib = parse_float(row.get(qib_c, np.nan))
            nii = parse_float(row.get(nii_c, np.nan))
            if np.isnan(nii) and nii_star in row:
                nii = parse_float(row.get(nii_star, np.nan))
            ret = parse_float(row.get(ret_c, np.nan))
            tot = parse_float(row.get(tot_c, np.nan))
            return qib, nii, ret, tot
            
    # Fallback to Total_subscriptions if available
    tot_fallback = parse_float(row.get("Total_subscriptions", np.nan))
    return np.nan, np.nan, np.nan, tot_fallback

def build_clean_master(data_dir: str = "data", fetch_extended_sessions: bool = True) -> pd.DataFrame:
    """
    Cleans raw data, engineers core features and outcomes, and saves ipo_master_clean.csv.
    """
    raw_csv = os.path.join(data_dir, "raw", "ipo_master_raw.csv")
    if not os.path.exists(raw_csv):
        from src.collector import collect_raw_data
        collect_raw_data(data_dir)
        
    df_raw = pd.read_csv(raw_csv, low_memory=False)
    market_mgr = MarketDataManager(os.path.join(data_dir, "raw"))
    outcome_mgr = OutcomeManager(os.path.join(data_dir, "raw"))
    
    clean_records = []
    
    for idx, row in df_raw.iterrows():
        segment = str(row.get("raw_source_segment", "Mainboard"))
        is_sme = (segment == "SME")
        
        # 1. Company Name & Symbol
        comp_name = clean_company_name(row.get("Issuer Company") or row.get("Company Name", ""))
        symbol = str(row.get("NSE_symbol", "")).strip()
        if symbol in ["nan", "None", "."]:
            symbol = ""
            
        # 2. Dates
        list_date = parse_date_safe(row.get("Listing Date"))
        close_date = parse_date_safe(row.get("Close Date"))
        open_date = parse_date_safe(row.get("Open Date")) if not is_sme else parse_open_from_range(row.get("IPO Date"))
        
        # Scope filter: 2015-01-01 to 2025-12-31
        if not list_date or list_date < "2015-01-01" or list_date > "2025-12-31":
            continue
            
        # 3. Exchange
        if is_sme:
            has_bse = pd.notna(row.get("BSE_Last_Trade")) or pd.notna(row.get("BSE_Open"))
            has_nse = pd.notna(row.get("NSE_Last_Trade")) or pd.notna(row.get("NSE_Open")) or bool(symbol)
            if has_bse and has_nse:
                exchange = "BSE, NSE"
            elif has_nse:
                exchange = "NSE"
            else:
                exchange = "BSE"
        else:
            raw_exch = str(row.get("Exchange", "BSE, NSE")).strip()
            exchange = raw_exch if raw_exch and raw_exch != "nan" else "BSE, NSE"
            
        # 4. Issue Structure
        issue_price = parse_float(row.get("Final_Issue_Price"))
        p_min, p_max = parse_price_band(row.get("Price Band"))
        if np.isnan(issue_price):
            issue_price = p_max if not np.isnan(p_max) else p_min
            
        total_size_cr = parse_issue_size_cr(row.get("Total Issue Size"))
        fresh_cr = parse_issue_size_cr(row.get("Fresh Issue"))
        ofs_cr = parse_issue_size_cr(row.get("Offer for Sale"))
        
        fresh_pct = round((fresh_cr / total_size_cr) * 100, 2) if total_size_cr and total_size_cr > 0 and pd.notna(fresh_cr) else np.nan
        ofs_pct = round((ofs_cr / total_size_cr) * 100, 2) if total_size_cr and total_size_cr > 0 and pd.notna(ofs_cr) else np.nan
        
        lot_size = parse_float(row.get("Lot Size"))
        if np.isnan(lot_size) and is_sme:
            lot_size = parse_float(row.get("Retail (Min)_Shares"))
            
        face_value = parse_float(row.get("Face Value per share"))
        sector = str(row.get("Sector", "")).strip()
        industry = str(row.get("Industry", "")).strip()
        
        # 5. Subscriptions
        qib_sub, nii_sub, ret_sub, tot_sub = extract_latest_subscription(row, is_sme=is_sme)
        if np.isnan(tot_sub):
            tot_sub = parse_float(row.get("Total_subscriptions"))
            
        # 6. Financial Attributes
        rev = parse_float(row.get("0_revenue"))
        pat = parse_float(row.get("0_profit after tax"))
        assets = parse_float(row.get("0_assets"))
        borrowings = parse_float(row.get("0_total borrowing"))
        
        rev_prev = parse_float(row.get("1_revenue"))
        pat_prev = parse_float(row.get("1_profit after tax"))
        
        rev_growth = round(((rev - rev_prev) / rev_prev) * 100, 2) if rev_prev and rev_prev > 0 and pd.notna(rev) else np.nan
        pat_growth = round(((pat - pat_prev) / abs(pat_prev)) * 100, 2) if pat_prev and abs(pat_prev) > 0 and pd.notna(pat) else np.nan
        pat_margin = round((pat / rev) * 100, 2) if rev and rev > 0 and pd.notna(pat) else np.nan
        roa = round((pat / assets) * 100, 2) if assets and assets > 0 and pd.notna(pat) else np.nan
        debt_to_assets = round(borrowings / assets, 4) if assets and assets > 0 and pd.notna(borrowings) else np.nan
        profitable = 1 if (pd.notna(pat) and pat > 0) else (0 if pd.notna(pat) else np.nan)
        
        # 7. Market & Macro Features at subscription close (Prediction Timestamp)
        pred_date = pd.to_datetime(close_date or open_date or list_date)
        market_feats = market_mgr.get_market_features(pred_date)
        
        # 8. Listing Day Outcomes
        if is_sme:
            open_p = parse_float(row.get("Open") or row.get("NSE_Open") or row.get("BSE_Open"))
            high_p = parse_float(row.get("High") or row.get("NSE_High") or row.get("BSE_High"))
            low_p = parse_float(row.get("Low") or row.get("NSE_Low") or row.get("BSE_Low"))
            close_p = parse_float(row.get("Last_Trade") or row.get("NSE_Last_Trade") or row.get("BSE_Last_Trade"))
        else:
            open_p = parse_float(row.get("NSE_Open"))
            high_p = parse_float(row.get("NSE_High"))
            low_p = parse_float(row.get("NSE_Low"))
            close_p = parse_float(row.get("NSE_Last_Trade") or row.get("BSE_Last_Trade"))
            
        # Target calculations
        if pd.notna(close_p) and pd.notna(issue_price) and issue_price > 0:
            listing_return_pct = round(((close_p - issue_price) / issue_price) * 100, 2)
            positive_return = 1 if listing_return_pct > 0 else 0
        else:
            listing_return_pct = np.nan
            positive_return = np.nan
            
        if pd.notna(open_p) and pd.notna(issue_price) and issue_price > 0:
            open_gain_pct = round(((open_p - issue_price) / issue_price) * 100, 2)
        else:
            open_gain_pct = np.nan

        # 9. Extended session outcomes (cached / yfinance)
        session_outcomes = {
            "close_price_7th_session": np.nan,
            "close_price_30th_session": np.nan,
            "max_drawdown_30_sessions_pct": np.nan,
            "return_7th_session_pct": np.nan,
            "return_30th_session_pct": np.nan,
            "positive_7_session_return": np.nan,
            "positive_30_session_return": np.nan,
        }
        if fetch_extended_sessions and symbol and pd.notna(issue_price):
            session_outcomes = outcome_mgr.calculate_session_outcomes(symbol, pd.to_datetime(list_date), issue_price)

        # Verification Status
        v_status = "verified"
        if pd.isna(listing_return_pct) or pd.isna(issue_price):
            v_status = "unresolved"
        elif pd.isna(tot_sub) or pd.isna(rev):
            v_status = "partially_verified"

        rec = {
            "company_name": comp_name,
            "company_symbol": symbol,
            "listing_segment": segment,
            "exchange": exchange,
            "ipo_open_date": open_date,
            "ipo_close_date": close_date,
            "listing_date": list_date,
            "issue_price_min": p_min,
            "issue_price_max": p_max,
            "issue_price": issue_price,
            "total_issue_size_cr": total_size_cr,
            "fresh_issue_amount_cr": fresh_cr,
            "ofs_amount_cr": ofs_cr,
            "fresh_issue_pct": fresh_pct,
            "ofs_pct": ofs_pct,
            "lot_size": lot_size,
            "face_value": face_value,
            "sector": sector if sector != "nan" else "",
            "industry": industry if industry != "nan" else "",
            "qib_subscription": qib_sub,
            "nii_subscription": nii_sub,
            "retail_subscription": ret_sub,
            "total_subscription": tot_sub,
            "revenue_cr": rev,
            "revenue_growth_pct": rev_growth,
            "pat_cr": pat,
            "pat_growth_pct": pat_growth,
            "pat_margin_pct": pat_margin,
            "total_assets_cr": assets,
            "total_borrowings_cr": borrowings,
            "return_on_assets_pct": roa,
            "debt_to_assets_ratio": debt_to_assets,
            "profitable_before_ipo": profitable,
            **market_feats,
            "source_url": str(row.get("url") or row.get("source_origin", "")),
            "verification_status": v_status,
            "listing_open_price": open_p,
            "listing_day_high": high_p,
            "listing_day_low": low_p,
            "listing_day_close_price": close_p,
            **session_outcomes,
            "listing_day_close_return_pct": listing_return_pct,
            "positive_listing_return": positive_return,
            "listing_opening_gain_pct": open_gain_pct,
        }
        clean_records.append(rec)
        
    df_clean = pd.DataFrame(clean_records)
    
    # Sort chronologically by listing_date, then company_name for stable tie-breaking
    df_clean = df_clean.sort_values(by=["listing_date", "company_name"]).reset_index(drop=True)
    
    # Generate stable unique ipo_id
    df_clean.insert(0, "ipo_id", [f"IPO_{i+1:04d}" for i in range(len(df_clean))])
    
    # Reorder according to schema
    final_cols = [c for c in CLEAN_MASTER_COLUMNS if c in df_clean.columns]
    df_clean = df_clean[final_cols]
    
    clean_csv = os.path.join(data_dir, "processed", "ipo_master_clean.csv")
    os.makedirs(os.path.dirname(clean_csv), exist_ok=True)
    df_clean.to_csv(clean_csv, index=False)
    logger.info(f"Saved {clean_csv} with {len(df_clean)} verified records.")
    return df_clean

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_clean_master()
