"""
Historical outcomes manager for IPO Intelligence Platform.
Calculates listing day prices and post-listing trading session outcomes (7th, 30th session, max drawdown)
strictly separating outcomes from predictor attributes.
"""
import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

class OutcomeManager:
    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = raw_dir
        self.cache_dir = os.path.join(raw_dir, "ticker_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_stock_history(self, symbol: str, listing_date: pd.Timestamp) -> pd.DataFrame:
        """
        Fetches stock price history from listing date onwards, with local file caching.
        """
        if pd.isna(symbol) or not str(symbol).strip():
            return pd.DataFrame()
        
        sym = str(symbol).strip().upper()
        cache_file = os.path.join(self.cache_dir, f"{sym}.csv")
        
        if os.path.exists(cache_file) and os.path.getsize(cache_file) > 100:
            df = pd.read_csv(cache_file, parse_dates=[0], index_col=0)
            return df
        
        # Try .NS first, then .BO
        for ext in [".NS", ".BO"]:
            ticker = f"{sym}{ext}"
            try:
                start_dt = (pd.to_datetime(listing_date) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
                end_dt = (pd.to_datetime(listing_date) + pd.Timedelta(days=120)).strftime("%Y-%m-%d")
                df = yf.download(ticker, start=start_dt, end=end_dt, progress=False)
                if df is not None and len(df) > 0:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.to_csv(cache_file)
                    return df
            except Exception as e:
                logger.debug(f"Failed to fetch {ticker}: {e}")
                continue
                
        return pd.DataFrame()

    def calculate_session_outcomes(self, symbol: str, listing_date: pd.Timestamp, issue_price: float) -> dict:
        """
        Calculates 7th trading session, 30th trading session closing prices and 30-session maximum drawdown.
        Counting is strictly based on actual market trading sessions from the listing date.
        """
        outcomes = {
            "close_price_7th_session": np.nan,
            "close_price_30th_session": np.nan,
            "max_drawdown_30_sessions_pct": np.nan,
            "return_7th_session_pct": np.nan,
            "return_30th_session_pct": np.nan,
            "positive_7_session_return": np.nan,
            "positive_30_session_return": np.nan,
        }
        
        if pd.isna(symbol) or pd.isna(listing_date) or pd.isna(issue_price) or issue_price <= 0:
            return outcomes

        df = self.fetch_stock_history(symbol, listing_date)
        if df.empty or "Close" not in df.columns:
            return outcomes

        # Filter trading sessions on or after listing date
        df_post = df[df.index >= pd.to_datetime(listing_date).tz_localize(None)].copy()
        if len(df_post) == 0:
            return outcomes

        # 7th trading session (index 6, 0-indexed)
        if len(df_post) >= 7:
            p7 = float(df_post["Close"].iloc[6])
            outcomes["close_price_7th_session"] = round(p7, 2)
            ret_7 = round(((p7 - issue_price) / issue_price) * 100, 2)
            outcomes["return_7th_session_pct"] = ret_7
            outcomes["positive_7_session_return"] = 1 if ret_7 > 0 else 0

        # 30th trading session (index 29, 0-indexed)
        if len(df_post) >= 30:
            p30 = float(df_post["Close"].iloc[29])
            outcomes["close_price_30th_session"] = round(p30, 2)
            ret_30 = round(((p30 - issue_price) / issue_price) * 100, 2)
            outcomes["return_30th_session_pct"] = ret_30
            outcomes["positive_30_session_return"] = 1 if ret_30 > 0 else 0

        # Max drawdown across first 30 sessions
        win_len = min(len(df_post), 30)
        if win_len >= 5:
            win = df_post.iloc[:win_len]
            # Use High and Low if present, else Close
            high_series = win["High"] if "High" in win.columns else win["Close"]
            low_series = win["Low"] if "Low" in win.columns else win["Close"]
            cum_peak = high_series.cummax()
            dd = (low_series - cum_peak) / cum_peak * 100
            outcomes["max_drawdown_30_sessions_pct"] = round(float(dd.min()), 2)

        return outcomes
