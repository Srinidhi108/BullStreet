"""
Macroeconomic and market indicators loader.
Downloads and caches historical NIFTY 50, India VIX, and USD/INR exchange rates
aligned with Indian trading calendars.
"""
import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

class MarketDataManager:
    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = raw_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        self.nifty_df = None
        self.vix_df = None
        self.usdinr_df = None
        self._load_or_fetch()

    def _fetch_series(self, ticker: str, filename: str) -> pd.DataFrame:
        filepath = os.path.join(self.raw_dir, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            try:
                # Read checking if multi-row headers exist
                df_test = pd.read_csv(filepath, nrows=3)
                if df_test.iloc[0].astype(str).str.contains(r"\^|[A-Z]").any():
                    # Has MultiIndex header rows from yfinance export
                    df = pd.read_csv(filepath, skiprows=[1, 2], index_col=0)
                else:
                    df = pd.read_csv(filepath, index_col=0)
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df[df.index.notna()].sort_index()
                if len(df) > 0 and "Close" in df.columns:
                    return df
            except Exception as e:
                logger.warning(f"Error reading cache {filepath}: {e}, re-downloading...")
        
        logger.info(f"Downloading historical market series for {ticker}...")
        df = yf.download(ticker, start="2014-01-01", end="2026-01-01", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df.to_csv(filepath)
        return df

    def _load_or_fetch(self):
        self.nifty_df = self._fetch_series("^NSEI", "nifty_daily.csv")
        self.vix_df = self._fetch_series("^INDIAVIX", "vix_daily.csv")
        self.usdinr_df = self._fetch_series("INR=X", "usdinr_daily.csv")
        
        # Calculate trailing features for Nifty
        self.nifty_df["Close"] = pd.to_numeric(self.nifty_df["Close"], errors="coerce")
        self.nifty_df["nifty_ret_5d"] = self.nifty_df["Close"].pct_change(5) * 100
        self.nifty_df["nifty_ret_20d"] = self.nifty_df["Close"].pct_change(20) * 100
        self.nifty_df["nifty_ret_60d"] = self.nifty_df["Close"].pct_change(60) * 100
        self.nifty_df["nifty_vol_20d"] = (
            self.nifty_df["Close"].pct_change().rolling(20).std() * np.sqrt(252) * 100
        )
        
        self.vix_df["Close"] = pd.to_numeric(self.vix_df["Close"], errors="coerce")
        self.usdinr_df["Close"] = pd.to_numeric(self.usdinr_df["Close"], errors="coerce")

    def get_market_features(self, as_of_date: pd.Timestamp) -> dict:
        """
        Calculates market indicators strictly on or before the given as_of_date (prediction timestamp).
        Guarantees zero future lookahead.
        """
        if pd.isna(as_of_date):
            return {
                "nifty_return_5d": np.nan,
                "nifty_return_20d": np.nan,
                "nifty_return_60d": np.nan,
                "nifty_volatility_20d": np.nan,
                "india_vix": np.nan,
                "usd_inr_rate": np.nan
            }
        
        dt = pd.to_datetime(as_of_date).tz_localize(None)
        
        # Filter strictly on or prior to prediction timestamp
        nifty_sub = self.nifty_df[self.nifty_df.index <= dt]
        vix_sub = self.vix_df[self.vix_df.index <= dt]
        usd_sub = self.usdinr_df[self.usdinr_df.index <= dt]
        
        nifty_row = nifty_sub.iloc[-1] if len(nifty_sub) > 0 else None
        vix_row = vix_sub.iloc[-1] if len(vix_sub) > 0 else None
        usd_row = usd_sub.iloc[-1] if len(usd_sub) > 0 else None
        
        return {
            "nifty_return_5d": round(float(nifty_row["nifty_ret_5d"]), 4) if nifty_row is not None and pd.notna(nifty_row["nifty_ret_5d"]) else np.nan,
            "nifty_return_20d": round(float(nifty_row["nifty_ret_20d"]), 4) if nifty_row is not None and pd.notna(nifty_row["nifty_ret_20d"]) else np.nan,
            "nifty_return_60d": round(float(nifty_row["nifty_ret_60d"]), 4) if nifty_row is not None and pd.notna(nifty_row["nifty_ret_60d"]) else np.nan,
            "nifty_volatility_20d": round(float(nifty_row["nifty_vol_20d"]), 4) if nifty_row is not None and pd.notna(nifty_row["nifty_vol_20d"]) else np.nan,
            "india_vix": round(float(vix_row["Close"]), 4) if vix_row is not None and pd.notna(vix_row["Close"]) else np.nan,
            "usd_inr_rate": round(float(usd_row["Close"]), 4) if usd_row is not None and pd.notna(usd_row["Close"]) else np.nan,
        }
