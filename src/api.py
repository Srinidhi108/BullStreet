"""
api.py - High-Performance FastAPI Backend for BullStreet Bloomberg Terminal
"""
import os
import json
import logging
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.model_engine import engine, FEATURE_NAMES

logger = logging.getLogger(__name__)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert missing/non-finite dataset values into JSON-safe numbers."""
    try:
        if value is None or pd.isna(value):
            return default
        result = float(value)
        return default if not np.isfinite(result) else result
    except (TypeError, ValueError):
        return default

# Initialize FastAPI App
app = FastAPI(
    title="BullStreet IPO Intelligence Terminal API",
    description="Bloomberg-grade dual ML inference and financial analytics API for Indian IPOs",
    version="2.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Master Dataset into memory for ultra-fast queries
DATA_CLEAN_PATH = "data/processed/ipo_master_clean.csv"
DATA_SPLITS_DIR = "data/splits"

clean_df = pd.DataFrame()
if os.path.exists(DATA_CLEAN_PATH):
    clean_df = pd.read_csv(DATA_CLEAN_PATH)
    # Fill NAs for JSON serialization safety
    clean_df["company_name"] = clean_df["company_name"].fillna("Unknown")
    clean_df["listing_segment"] = clean_df["listing_segment"].fillna("Mainboard")
    clean_df["sector"] = clean_df["sector"].fillna("General")


# Startup event: ensure model engine is loaded
@app.on_event("startup")
def startup_event():
    logger.info("Initializing BullStreet IPO Model Engine...")
    loaded = engine.load_artifacts()
    if not loaded:
        logger.warning("No saved model bundle found. Training engine from scratch...")
        engine.train_and_evaluate()
    logger.info("BullStreet Engine ready for Bloomberg Terminal requests.")


class PredictionRequest(BaseModel):
    ipo_id: Optional[str] = None
    features: Optional[Dict[str, Any]] = None


@app.get("/api/health")
def get_health():
    return {
        "status": "ONLINE",
        "system": "BULLSTREET INTELLIGENCE TERMINAL",
        "version": "2.0.0",
        "model_engine_fitted": engine.fitted_,
        "total_ipos_indexed": len(clean_df),
        "coverage_period": "2015 - 2024",
    }


@app.get("/api/benchmarks")
def get_benchmarks():
    """Returns validation split model evaluation metrics."""
    return engine.benchmarks


@app.get("/api/market-pulse")
def get_market_pulse():
    """Returns macro market and historical summary for terminal tape."""
    if clean_df.empty:
        return {}

    total_ipos = len(clean_df)
    mainboard_count = int((clean_df["listing_segment"] == "Mainboard").sum())
    sme_count = int((clean_df["listing_segment"] == "SME").sum())
    avg_return = round(float(clean_df["listing_day_close_return_pct"].mean()), 2)
    win_rate = round(float(clean_df["positive_listing_return"].mean() * 100), 1)

    # Yearly breakdown
    clean_df["listing_year"] = pd.to_datetime(clean_df["listing_date"], errors="coerce").dt.year
    yearly = (
        clean_df.groupby("listing_year")
        .agg(
            count=("ipo_id", "count"),
            avg_return=("listing_day_close_return_pct", "mean"),
            win_rate=("positive_listing_return", lambda x: round(x.mean() * 100, 1)),
        )
        .reset_index()
        .dropna()
        .to_dict(orient="records")
    )

    return {
        "total_ipos": total_ipos,
        "mainboard_count": mainboard_count,
        "sme_count": sme_count,
        "avg_listing_return_pct": avg_return,
        "overall_win_rate_pct": win_rate,
        "yearly_trends": yearly,
    }


@app.get("/api/ipos")
def list_ipos(
    q: Optional[str] = Query(None, description="Search query by name, symbol, or ID"),
    segment: Optional[str] = Query(None, description="Mainboard or SME"),
    sector: Optional[str] = Query(None, description="Sector filter"),
    limit: int = Query(50, ge=1, le=500),
):
    """Search and filter IPOs."""
    if clean_df.empty:
        return []

    res = clean_df.copy()

    if segment:
        res = res[res["listing_segment"].str.lower() == segment.lower()]

    if sector and sector != "ALL":
        res = res[res["sector"].str.lower() == sector.lower()]

    if q:
        query = q.strip().lower()
        res = res[
            res["ipo_id"].str.lower().str.contains(query, na=False)
            | res["company_name"].str.lower().str.contains(query, na=False)
            | res["company_symbol"].astype(str).str.lower().str.contains(query, na=False)
        ]

    # Sort recent listings first
    res = res.sort_values("listing_date", ascending=False).head(limit)

    output = []
    for _, row in res.iterrows():
        output.append({
            "ipo_id": row["ipo_id"],
            "company_name": row["company_name"],
            "company_symbol": str(row.get("company_symbol", "")),
            "listing_segment": row["listing_segment"],
            "exchange": row["exchange"],
            "listing_date": str(row["listing_date"]),
            "issue_price": safe_float(row.get("issue_price")),
            "total_issue_size_cr": safe_float(row.get("total_issue_size_cr")),
            "total_subscription": safe_float(row.get("total_subscription")),
            "actual_return_pct": round(safe_float(row.get("listing_day_close_return_pct")), 2),
            "positive_listing": int(safe_float(row.get("positive_listing_return"))),
            "sector": str(row.get("sector") or "General"),
        })

    return output


@app.get("/api/ipos/{ipo_id}")
def get_ipo_detail(ipo_id: str):
    """Retrieves complete financial, structural, and actual outcome record for an IPO."""
    if clean_df.empty:
        raise HTTPException(status_code=404, detail="Dataset not loaded")

    match = clean_df[clean_df["ipo_id"].str.upper() == ipo_id.upper()]
    if match.empty:
        # Try search by symbol
        match = clean_df[clean_df["company_symbol"].astype(str).str.upper() == ipo_id.upper()]
        if match.empty:
            raise HTTPException(status_code=404, detail=f"IPO {ipo_id} not found")

    row = match.iloc[0].to_dict()
    # Replace NaNs with None for clean JSON
    clean_row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    return clean_row


@app.post("/api/predict")
def predict_ipo(req: PredictionRequest):
    """
    Executes real-time dual ML prediction (Classification + Regression + SHAP factors).
    Can accept an ipo_id OR a custom feature payload.
    """
    features = {}

    if req.ipo_id:
        match = clean_df[clean_df["ipo_id"].str.upper() == req.ipo_id.upper()]
        if match.empty:
            match = clean_df[clean_df["company_symbol"].astype(str).str.upper() == req.ipo_id.upper()]
        if match.empty:
            raise HTTPException(status_code=404, detail=f"IPO ID '{req.ipo_id}' not found")
        features = match.iloc[0].to_dict()
    elif req.features:
        features = req.features
    else:
        raise HTTPException(status_code=400, detail="Must provide either ipo_id or features payload")

    try:
        prediction_result = engine.predict_row(features)
        
        # Attach basic metadata for frontend display
        prediction_result["metadata"] = {
            "ipo_id": features.get("ipo_id", "CUSTOM_SIM"),
            "company_name": features.get("company_name", "Hypothetical Scenario"),
            "listing_segment": features.get("listing_segment", "Mainboard"),
            "issue_price": safe_float(features.get("issue_price"), 100.0),
            "total_subscription": safe_float(features.get("total_subscription"), 1.0),
            "actual_listing_return_pct": (
                None if pd.isna(features.get("listing_day_close_return_pct"))
                else safe_float(features.get("listing_day_close_return_pct"))
            ),
            "actual_positive_return": (
                None if pd.isna(features.get("positive_listing_return"))
                else int(safe_float(features.get("positive_listing_return")))
            ),
        }
        return prediction_result
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/api/peers/{ipo_id}")
def get_peers(ipo_id: str, n: int = 5):
    """Returns top historical peer IPOs for comparative valuation."""
    peers = engine.get_similar_peers(ipo_id, n=n)
    return peers


# Serve Frontend Static Files
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
