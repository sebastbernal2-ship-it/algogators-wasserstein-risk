"""algogators_wrisk configuration."""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

# --- Universe ---
# Matches confirmed symbols in the DB
UNIVERSE: list[str] = ["NG", "ZT", "ZF", "6L", "6A", "HO"]

# --- Date Range ---
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

# --- Rolling Windows ---
RV_PAST_WINDOW = 20
RV_FUTURE_WINDOW = 20
LAMBDA1_WINDOW = 60

# --- Thresholds & Regression ---
W_EVENT_QUANTILE = 0.95
EXPOSURE_ON_EVENT = 0.5
HAC_LAGS = 5

# --- Data Source Metadata ---
DB_SCHEMA = "futures_data"
PRICES_TABLE = "new_data_ohlcv_1d"
COL_DATE = "time"
COL_SYMBOL = "symbol"
COL_PRICE = "close"