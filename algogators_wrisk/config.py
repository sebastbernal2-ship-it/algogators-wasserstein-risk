"""algogators_wrisk configuration.

This module centralizes defaults for the repo so notebooks/scripts can import a
single source of truth for universe selection, windows, and thresholds.
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Universe & date range
# ----------------------------

# A small, liquid, cross-asset futures universe (placeholders; edit freely).
# Convention: use your internal continuous-futures identifiers.
UNIVERSE: list[str] = [
    "ES.v.0",
    "NQ.v.0",
    "YM.v.0",
    "RTY.v.0",
    "CL.v.0",
    "NG.v.0",
    "GC.v.0",
    "SI.v.0",
    "ZN.v.0",
    "ZF.v.0",
    "6E.v.0",
    "6J.v.0",
]

# Default backtest / research range (business days).
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

# ----------------------------
# Rolling windows
# ----------------------------

# Realized-vol window (past) and forward window (future) in trading days.
RV_PAST_WINDOW = 20
RV_FUTURE_WINDOW = 20

# Correlation eigenvalue window (days) for rolling lambda_1.
LAMBDA1_WINDOW = 60

# ----------------------------
# Thresholds
# ----------------------------

# Event days defined as high Wasserstein shift.
W_EVENT_QUANTILE = 0.95

# Strategy conditioning: exposure multiplier on event days.
EXPOSURE_ON_EVENT = 0.5

# Regression settings
HAC_LAGS = 5

# ----------------------------
# Data sources (placeholders)
# ----------------------------

# These are placeholders for when you wire up your internal DB/data library.
DB_SCHEMA = "research"
PRICES_TABLE = "continuous_futures_daily"

# Column name conventions for your internal loader (placeholders).
COL_DATE = "date"
COL_SYMBOL = "symbol"
COL_PRICE = "settle"
