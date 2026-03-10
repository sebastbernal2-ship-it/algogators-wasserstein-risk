"""Data loading and return computation.

This repo is designed to plug into an internal data library later. For now,
`load_continuous_futures_prices` generates realistic-ish *fake* daily settle
prices so the research pipeline runs end-to-end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_continuous_futures_prices(
    universe: list[str],
    start_date: str,
    end_date: str,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Load daily continuous futures prices for a universe using algogators-data.

    Parameters
    ----------
    universe:
        List of futures identifiers (e.g., ["ES.v.0", "CL.v.0", ...]).
    start_date, end_date:
        Date range (inclusive) parsed by pandas.
    seed:
        RNG seed (kept for backward compatibility).

    Returns
    -------
    pd.DataFrame
        Wide DataFrame of settle prices, indexed by business date with one
        column per symbol.
    """
    try:
        from algogators_data import get_futures_data
    except ImportError:
        raise ImportError(
            "algogators-data is missing. Run: pip install algogators-data"
        )

    start_ts = pd.to_datetime(start_date, utc=True)
    end_ts = pd.to_datetime(end_date, utc=True)
    
    series_list = []
    
    for sym in universe:
        df = get_futures_data(sym)
        if df is None or df.empty:
            print(f"Warning: No data found for {sym}.")
            continue
            
        # Ensure time column is properly typed and tz-aware
        if 'time' not in df.columns:
            raise KeyError(f"'time' column missing from data for {sym}.")
            
        df['time'] = pd.to_datetime(df['time'], utc=True)
        
        # Filter dates
        mask = (df['time'] >= start_ts) & (df['time'] <= end_ts)
        df_filtered = df[mask].copy()
        
        if df_filtered.empty:
            print(f"Warning: No data in specified date range for {sym}.")
            continue
            
        # Try to find a valid price column (settle or close)
        price_col = None
        for col in ['settle', 'close', 'price']:
            if col in df_filtered.columns:
                price_col = col
                break
                
        if not price_col:
            raise ValueError(f"No price column (settle, close, price) found for {sym}.")
            
        s = df_filtered.set_index('time')[price_col]
        # Drop duplicates just in case
        s = s[~s.index.duplicated(keep='last')]
        s.name = sym
        series_list.append(s)

    if not series_list:
        raise ValueError("No valid data fetched for the specified universe and date range.")
        
    wide_df = pd.concat(series_list, axis=1)
    
    # Normalize index to business dates if needed
    wide_df.index = wide_df.index.normalize()
    wide_df.sort_index(inplace=True)
    
    return wide_df


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from a price panel.

    Parameters
    ----------
    prices:
        Wide DataFrame of positive prices, indexed by date.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame of log returns aligned to dates (first row dropped).
    """

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if prices.shape[1] == 0:
        raise ValueError("prices has no columns")

    px = prices.sort_index()
    if (px <= 0).any().any():
        raise ValueError("Prices must be strictly positive to compute log returns.")

    rets = np.log(px).diff()
    return rets.dropna(how="all")
