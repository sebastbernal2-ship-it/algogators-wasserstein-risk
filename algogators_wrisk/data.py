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
    import os
    import pandas as pd
    from sqlalchemy import create_engine
    
    # Use environment vars configured via python-dotenv
    db_user = os.environ.get('DB_USER')
    db_password = os.environ.get('DB_PASSWORD')
    db_host = os.environ.get('DB_HOST')
    db_name = os.environ.get('DB_NAME')
    
    if not all([db_user, db_password, db_host, db_name]):
         raise ValueError("Missing database credentials in environment variables.")
         
    conn_str = f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:5432/{db_name}"
    engine = create_engine(conn_str)
    
    # Optional logic: If internal config changes schema/table, update here!
    # For now we use the ones defined locally or default to standard pg admin names.
    from algogators_wrisk import config
    schema = getattr(config, 'DB_SCHEMA', 'futures_data')
    table = getattr(config, 'PRICES_TABLE', 'ohlcv_1d')

    series_list = []
    
    for sym in universe:
        query = f"SELECT time, close FROM {schema}.{table} WHERE symbol = '{sym}' AND time BETWEEN '{start_date}' AND '{end_date}' ORDER BY time ASC"
        
        try:
            df = pd.read_sql(query, engine)
        except Exception as e:
            print(f"Warning: Database query failed for {sym}: {e}")
            continue

        if not df.empty:
            df['time'] = pd.to_datetime(df['time'], utc=True).dt.floor('D')
            s = df.set_index('time')['close'].rename(sym)
            series_list.append(s[~s.index.duplicated(keep='last')])
        else:
            print(f"Warning: No data found for {sym}")

    if not series_list:
        raise ValueError("No valid data fetched for the specified universe and date range.")
        
    wide_df = pd.concat(series_list, axis=1)
    
    # Normalize index to business dates if needed
    wide_df.index = wide_df.index.normalize()
    wide_df.sort_index(inplace=True)
    
    return wide_df
            


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
