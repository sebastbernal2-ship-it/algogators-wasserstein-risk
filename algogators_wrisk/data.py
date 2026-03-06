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
    """Load daily continuous futures prices for a universe (placeholder).

    Parameters
    ----------
    universe:
        List of futures identifiers (e.g., ["ES", "CL", ...]).
    start_date, end_date:
        Date range (inclusive) parsed by pandas.
    seed:
        RNG seed so the notebook is reproducible.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame of settle prices, indexed by business date with one
        column per symbol.

    Notes
    -----
    TODO: Replace this function with a call to your internal
    `algogators-data` library / DB.
    """

    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    if len(dates) < 5:
        raise ValueError("Date range too short; need at least ~1 week of data.")

    rng = np.random.default_rng(seed)

    n = len(dates)
    m = len(universe)

    # Common "market" factor + idiosyncratic components to create correlation.
    market = rng.normal(loc=0.0, scale=0.0075, size=n)
    idio = rng.normal(loc=0.0, scale=0.0125, size=(n, m))

    # Vol clustering: scale returns by a slow-moving volatility process.
    vol = np.sqrt(
        0.00003
        + 0.95
        * pd.Series(market**2, index=dates).ewm(span=30, adjust=False).mean().values
    )
    vol = vol / np.nanmean(vol)

    # Log returns with modest drift differences across contracts.
    drift = rng.normal(loc=0.00005, scale=0.00005, size=m)
    log_rets = drift + (0.6 * market[:, None] + 0.4 * idio) * vol[:, None]

    # Build prices from log returns.
    start_prices = rng.uniform(50.0, 200.0, size=m)
    log_prices = np.log(start_prices)[None, :] + np.cumsum(log_rets, axis=0)
    prices = np.exp(log_prices)

    return pd.DataFrame(prices, index=dates, columns=list(universe))


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
