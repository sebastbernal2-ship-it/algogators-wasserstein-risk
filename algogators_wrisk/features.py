"""Feature engineering for the Wasserstein risk index.

Core objects in this repo are daily, cross-sectional futures returns:
- A return *matrix* \(R_{t,i}\) with dates as rows and symbols as columns.
- A distribution-shift index \(W_t\) computed via 1D \(W_1\) between
  consecutive cross-sectional return distributions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_return_matrix(returns: pd.DataFrame, *, universe: list[str] | None = None) -> pd.DataFrame:
    """Build a clean cross-sectional return matrix.

    Parameters
    ----------
    returns:
        Wide DataFrame of returns with dates as index, symbols as columns.
    universe:
        Optional list of symbols to select/reorder.

    Returns
    -------
    pd.DataFrame
        Return matrix aligned by date with columns in universe order if given.
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")

    r = returns.sort_index()
    if universe is not None:
        cols = [c for c in universe if c in r.columns]
        r = r[cols]
    return r.astype(float)


def wasserstein_1d_equal_weight(x: np.ndarray, y: np.ndarray) -> float:
    """Compute 1D 1-Wasserstein distance for equal-weight empirical samples.

    For two equally-weighted empirical distributions, \(W_1\) reduces to the
    average absolute difference between sorted samples.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if x.size == 0 or y.size == 0:
        return np.nan

    n = min(x.size, y.size)
    xs = np.sort(x)[:n]
    ys = np.sort(y)[:n]
    return float(np.mean(np.abs(xs - ys)))


def compute_wasserstein_shift_index(ret_matrix: pd.DataFrame) -> pd.Series:
    """Compute daily distribution-shift index \(W_t\) from cross-sectional returns.

    \(W_t\) is defined as 1D \(W_1\) distance between the cross-sectional
    distributions on day \(t\) and day \(t-1\).
    """

    r = build_return_matrix(ret_matrix)
    dates = r.index
    w = np.full(len(dates), np.nan, dtype=float)

    for t in range(1, len(dates)):
        w[t] = wasserstein_1d_equal_weight(r.iloc[t].values, r.iloc[t - 1].values)

    return pd.Series(w, index=dates, name="W")


def compute_market_return(ret_matrix: pd.DataFrame) -> pd.Series:
    """Compute a simple equal-weight 'market' return from cross-sectional returns."""

    r = build_return_matrix(ret_matrix)
    mkt = r.mean(axis=1)
    mkt.name = "mkt_ret"
    return mkt


def compute_realized_volatility(
    ret_series: pd.Series,
    window: int,
    *,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """Compute rolling realized volatility of a daily return series."""

    if window <= 1:
        raise ValueError("window must be > 1")

    rv = ret_series.rolling(window).std()
    if annualize:
        rv = rv * np.sqrt(periods_per_year)
    rv.name = f"rv_{window}"
    return rv


def compute_rolling_lambda1(
    ret_matrix: pd.DataFrame,
    window: int,
    *,
    min_obs: int | None = None,
) -> pd.Series:
    """Compute rolling largest eigenvalue of the correlation matrix.

    Parameters
    ----------
    ret_matrix:
        Cross-sectional returns (dates x assets).
    window:
        Rolling window length (days).
    min_obs:
        Minimum non-NaN observations per asset within the window. Defaults to
        `window // 2`.
    """

    if window <= 2:
        raise ValueError("window must be > 2")
    if min_obs is None:
        min_obs = max(3, window // 2)

    r = build_return_matrix(ret_matrix)
    dates = r.index
    out = np.full(len(dates), np.nan, dtype=float)

    for t in range(window - 1, len(dates)):
        w = r.iloc[t - window + 1 : t + 1]
        # Keep columns with enough data; then drop any remaining rows with NaNs
        ok_cols = w.count(axis=0) >= min_obs
        w = w.loc[:, ok_cols]
        w = w.dropna(axis=0, how="any")

        if w.shape[0] < 3 or w.shape[1] < 2:
            continue

        corr = np.corrcoef(w.values, rowvar=False)
        if not np.all(np.isfinite(corr)):
            continue

        # Largest eigenvalue of a symmetric matrix.
        eigvals = np.linalg.eigvalsh(corr)
        out[t] = float(np.max(eigvals))

    return pd.Series(out, index=dates, name=f"lambda1_{window}")
