"""Analysis utilities: core panel, regressions, event studies, strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import features


def build_core_panel(
    ret_matrix: pd.DataFrame,
    *,
    rv_past_window: int = 20,
    rv_future_window: int = 20,
    lambda1_window: int = 60,
    annualize_rv: bool = True,
) -> pd.DataFrame:
    """Build the core daily panel used throughout the repo.

    The panel includes:
    - `W`: day-to-day cross-sectional distribution shift index
    - `mkt_ret`: equal-weight cross-sectional market return
    - `rv_past`: trailing realized vol of `mkt_ret`
    - `rv_future`: forward realized vol of `mkt_ret` (target variable)
    - `lambda1`: rolling largest eigenvalue of correlation matrix
    """

    r = features.build_return_matrix(ret_matrix)
    w = features.compute_wasserstein_shift_index(r)
    mkt = features.compute_market_return(r)

    rv_past = features.compute_realized_volatility(
        mkt, rv_past_window, annualize=annualize_rv
    ).rename("rv_past")

    # Forward realized vol: std over next window (shifted backwards so it lines up on t).
    rv_future = (
        mkt.rolling(rv_future_window).std().shift(-rv_future_window + 1).rename("rv_future")
    )
    if annualize_rv:
        rv_future = rv_future * np.sqrt(252)

    lambda1 = features.compute_rolling_lambda1(r, lambda1_window).rename("lambda1")

    panel = pd.concat([w, mkt, rv_past, rv_future, lambda1], axis=1)
    return panel


def run_rv_regression(
    panel: pd.DataFrame,
    *,
    hac_lags: int = 5,
    add_const: bool = True,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Run regression: rv_future ~ W + rv_past + lambda1 with HAC SEs."""

    cols = ["rv_future", "W", "rv_past", "lambda1"]
    df = panel[cols].dropna()
    y = df["rv_future"]
    X = df[["W", "rv_past", "lambda1"]]
    if add_const:
        X = sm.add_constant(X)

    model = sm.OLS(y, X)
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)})
    return res


@dataclass(frozen=True)
class EventStudyResult:
    """Container for event-study outputs."""

    events: pd.DatetimeIndex
    stacked: pd.DataFrame
    avg_path: pd.Series


def make_event_study_dataset(
    panel: pd.DataFrame,
    *,
    w_col: str = "W",
    value_col: str = "mkt_ret",
    quantile: float = 0.95,
    pre: int = 10,
    post: int = 10,
    min_gap: int = 0,
) -> EventStudyResult:
    """Construct an event-study dataset around high-`W` days.

    Parameters
    ----------
    panel:
        Output of `build_core_panel`.
    w_col:
        Column used to define events (default `W`).
    value_col:
        Column to study around events (e.g., `mkt_ret`, `rv_future`).
    quantile:
        Event threshold defined by `panel[w_col].quantile(quantile)`.
    pre, post:
        Number of days before/after the event day to include.
    min_gap:
        Minimum number of non-event days required between events (simple
        de-clustering). Set 0 to keep all.

    Returns
    -------
    EventStudyResult
        `stacked` contains one row per (event, tau) with columns:
        - event_date, tau, value
    """

    df = panel[[w_col, value_col]].dropna().copy()
    thr = df[w_col].quantile(quantile)
    candidates = df.index[df[w_col] >= thr]

    # Optional de-clustering of events.
    if min_gap > 0 and len(candidates) > 1:
        kept = [candidates[0]]
        for dt in candidates[1:]:
            if (dt - kept[-1]).days > min_gap:
                kept.append(dt)
        events = pd.DatetimeIndex(kept)
    else:
        events = pd.DatetimeIndex(candidates)

    rows: list[dict] = []
    for ev in events:
        # Ensure the full window is within the index.
        if ev not in df.index:
            continue

        loc = df.index.get_loc(ev)
        start = loc - pre
        end = loc + post
        if start < 0 or end >= len(df.index):
            continue

        window = df.iloc[start : end + 1][value_col]
        taus = np.arange(-pre, post + 1)
        for tau, val in zip(taus, window.values):
            rows.append({"event_date": ev, "tau": int(tau), "value": float(val)})

    stacked = pd.DataFrame(rows)
    if stacked.empty:
        avg = pd.Series(dtype=float, name="avg_value")
    else:
        avg = stacked.groupby("tau")["value"].mean().rename("avg_value")

    return EventStudyResult(events=events, stacked=stacked, avg_path=avg)


def run_strategy_conditioning_experiment(
    panel: pd.DataFrame,
    *,
    w_col: str = "W",
    ret_col: str = "mkt_ret",
    quantile: float = 0.95,
    exposure_on_event: float = 0.5,
) -> pd.DataFrame:
    """Baseline vs conditioned exposure (scaled down on high-`W` days).

    Returns a DataFrame with daily and cumulative PnL for:
    - `baseline`: exposure = 1 always
    - `conditioned`: exposure = `exposure_on_event` on high-`W` days
    """

    df = panel[[w_col, ret_col]].dropna().copy()
    thr = df[w_col].quantile(quantile)
    is_event = df[w_col] >= thr

    baseline_exp = pd.Series(1.0, index=df.index, name="baseline_exp")
    cond_exp = pd.Series(1.0, index=df.index, name="conditioned_exp")
    cond_exp.loc[is_event] = float(exposure_on_event)

    baseline_pnl = (baseline_exp * df[ret_col]).rename("baseline_pnl")
    conditioned_pnl = (cond_exp * df[ret_col]).rename("conditioned_pnl")

    out = pd.concat([df[w_col], df[ret_col], is_event.rename("is_event"), baseline_exp, cond_exp, baseline_pnl, conditioned_pnl], axis=1)
    out["baseline_cum"] = out["baseline_pnl"].cumsum()
    out["conditioned_cum"] = out["conditioned_pnl"].cumsum()
    return out
