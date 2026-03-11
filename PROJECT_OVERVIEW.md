# Project Overview

This document provides a comprehensive overview of the `algogators_wrisk` repository, reflecting its current state, architecture, methodology, and next steps.

## Project Goal
The primary goal of this project is to construct and analyze a Wasserstein-based distribution-shift index ($W_t$) for cross-sectional futures returns. This metric serves as a novel indicator for risk and regime detection, which can subsequently be used for conditioning systematic trading strategies. 

**Intuition behind $W_t$**: 
While standard volatility measures the variance of a single asset or a broad market index, and correlation measures linear dependence among assets, $W_t$ captures the evolution of the *entire shape* of the cross-sectional return distribution from one day to the next. By measuring the 1-Wasserstein distance between consecutive daily return distributions, $W_t$ identifies broad structural changes or market regimes that aggregate volatility and correlation metrics might miss.

---

## Current Architecture

The repository is organized into a core Python package (`algogators_wrisk/`), a Jupyter Notebook for end-to-end demonstration (`notebooks/`), and configuration files.

### High-Level Components
- **Config**: Centralized settings for the futures universe, date ranges, computation windows, and database connection.
- **Data**: Ingestion of futures prices from a PostgreSQL database and computation of log returns.
- **Features**: Core mathematical operations to compute the Wasserstein shift index, realized volatility, and correlation eigenvalues.
- **Analysis**: Utilities to aggregate metrics into a single panel and run inferential statistics, event studies, and strategy backtests.
- **Notebook**: A demonstration pipeline tying the modules together.

### Module Summaries
- **`config.py`**: Defines project-wide configurations. Key setups include the futures universe (`UNIVERSE`), date bounds, rolling window sizes (`RV_PAST_WINDOW`, `LAMBDA1_WINDOW`), and database schema variables.
- **`data.py`**: Handles database interactions. Its main function `load_continuous_futures_prices` fetches daily futures prices from a PostgreSQL database using SQLAlchemy, while `compute_log_returns` converts these prices into a clean cross-sectional matrix of log returns.
- **`features.py`**: The mathematical engine of the project. Contains `compute_wasserstein_shift_index` to calculate $W_t$, `compute_realized_volatility` for trailing/forward RV, and `compute_rolling_lambda1` to extract the largest eigenvalue from the rolling correlation matrix.
- **`analysis.py`**: Houses higher-level orchestration and research functions. `build_core_panel` aggregates returns and features into a single analytical DataFrame. `run_rv_regression` fits a linear model using HAC standard errors. It also contains `make_event_study_dataset` and `run_strategy_conditioning_experiment` for structural backtesting.

### Data Flow
1. **Raw Futures Prices**: Queried from a PostgreSQL database based on configurations.
2. **Returns**: Prices are transformed into a wide log-return matrix.
3. **Feature Engineering**: The return matrix is used to compute $W_t$, realized volatility (past and forward), and $\lambda_1$.
4. **Core Panel**: Features are aligned into a consolidated DataFrame alongside market returns.
5. **Downstream Analysis**: The panel feeds into OLS regressions (to predict future volatility), event studies, or strategy conditional exposure scaling.

---

## Methodology (As Implemented)

### Feature Computation
- **Wasserstein Index ($W_t$)**: Computed as the 1D 1-Wasserstein distance between the cross-sectional, equally-weighted empirical return samples on day $t$ and day $t-1$. Because the samples are equally weighted, it reduces to the average absolute difference between the sorted return arrays of consecutive days.
- **Realized Volatility (RV)**: Calculated as the rolling standard deviation of the equal-weight market return over specific past or future windows, scaled up to annualized terms.
- **Rolling Correlation Eigenvalue ($\lambda_1$)**: Computed as the largest eigenvalue of the rolling correlation matrix of cross-sectional returns over a given window. A higher $\lambda_1$ indicates a stronger dominant market factor.

### Analytical Workflows
- **Regressions**: `run_rv_regression` predicts forward realized volatility (`rv_future`) using the current Wasserstein index (`W`), past realized volatility (`rv_past`), and the correlation eigenvalue (`lambda1`). It uses statsmodels OLS with HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors.
- **Event Studies**: `make_event_study_dataset` centers a rolling window DataFrame around days where $W_t$ exceeds a specified `quantile`. It includes optional de-clustering to ensure event distinctiveness and computes average paths along the defined variable (e.g., market return).
- **Strategy Conditioning**: `run_strategy_conditioning_experiment` simulates a simple strategy that scales exposure down (e.g., from 1.0 to 0.5) on days when $W_t$ crosses a high-risk quantile boundary, comparing the conditioned PnL strictly against a baseline strategy.

### Encoded Assumptions in Configuration
- **Universe**: Hardcoded to an specific basket of 6 futures contracts (`NG`, `ZT`, `ZF`, `6L`, `6A`, `HO`).
- **Windows**: The past/future RV windows are 20 days, and the $\lambda_1$ correlation window is 60 days.
- **Thresholds**: The extreme event quantile for $W_t$ is defined at the 95th percentile, and the conditioned strategy scales exposure down to 0.5 upon hitting this threshold.

---

## Current Status

### Working End-to-End
- Database connection via `SQLAlchemy` (leveraging `.env` credentials) successfully pulls continuous futures prices.
- Computation of cross-sectional log returns handles missing data natively.
- Full generation of the core risk metric panel ($W_t$, $RV_{past}$, $RV_{future}$, $\lambda_1$).
- Regression analysis runs successfully with valid coefficients and statistics.
- The Jupyter Notebook demonstrates the data ingestion, panel generation, regression execution, and provides clean visual overlays of cumulative returns and $W_t$ against historical RV.

### Partially Stubbed or Unused
- **`algogators-data` integration**: The `data.py` layer conceptually expects an internal `algogators-data` backend but actively falls back to a custom standalone SQLAlchemy implementation using generic variables.
- **Untapped Analysis Logic**: The `analysis.make_event_study_dataset` and `analysis.run_strategy_conditioning_experiment` functions are complete but are not yet implemented or visualized in the demonstration notebook.

---

## Next Steps & Open Questions

### Concrete Next Steps
1. **Incorporate Unused Functions**: Integrate the `EventStudyResult` and strategy conditioning mechanics into the primary Jupyter notebook with appropriate visualizations.
2. **Robustness Checks**: Systematically analyze sensitivity to the rolling windows (20 vs. 60 days) and event quantiles (0.90 vs 0.95 vs 0.99) to validate stability.
3. **Data Handling Refinements**: Implement stricter survivorship bias handling or minimum liquidity boundaries before generating the equal-weighted distribution cross-section.
4. **Internal Integration**: Swap the hardcoded `create_engine` pipeline with the native internal `algogators-data` Python package once formally available.

### Design / API Questions (System Architecture)
- **Multivariate Optimal Transport**: Could/should $W_t$ be expanded beyond 1D to handle multivariate interactions or be weighted non-uniformly (e.g., liquidity, market capitalization, volatility-weighted)?
- **Module Extensibility**: How should we structure the repository to allow researchers to easily stack alternative Strategy Conditioning definitions beyond simple static exposure scaling? 
- **Experiment Tracking**: How should we log and track backtest variables natively within the framework? Is there a need for MLflow or an internal database schema explicitly for configurations and metrics?
