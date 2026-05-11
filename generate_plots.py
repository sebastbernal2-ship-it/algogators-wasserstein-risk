"""Standalone script to generate paper-ready plots and tables with standardized LaTeX notation.
Outputs high-resolution PNGs to the plots/ directory."""
import os, sys

sys.path.insert(0, os.path.abspath('.'))
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from dotenv import load_dotenv
load_dotenv()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine
import statsmodels.api as sm
from algogators_wrisk import config, features, analysis

# Standard Notation Mapping
NOTATION = {
    'W': r'$W_t$',
    'W_t': r'$W_t$',
    'lambda1': r'$\lambda_1$',
    'rv_past': r'$RV_{past}$',
    'rv_future': r'$RV_{future}$',
    'mdd_future': r'$MDD_{future}$',
    'mkt_ret': r'$R_{mkt}$',
    'beta_W': r'$\beta_{W_t}$',
    'p(W)': r'$p$-value ($W_t$)',
    'R2': r'$R^2$',
    'W_t (beta)': r'$\beta_{W_t}$',
    'W_t (p-val)': r'$p$-value ($W_t$)',
    'lambda1 (beta)': r'$\beta_{\lambda_1}$',
    'rv_past (beta)': r'$\beta_{RV_{past}}$',
    'Wt beta': r'$\beta_{W_t}$',
    'p-val': r'$p$-value ($W_t$)'
}

PLOTS_DIR = os.path.join(os.path.abspath('.'), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Plot: {name}")

def format_df(df):
    """Format dataframe values and rename columns/index using standardized notation."""
    def fmt(x):
        if isinstance(x, (float, np.float64, np.float32)):
            return f"{x:.4f}"
        return str(x)
    
    # Rename index and columns
    df = df.rename(columns=NOTATION).rename(index=NOTATION)
    if df.index.name in NOTATION:
        df.index.name = NOTATION[df.index.name]
    
    return df.map(fmt)

def save_table(df, name, title=None, include_index=True):
    path = os.path.join(PLOTS_DIR, name)
    
    disp_df = df.copy()
    if include_index:
        disp_df = disp_df.reset_index()
    
    disp_df = format_df(disp_df)
    
    # Estimate size
    cell_h = 0.45
    cell_w = 2.0
    fig_h = len(disp_df) * cell_h + 1.5 # Increased buffer
    fig_w = len(disp_df.columns) * cell_w
    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')
    
    if title:
        # Use set_title with pad for better separation
        ax.set_title(title, weight='bold', size=16, pad=30)
        
    table = ax.table(cellText=disp_df.values,
                     colLabels=disp_df.columns,
                     loc='center',
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Professional Styling
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#cccccc')
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#333333') # Dark professional header
            cell.set_linewidth(1.0)
        elif row % 2 == 0:
            cell.set_facecolor('#f9f9f9') # Subtle zebra stripping
            
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Table: {name}")

# ============================================================
# Load Data & Prep
# ============================================================
print("Loading data...")
if os.environ.get('DB_USER'):
    conn_str = f"postgresql+psycopg://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}@{os.environ['DB_HOST']}:5432/{os.environ['DB_NAME']}"
    engine = create_engine(conn_str)
    def fetch_data(symbols, start, end):
        price_series = []
        for symbol in symbols:
            query = f"SELECT time, close FROM {config.DB_SCHEMA}.{config.PRICES_TABLE} WHERE symbol = '{symbol}' AND time BETWEEN '{start}' AND '{end}' ORDER BY time ASC"
            df = pd.read_sql(query, engine)
            if not df.empty:
                df['time'] = pd.to_datetime(df['time']).dt.floor('D')
                s = df.set_index('time')['close'].rename(symbol)
                price_series.append(s[~s.index.duplicated(keep='last')])
        return pd.concat(price_series, axis=1).sort_index().ffill().dropna()
    all_data = fetch_data(config.UNIVERSE, config.START_DATE, config.END_DATE)
else:
    dates = pd.date_range('2015-01-01', '2025-01-01', freq='B')
    all_data = pd.DataFrame(np.random.randn(len(dates), 10).cumsum(axis=0) + 100, index=dates, columns=config.UNIVERSE)

prices = all_data.drop(columns=['ZT', 'ZF'], errors='ignore')
returns = np.log(prices).diff().dropna()
panel_op = analysis.build_core_panel(returns, rv_past_window=config.OP_RV_WINDOW, rv_future_window=config.OP_RV_WINDOW, lambda1_window=config.OP_LAMBDA_WINDOW)

def tz_safe(dt_str, index):
    dt = pd.to_datetime(dt_str)
    if hasattr(index, 'tz') and index.tz is not None:
        dt = dt.tz_localize(index.tz)
    return dt

sns.set_theme(style='whitegrid', font='serif')

# ============================================================
# FIG 1: Distribution Shift Geometry
# ============================================================
print("\nFig 1: Distribution Shift Geometry...")
crisis_dt = tz_safe('2020-03-11', returns.index)
idx = returns.index.searchsorted(crisis_dt)
if idx < len(returns): crisis_dt = returns.index[idx]
normal_dts = returns.loc['2019-06-01':'2019-06-30'].index
normal_dt = normal_dts[len(normal_dts) // 2] if len(normal_dts) > 0 else returns.index[100]

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
def plot_geom(ax, dt, title):
    loc = returns.index.get_loc(dt)
    p_prior = returns.iloc[loc-1].values
    p_curr = returns.iloc[loc].values
    sns.kdeplot(p_prior, ax=ax, label="Prior Day", color='gray', linestyle='--', fill=True, alpha=0.1)
    sns.kdeplot(p_curr, ax=ax, label=f"Day {dt.date()}", color='darkblue', linewidth=2)
    w_val = panel_op.loc[dt, 'W'] if dt in panel_op.index else np.nan
    ax.set_title(f"{title}\n$W_t$ = {w_val:.4f}", fontsize=14, weight='bold')
    ax.set_xlabel("Cross-Sectional Returns")
    ax.legend()

plot_geom(axes[0], normal_dt, "Normal Market Regime")
plot_geom(axes[1], crisis_dt, "High-Shift Crisis Regime (Tail Event)")
axes[0].set_ylabel("Density")
fig.tight_layout()
save(fig, "fig1_distribution_shift.png")

# ============================================================
# FIG 1b: W_t Time Series
# ============================================================
print("Fig 1b...")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(panel_op.index, panel_op['W'], color='steelblue', linewidth=0.8)
q95 = panel_op['W'].quantile(0.95)
ax.axhline(q95, color='crimson', linestyle='--', alpha=0.6, label=f'95th Percentile ({q95:.4f})')
ax.set_title(r'Wasserstein-1 Index ($W_t$) Time Series', fontsize=15, weight='bold')
ax.set_ylabel(r'$W_t$', fontsize=13)
ax.legend()
save(fig, 'fig1b_wt_timeseries_annotated.png')

# ============================================================
# FIG 2: W_t vs RV vs Lambda1
# ============================================================
print("Fig 2...")
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
ax_top.set_ylabel(r'$W_t$', color='tab:blue', fontsize=13)
ax_top.plot(panel_op.index, panel_op['W'], color='tab:blue', alpha=0.8, linewidth=0.8, label=r'$W_t$')
ax_top.tick_params(axis='y', labelcolor='tab:blue')
ax_rv = ax_top.twinx()
ax_rv.set_ylabel(r'Realized Volatility ($RV_{past}$)', color='tab:orange', fontsize=13)
ax_rv.plot(panel_op.index, panel_op['rv_past'], color='tab:orange', alpha=0.6, linestyle='--', label=r'$RV_{past}$')
ax_rv.tick_params(axis='y', labelcolor='tab:orange')
ax_top.set_title(r'Distribution Shift ($W_t$) vs. Market Volatility', fontsize=15, weight='bold')
ax_bot.plot(panel_op.index, panel_op['lambda1'], color='black', linewidth=0.9, label=r'$\lambda_1$')
ax_bot.set_ylabel(r'$\lambda_1$ (Eigenvalue)', fontsize=13)
ax_bot.set_title(r'Dominant Rolling Correlation ($\lambda_1$)', fontsize=14)
fig.tight_layout()
save(fig, 'fig2_wt_rv_lambda1_overlay.png')

# ============================================================
# FIG 4: Hexbin W_t vs RV
# ============================================================
print("Fig 4...")
df_hb = panel_op[['W', 'rv_future']].dropna()
fig, ax = plt.subplots(figsize=(8, 6))
hb = ax.hexbin(df_hb['W'], df_hb['rv_future'], gridsize=40, cmap='YlOrRd', mincnt=1)
cb = fig.colorbar(hb, ax=ax); cb.set_label('Frequency')
sns.regplot(x='W', y='rv_future', data=df_hb, scatter=False, color='black', ax=ax, line_kws={'linestyle':'--'})
ax.set_title(r'Joint Distribution: $W_t$ vs. Forward $RV$', fontsize=15, weight='bold')
ax.set_xlabel(r'Wasserstein Risk Index ($W_t$)', fontsize=13)
ax.set_ylabel(r'Forward Realized Volatility ($RV_{future}$)', fontsize=13)
save(fig, 'fig4_hexbin_wt_vs_rv.png')

# ============================================================
# TABLES 1, 2, 3: Standard Statistics
# ============================================================
print("Tables 1-3...")
# Table 1
t1 = analysis.get_summary_stats(panel_op['W']).rename(index={0: 'W'})
save_table(t1, "table1_summary_stats.png", r"Univariate Statistics for $W_t$")

# Table 2
t2 = analysis.run_correlation_analysis(panel_op).rename(index=NOTATION)
save_table(t2, "table2_correlations.png", r"Correlations of $W_t$ with Risk Proxies")

# Table 3
res_rv = analysis.run_rv_regression(panel_op, target_col='rv_future')
res_mdd = analysis.run_rv_regression(panel_op, target_col='mdd_future')
def ext_reg(res, name):
    return {
        'Target': name,
        'W_t (beta)': res.params.get('W', np.nan),
        'W_t (p-val)': res.pvalues.get('W', np.nan),
        'lambda1 (beta)': res.params.get('lambda1', np.nan),
        'rv_past (beta)': res.params.get('rv_past', np.nan),
        'R2': res.rsquared
    }
t3 = pd.DataFrame([ext_reg(res_rv, r'$RV_{future}$'), ext_reg(res_mdd, r'$MDD_{future}$')]).set_index('Target')
save_table(t3, "table3_regressions.png", r"Predictive Regression Coefficients ($\beta$)")

# ============================================================
# Event Study (Fig 6 + Table 4)
# ============================================================
print("Event Study...")
es_rv = analysis.make_event_study_dataset(panel_op, value_col='rv_future', quantile=config.OP_W_QUANTILE, pre=10, post=10, min_gap=5)
es_lam = analysis.make_event_study_dataset(panel_op, value_col='lambda1', quantile=config.OP_W_QUANTILE, pre=10, post=10, min_gap=5)
es_mdd = analysis.make_event_study_dataset(panel_op, value_col='mdd_future', quantile=config.OP_W_QUANTILE, pre=10, post=10, min_gap=5)
# Random
n_ev = len(es_rv.events)
er_rv = analysis.make_random_event_baseline(panel_op, value_col='rv_future', n_events=n_ev)
er_lam = analysis.make_random_event_baseline(panel_op, value_col='lambda1', n_events=n_ev)
er_mdd = analysis.make_random_event_baseline(panel_op, value_col='mdd_future', n_events=n_ev)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(es_rv.avg_path.index, es_rv.avg_path.values, 'o-', label=r'High-$W_t$')
axes[0].plot(er_rv.avg_path.index, er_rv.avg_path.values, 'k--', alpha=0.5, label='Random')
axes[0].set_title(r'Path of $RV_{future}$'); axes[0].legend()
axes[1].plot(es_lam.avg_path.index, es_lam.avg_path.values, 's-', label=r'High-$W_t$')
axes[1].plot(er_lam.avg_path.index, er_lam.avg_path.values, 'k--', alpha=0.5, label='Random')
axes[1].set_title(r'Path of $\lambda_1$'); axes[1].legend()
axes[2].plot(es_mdd.avg_path.index, es_mdd.avg_path.values, 'D-', label=r'High-$W_t$')
axes[2].plot(er_mdd.avg_path.index, er_mdd.avg_path.values, 'k--', alpha=0.5, label='Random')
axes[2].set_title(r'Path of $MDD_{future}$'); axes[2].legend()
fig.suptitle(r'Risk Metric Trajectories around Distribution Shifts', fontsize=16, weight='bold', y=1.05)
save(fig, 'fig6_event_study_comparison.png')

t4 = pd.DataFrame({
    'Metric': [r'$RV_{future}$', r'$\lambda_1$', r'$MDD_{future}$'],
    'High-Wt Mean': [es_rv.avg_path.loc[0], es_lam.avg_path.loc[0], es_mdd.avg_path.loc[0]],
    'Random Mean': [er_rv.avg_path.loc[0], er_lam.avg_path.loc[0], er_mdd.avg_path.loc[0]]
}).set_index('Metric')
save_table(t4, "table4_event_study_stats.png", r"Event Study Numerical Summary ($t=0$)")

# ============================================================
# Strategy Performance (Fig 7-8 + Table 5)
# ============================================================
print("Strategy...")
strat = analysis.run_strategy_conditioning_experiment(panel_op, quantile=config.OP_W_QUANTILE, exposure_on_event=config.OP_EXPOSURE_ON_EVENT)
def ext_st(pnl):
    mu = pnl.mean()*252; vol = pnl.std()*np.sqrt(252); s = mu/vol; mdd = (pnl.cumsum()-pnl.cumsum().cummax()).min()
    return {'Return': f"{mu:.2%}", 'Vol': f"{vol:.2%}", 'Sharpe': f"{s:.3f}", 'MaxDD': f"{mdd:.2%}"}
t5 = pd.DataFrame([{'Strategy': 'Baseline', **ext_st(strat['baseline_pnl'])}, {'Strategy': 'Conditioned', **ext_st(strat['conditioned_pnl'])}]).set_index('Strategy')
save_table(t5, "table5_strategy_performance.png", "Economic Utility & Performance Metrics")

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
axes[0].plot(strat.index, strat['baseline_cum'], label='Baseline', color='steelblue')
axes[0].plot(strat.index, strat['conditioned_cum'], label='Conditioned', color='darkorange')
axes[0].set_title('Cumulative Performance', fontsize=14); axes[0].legend()
dd_b = features.compute_drawdowns(strat['baseline_pnl']); dd_c = features.compute_drawdowns(strat['conditioned_pnl'])
axes[1].fill_between(dd_b.index, dd_b, color='steelblue', alpha=0.3, label='Baseline')
axes[1].fill_between(dd_c.index, dd_c, color='darkorange', alpha=0.5, label='Conditioned')
axes[1].set_title('Drawdown Profile', fontsize=14); axes[1].legend()
save(fig, 'fig7_fig8_pnl_drawdown.png')

# ============================================================
# Robustness (Tables 6-8)
# ============================================================
print("Robustness...")
# Table 6
sens = []; res_b = analysis.run_rv_regression(panel_op)
sens.append({'Asset Dropped': 'None', 'Wt beta': res_b.params['W'], 'p-val': res_b.pvalues['W']})
for sym in config.UNIVERSE[:5]:  # show 5 results for depth
    sub_r = returns.drop(columns=[sym], errors='ignore')
    sw = features.compute_wasserstein_shift_index(sub_r)
    sp = panel_op.copy(); sp['W'] = sw; sr = analysis.run_rv_regression(sp)
    sens.append({'Asset Dropped': sym, 'Wt beta': sr.params['W'], 'p-val': sr.pvalues['W']})
t6 = pd.DataFrame(sens).set_index('Asset Dropped')
save_table(t6, "table6_universe_sensitivity.png", r"Universe Sensitivity ($\beta_{W_t}$ stability)")

# Table 7
p_is, p_oos = analysis.split_time_series(panel_op)
ris = analysis.run_rv_regression(p_is); roos = analysis.run_rv_regression(p_oos)
t7 = pd.DataFrame([{'Regime': 'In-Sample', 'beta': ris.params['W'], 'p-val': ris.pvalues['W']}, {'Regime': 'Out-of-Sample', 'beta': roos.params['W'], 'p-val': roos.pvalues['W']}]).set_index('Regime')
save_table(t7, "table7_oos_comparison.png", r"Regime Stability (In vs. Out Sample)")

# Table 8
c_s = tz_safe('2020-02-15', panel_op.index); c_e = tz_safe('2020-04-30', panel_op.index)
p_nc = panel_op.loc[~((panel_op.index >= c_s) & (panel_op.index <= c_e))]
rnc = analysis.run_rv_regression(p_nc)
t8 = pd.DataFrame([{'Sample': 'Full', 'beta': res_b.params['W'], 'p-val': res_b.pvalues['W']}, {'Sample': 'Excl. COVID', 'beta': rnc.params['W'], 'p-val': rnc.pvalues['W']}]).set_index('Sample')
save_table(t8, "table8_crisis_isolation.png", "Robustness - Crisis Isolation")

print("\nCleanup Successful. All figures and tables generated with uniform LaTeX notation.")
