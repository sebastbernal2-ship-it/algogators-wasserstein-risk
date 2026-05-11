"""Script to update notebook cells to match the plot generation."""
import json

notebook_path = r'c:\Users\ortal\algogators-wasserstein-risk\notebooks\01_wasserstein_risk_index.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 1: Add Hexbin cell
hexbin_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "new_hexbin_cell",
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- Fig 4: Hexbin W_t vs Forward RV ---\n",
        "df_hb = panel_op[['W', 'rv_future']].dropna()\n",
        "fig, ax = plt.subplots(figsize=(8, 6))\n",
        "hb = ax.hexbin(df_hb['W'], df_hb['rv_future'], gridsize=40, cmap='YlOrRd', mincnt=1)\n",
        "cb = fig.colorbar(hb, ax=ax)\n",
        "cb.set_label('Count')\n",
        "sns.regplot(x='W', y='rv_future', data=df_hb, scatter=False, color='black', ax=ax, line_kws={'linestyle':'--'})\n",
        "ax.set_title('Joint Distribution: $W_t$ vs Forward RV (Fig 4)', fontsize=14)\n",
        "ax.set_xlabel('Wasserstein Risk Index ($W_t$)')\n",
        "ax.set_ylabel('Next 20-Day Realized Volatility')\n",
        "plt.savefig(os.path.join(PLOTS_DIR, 'fig4_hexbin_wt_vs_rv.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()"
    ]
}

# Find where to insert hexbin (after regressions)
for i, cell in enumerate(nb['cells']):
    if cell.get('id') == 'regression_table':
        nb['cells'].insert(i+1, hexbin_cell)
        break

# Modify cumulative_returns (e89c1f20) to distribution shift
for i, cell in enumerate(nb['cells']):
    if cell.get('id') == 'e89c1f20':
        cell['source'] = [
            "# --- Fig 1: Geometric Distribution Shift (Optimal Transport) ---\n",
            "crisis_dt = pd.to_datetime('2020-03-11')\n",
            "if hasattr(returns.index, 'tz') and returns.index.tz is not None:\n",
            "    crisis_dt = crisis_dt.tz_localize(returns.index.tz)\n",
            "idx = returns.index.searchsorted(crisis_dt)\n",
            "if idx < len(returns):\n",
            "    crisis_dt = returns.index[idx]\n",
            "\n",
            "normal_dts = returns.loc['2019-06-01':'2019-06-30'].index\n",
            "normal_dt = normal_dts[len(normal_dts) // 2] if len(normal_dts) > 0 else returns.index[100]\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)\n",
            "\n",
            "def plot_geom(ax, dt, title):\n",
            "    loc = returns.index.get_loc(dt)\n",
            "    if loc == 0: loc = 1\n",
            "    p_prior = returns.iloc[loc-1].values\n",
            "    p_curr = returns.iloc[loc].values\n",
            "    sns.kdeplot(p_prior, ax=ax, label=\"Prior Day\", color='gray', linestyle='--', fill=True, alpha=0.1)\n",
            "    sns.kdeplot(p_curr, ax=ax, label=f\"Day {dt.date()}\", color='firebrick', linewidth=2)\n",
            "    w_val = panel_op.loc[dt, 'W'] if dt in panel_op.index else np.nan\n",
            "    ax.set_title(f\"{title}\\n$W_t$ = {w_val:.4f}\", fontsize=14)\n",
            "    ax.set_xlabel(\"Cross-Sectional Returns\")\n",
            "    ax.legend()\n",
            "\n",
            "plot_geom(axes[0], normal_dt, \"Normal Market Day\")\n",
            "plot_geom(axes[1], crisis_dt, \"High-Shift Crisis Day (COVID-19)\")\n",
            "axes[0].set_ylabel(\"Density\")\n",
            "fig.tight_layout()\n",
            "plt.savefig(os.path.join(PLOTS_DIR, 'fig1_distribution_shift.png'), dpi=150, bbox_inches='tight')\n",
            "plt.show()"
        ]
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated with new figures.")
