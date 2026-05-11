"""Fix Fig 2: restore W_t vs RV dual-axis overlay with proper scaling, 
add lambda1 as a separate subplot. Colors: blue, orange, black."""
import json

notebook_path = r'c:\Users\ortal\algogators-wasserstein-risk\notebooks\01_wasserstein_risk_index.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('id') == 'f62b489d':
        cell['source'] = [
            "# --- Fig 2: W_t vs Realized Volatility (dual-axis overlay) + Lambda1 subplot ---\n",
            "fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)\n",
            "\n",
            "# Top panel: W_t (blue, left axis) vs RV (orange, right axis)\n",
            "color_w = '#1f77b4'   # blue\n",
            "color_rv = '#ff7f0e'  # orange\n",
            "color_lam = 'black'\n",
            "\n",
            "ax_top.set_ylabel('Wasserstein Risk Index ($W_t$)', color=color_w, fontsize=12)\n",
            "ax_top.plot(panel_op.index, panel_op['W'], color=color_w, alpha=0.9, linewidth=0.8, label='$W_t$')\n",
            "ax_top.tick_params(axis='y', labelcolor=color_w)\n",
            "\n",
            "ax_rv = ax_top.twinx()\n",
            "ax_rv.set_ylabel('Realized Volatility (annualized)', color=color_rv, fontsize=12)\n",
            "ax_rv.plot(panel_op.index, panel_op['rv_past'], color=color_rv, alpha=0.7, linestyle='--', linewidth=1.2, label='RV Past')\n",
            "ax_rv.tick_params(axis='y', labelcolor=color_rv)\n",
            "\n",
            "# Combined legend\n",
            "lines1, labels1 = ax_top.get_legend_handles_labels()\n",
            "lines2, labels2 = ax_rv.get_legend_handles_labels()\n",
            "ax_top.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)\n",
            "ax_top.set_title('Distribution Shift ($W_t$) vs Realized Volatility (Fig 2a)', fontsize=14)\n",
            "ax_top.grid(True, alpha=0.2)\n",
            "\n",
            "# Bottom panel: Lambda1\n",
            "ax_bot.plot(panel_op.index, panel_op['lambda1'], color=color_lam, linewidth=0.9, label='$\\\\lambda_1$')\n",
            "ax_bot.set_ylabel('$\\\\lambda_1$ (correlation eigenvalue)', fontsize=12)\n",
            "ax_bot.set_xlabel('Date', fontsize=12)\n",
            "ax_bot.set_title('Rolling Correlation Eigenvalue $\\\\lambda_1$ (Fig 2b)', fontsize=14)\n",
            "ax_bot.legend(loc='upper left', fontsize=10)\n",
            "ax_bot.grid(True, alpha=0.2)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig(os.path.join(PLOTS_DIR, 'fig2_wt_rv_lambda1_overlay.png'), dpi=150, bbox_inches='tight')\n",
            "plt.show()"
        ]
        print("Updated Fig 2 cell in notebook")
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Done.")
