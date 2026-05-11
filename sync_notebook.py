"""Script to synchronize notebook cells with the standardized LaTeX notation and table display."""
import json

notebook_path = r'c:\Users\ortal\algogators-wasserstein-risk\notebooks\01_wasserstein_risk_index.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define notation mapping for string replacements in code source
notation_map = {
    "'W'": "r'$W_t$'",
    "'W_t'": "r'$W_t$'",
    "\"W\"": "r'$W_t$'",
    "lambda1": r"$\lambda_1$",
    "rv_past": r"$RV_{past}$",
    "rv_future": r"$RV_{future}$",
    "mdd_future": r"$MDD_{future}$",
    "beta_W": r"$\beta_{W_t}$",
    "p(W)": r"$p$-value ($W_t$)"
}

# Targeted cell updates for notation consistency
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = cell['source']
        new_src = []
        for line in src:
            # Update titles and labels
            line = line.replace("'W'", "r'$W_t$'").replace("\"W\"", "r'$W_t$'")
            line = line.replace("lambda1", r"$\lambda_1$")
            line = line.replace("rv_past", r"$RV_{past}$")
            line = line.replace("rv_future", r"$RV_{future}$")
            line = line.replace("mdd_future", r"$MDD_{future}$")
            
            # Specific fixes for the user images
            if "W_t" in line and "$" not in line:
                line = line.replace("W_t", r"$W_t$")
            
            # Fix Beta and p-val naming in regression cells
            if "W_t (beta)" in line: line = line.replace("W_t (beta)", r"$\beta_{W_t}$")
            if "W_t (p-val)" in line: line = line.replace("W_t (p-val)", r"$p$-value ($W_t$)")
            
            new_src.append(line)
        cell['source'] = new_src

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook synchronized with standardized notation.")
