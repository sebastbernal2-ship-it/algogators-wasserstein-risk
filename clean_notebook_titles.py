"""Script to remove 'Table #:' prefixes from notebook titles."""
import json
import re

notebook_path = r'c:\Users\ortal\algogators-wasserstein-risk\notebooks\01_wasserstein_risk_index.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = cell['source']
        new_src = []
        for line in src:
            # Match "Table X: " or "Table X " inside quotes
            line = re.sub(r'["\']Table \d+[:]?\s*', "'", line)
            new_src.append(line)
        cell['source'] = new_src

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook titles cleaned.")
