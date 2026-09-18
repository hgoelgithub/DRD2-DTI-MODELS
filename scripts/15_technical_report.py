"""
Generate a compact technical report from available results

Evaluation design
-----------------
1. D2_training_set_Ki.csv is the development dataset.
2. StratifiedKFold(n_splits=10, shuffle=True, random_state=42) is applied only to that training dataset.
3. Each fold trains on 9/10 of the development data and validates on 1/10.
4. D2_test_scaffold_split_Ki.csv is never part of cross-validation.
5. After CV, a final model is trained on all development data and evaluated once on the held-out test data.
6. Threshold-based metrics use one fixed threshold: 0.50.
7. This file is self-contained and does not import project helper modules.
"""

from pathlib import Path
import pandas as pd

def root():
    for p in [Path.cwd(),Path.cwd().parent,Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()]:
        if (p/'results').exists() and (p/'data').exists(): return p
    raise FileNotFoundError('Project root not found')
ROOT=root(); R=ROOT/'results'; bench=R/'10_final_benchmark.csv'
lines=[
    '# DRD2 Scientific AI — Technical Results','',
    '## Evaluation design',
    '- Development data: `D2_training_set_Ki.csv`',
    '- 10-fold `StratifiedKFold` on development data',
    '- CV Train metrics: 90% training portion of each fold',
    '- CV Validation metrics: unseen 10% validation portion of each fold',
    '- Test metrics: the same held-out test set is evaluated by each of the 10 fold-trained models',
    '- Test results are summarized as mean ± SD across the 10 fold-trained models',
    '- Fixed classification threshold: 0.50','',
    '## Available model results'
]
if bench.exists(): lines.append(pd.read_csv(bench).to_markdown(index=False))
else: lines.append('Run model scripts and `10_benchmark_results.py` to populate the benchmark.')
lines += ['','## Learning curves','Deep-learning scripts save epoch-vs-training/validation-loss PNG files in `results/`.','Training loss decreasing while validation loss rises is a classic overfitting pattern.','Both losses remaining high can indicate underfitting.']
report='\n'.join(lines); (R/'15_technical_report.md').write_text(report); print(report)
