"""
10 | DRD2: Consolidated Model Benchmark

Combine available model summaries into a single traceable DRD2 benchmark table.

Method
------
Read results/*summary.csv files containing a model column, retain each source filename, and
align metric columns across workflows. Model training and metric computation occur upstream.

Outputs
-------
Saved under results/:
- 10_final_benchmark.csv

Outcome and Interpretation
--------------------------
The combined table reflects only the summary files currently present. Missing columns remain
empty; inspect source files and evaluation protocols before comparing rows. Rerun after
updating model results.
"""

from pathlib import Path
import pandas as pd

# Find the directory containing both data/ and results/ before reading saved artifacts.
def root():
    for p in [Path.cwd(),Path.cwd().parent,Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()]:
        if (p/'results').exists() and (p/'data').exists(): return p
    raise FileNotFoundError('Project root not found')
R=root()/'results'; frames=[]
# Collect model summary files that already exist; filenames preserve result provenance.
for f in sorted(R.glob('*summary.csv')):
    try:
        df=pd.read_csv(f)
        if 'model' in df.columns: df.insert(0,'source_file',f.name); frames.append(df)
    except Exception as e: print('Skipped',f.name,e)
if not frames: raise FileNotFoundError('No summary CSVs yet. Run model scripts first.')
# Align columns across model summaries; unavailable metrics become missing values.
out=pd.concat(frames,ignore_index=True,sort=False); out.to_csv(R/'10_final_benchmark.csv',index=False)
cols=[c for c in [
    'source_file','model',
    'CV_Train_ROC_AUC_mean','CV_Train_ROC_AUC_std',
    'CV_Validation_ROC_AUC_mean','CV_Validation_ROC_AUC_std',
    'Test_ROC_AUC_mean','Test_ROC_AUC_std',
    'CV_Train_PR_AUC_inactive_mean','CV_Train_PR_AUC_inactive_std',
    'CV_Validation_PR_AUC_inactive_mean','CV_Validation_PR_AUC_inactive_std',
    'Test_PR_AUC_inactive_mean','Test_PR_AUC_inactive_std',
    'CV_Train_MCC_mean','CV_Train_MCC_std',
    'CV_Validation_MCC_mean','CV_Validation_MCC_std',
    'Test_MCC_mean','Test_MCC_std',
    'CV_Train_BalancedAcc_mean','CV_Train_BalancedAcc_std',
    'CV_Validation_BalancedAcc_mean','CV_Validation_BalancedAcc_std',
    'Test_BalancedAcc_mean','Test_BalancedAcc_std'
] if c in out.columns]
print(out[cols].to_string(index=False))