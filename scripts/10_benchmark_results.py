"""
Collect available CV and test summaries

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
R=root()/'results'; frames=[]
for f in sorted(R.glob('*summary.csv')):
    try:
        df=pd.read_csv(f)
        if 'model' in df.columns: df.insert(0,'source_file',f.name); frames.append(df)
    except Exception as e: print('Skipped',f.name,e)
if not frames: raise FileNotFoundError('No summary CSVs yet. Run model scripts first.')
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
