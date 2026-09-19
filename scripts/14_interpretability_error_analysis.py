"""
14 | DRD2: Model Interpretation and Prediction Error Analysis

Examine held-out classification errors and fingerprint feature importance for a DRD2 random
forest.

Method
------
Fit one random forest on all development data, classify the held-out test molecules, label
false positives and false negatives, and rank fingerprint bits by model importance.

Outputs
-------
Saved under results/:
- 14_test_error_analysis.csv
- 14_rf_feature_importance.csv

Outcome and Interpretation
--------------------------
The error table supports compound-level inspection; the importance table identifies influential
fingerprint bits in this fitted model. Hashed bits may represent multiple environments, so
importance alone does not identify a unique chemical substructure.
"""

# SECTION: Configuration and Input Data
from pathlib import Path
import random
import numpy as np
import pandas as pd

# Set reproducible pseudo-random seeds; hardware and library differences can still affect results.
SEED=42
N_SPLITS=10
# Use the same active-class cutoff throughout this workflow; this is not a tuned threshold.
THRESHOLD=0.50
random.seed(SEED); np.random.seed(SEED)

# Locate the project from script or notebook execution; notebooks may not define __file__.
def project_root():
    candidates=[Path.cwd(),Path.cwd().parent]
    if "__file__" in globals(): candidates.insert(0,Path(__file__).resolve().parents[1])
    for p in candidates:
        if (p/"data"/"D2_training_set_Ki.csv").exists(): return p
    raise FileNotFoundError("Run from the project root, scripts/, or notebooks/.")

ROOT=project_root(); RESULTS=ROOT/"results"; RESULTS.mkdir(exist_ok=True)
train_df=pd.read_csv(ROOT/"data"/"D2_training_set_Ki.csv")
test_df=pd.read_csv(ROOT/"data"/"D2_test_scaffold_split_Ki.csv")
for name,df in [("training",train_df),("test",test_df)]:
    if not {"SMILES","Activity"}.issubset(df.columns): raise ValueError(f"{name} file must contain SMILES and Activity")
print(f"Training: {train_df.shape} | active fraction={train_df.Activity.mean():.4f}")
print(f"Test:     {test_df.shape} | active fraction={test_df.Activity.mean():.4f}")
# SECTION: Molecular Representation: Morgan Fingerprints
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
FP_SIZE=2048; MORGAN_RADIUS=2
# Initialize the shared fingerprint generator; identical settings are used for development and test molecules.
fp_gen=rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS,fpSize=FP_SIZE)
# Create one fixed-length binary fingerprint per SMILES, preserving row order.
# Radius 2 describes local atom neighborhoods; hashed bits can represent multiple fragments.
# Fail on invalid SMILES instead of silently training on an all-zero placeholder.
def morgan_matrix(smiles):
    X=np.zeros((len(smiles),FP_SIZE),dtype=np.uint8); invalid=[]
    for i,s in enumerate(smiles):
        mol=Chem.MolFromSmiles(str(s))
        if mol is None: invalid.append(i); continue
        fp=fp_gen.GetFingerprint(mol); DataStructs.ConvertToNumpyArray(fp,X[i])
    if invalid: raise ValueError(f"Invalid SMILES at rows {invalid[:10]}")
    return X
# SECTION: Classification Metrics
from sklearn.metrics import roc_auc_score,average_precision_score,matthews_corrcoef,balanced_accuracy_score,recall_score,precision_score,brier_score_loss,confusion_matrix

# Compare binary labels (0 inactive, 1 active) with P(active).
# ROC AUC measures ranking; PR_AUC keys use average precision, not trapezoidal area.
# MCC and balanced accuracy summarize label predictions; Brier measures probability error.
# Reversing labels/probabilities lets the same metrics describe the inactive class.
def evaluate(y_true,p_active,threshold=THRESHOLD):
    y_true=np.asarray(y_true,dtype=int); p_active=np.asarray(p_active,dtype=float)
    y_pred=(p_active>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    return {"ROC_AUC":roc_auc_score(y_true,p_active),"PR_AUC_active":average_precision_score(y_true,p_active),"PR_AUC_inactive":average_precision_score(1-y_true,1-p_active),"MCC":matthews_corrcoef(y_true,y_pred),"BalancedAcc":balanced_accuracy_score(y_true,y_pred),"Recall_active":recall_score(y_true,y_pred,pos_label=1,zero_division=0),"Recall_inactive":recall_score(y_true,y_pred,pos_label=0,zero_division=0),"Precision_active":precision_score(y_true,y_pred,pos_label=1,zero_division=0),"Precision_inactive":precision_score(y_true,y_pred,pos_label=0,zero_division=0),"Brier":brier_score_loss(y_true,p_active),"threshold":threshold,"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp)}

# Aggregate metric means and sample standard deviations (ddof=1).
# When split-specific rows exist, keep Train, Validation, and Test summaries separate.
# Repeated test predictions come from different models on the same molecules.
def summarize_cv(df,model_name):
    cols=["ROC_AUC","PR_AUC_active","PR_AUC_inactive","MCC","BalancedAcc","Recall_active","Recall_inactive","Precision_active","Precision_inactive","Brier"]
    out={"model":model_name,"n_folds":len(df)}
    for c in cols:
        out[f"CV_{c}_mean"]=df[c].mean(); out[f"CV_{c}_std"]=df[c].std(ddof=1)
    return out
# SECTION: Test Errors and Fingerprint Importance
from sklearn.ensemble import RandomForestClassifier
X=morgan_matrix(train_df.SMILES); y=train_df.Activity.to_numpy(dtype=np.int64); X_test=morgan_matrix(test_df.SMILES); y_test=test_df.Activity.to_numpy(dtype=np.int64)
model=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=-1,random_state=SEED); model.fit(X,y); p=model.predict_proba(X_test)[:,1]; pred=(p>=THRESHOLD).astype(int)
# Attach probabilities and error categories. Confidence here is distance from 0.50, not calibration.
errors=test_df.copy(); errors['p_active']=p; errors['predicted']=pred; errors['error_type']=np.where((y_test==0)&(pred==1),'false_positive',np.where((y_test==1)&(pred==0),'false_negative','correct')); errors['confidence']=np.abs(p-.5)*2; errors.to_csv(RESULTS/'14_test_error_analysis.csv',index=False)
# Rank impurity-based forest importances for hashed bits; collisions prevent a one-bit/one-fragment interpretation.
imp=pd.DataFrame({'fingerprint_bit':np.arange(FP_SIZE),'importance':model.feature_importances_}).sort_values('importance',ascending=False); imp.to_csv(RESULTS/'14_rf_feature_importance.csv',index=False)
print(evaluate(y_test,p)); print(errors.error_type.value_counts()); print(imp.head(20).to_string(index=False))