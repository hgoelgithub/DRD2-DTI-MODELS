"""
Uncertainty and applicability-domain analysis

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

# SECTION: Imports, settings, and data
from pathlib import Path
import random
import numpy as np
import pandas as pd

SEED=42
N_SPLITS=10
THRESHOLD=0.50
random.seed(SEED); np.random.seed(SEED)

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
# SECTION: Morgan fingerprints
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
FP_SIZE=2048; MORGAN_RADIUS=2
fp_gen=rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS,fpSize=FP_SIZE)
def morgan_matrix(smiles):
    X=np.zeros((len(smiles),FP_SIZE),dtype=np.uint8); invalid=[]
    for i,s in enumerate(smiles):
        mol=Chem.MolFromSmiles(str(s))
        if mol is None: invalid.append(i); continue
        fp=fp_gen.GetFingerprint(mol); DataStructs.ConvertToNumpyArray(fp,X[i])
    if invalid: raise ValueError(f"Invalid SMILES at rows {invalid[:10]}")
    return X
# SECTION: Metrics
from sklearn.metrics import roc_auc_score,average_precision_score,matthews_corrcoef,balanced_accuracy_score,recall_score,precision_score,brier_score_loss,confusion_matrix

def evaluate(y_true,p_active,threshold=THRESHOLD):
    y_true=np.asarray(y_true,dtype=int); p_active=np.asarray(p_active,dtype=float)
    y_pred=(p_active>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    return {"ROC_AUC":roc_auc_score(y_true,p_active),"PR_AUC_active":average_precision_score(y_true,p_active),"PR_AUC_inactive":average_precision_score(1-y_true,1-p_active),"MCC":matthews_corrcoef(y_true,y_pred),"BalancedAcc":balanced_accuracy_score(y_true,y_pred),"Recall_active":recall_score(y_true,y_pred,pos_label=1,zero_division=0),"Recall_inactive":recall_score(y_true,y_pred,pos_label=0,zero_division=0),"Precision_active":precision_score(y_true,y_pred,pos_label=1,zero_division=0),"Precision_inactive":precision_score(y_true,y_pred,pos_label=0,zero_division=0),"Brier":brier_score_loss(y_true,p_active),"threshold":threshold,"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp)}

def summarize_cv(df,model_name):
    cols=["ROC_AUC","PR_AUC_active","PR_AUC_inactive","MCC","BalancedAcc","Recall_active","Recall_inactive","Precision_active","Precision_inactive","Brier"]
    out={"model":model_name,"n_folds":len(df)}
    for c in cols:
        out[f"CV_{c}_mean"]=df[c].mean(); out[f"CV_{c}_std"]=df[c].std(ddof=1)
    return out
# SECTION: CV ensemble uncertainty
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from rdkit import DataStructs
X=morgan_matrix(train_df.SMILES); y=train_df.Activity.to_numpy(dtype=np.int64); X_test=morgan_matrix(test_df.SMILES); y_test=test_df.Activity.to_numpy(dtype=np.int64)
skf=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED); ensemble=[]
for fold,(tr,va) in enumerate(skf.split(X,y),1):
    m=RandomForestClassifier(n_estimators=250,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=-1,random_state=SEED+fold); m.fit(X[tr],y[tr]); ensemble.append(m)
P=np.vstack([m.predict_proba(X_test)[:,1] for m in ensemble]); mean_p=P.mean(0); std_p=P.std(0)
train_fps=[fp_gen.GetFingerprint(Chem.MolFromSmiles(str(s))) for s in train_df.SMILES]; test_fps=[fp_gen.GetFingerprint(Chem.MolFromSmiles(str(s))) for s in test_df.SMILES]
nearest=[max(DataStructs.BulkTanimotoSimilarity(fp,train_fps)) for fp in test_fps]
out=pd.DataFrame({'SMILES':test_df.SMILES,'y_true':y_test,'p_active_mean':mean_p,'ensemble_std':std_p,'nearest_train_tanimoto':nearest,'y_pred':(mean_p>=THRESHOLD).astype(int)})
out.to_csv(RESULTS/'11_uncertainty_applicability_test.csv',index=False); print(evaluate(y_test,mean_p)); print(out[['ensemble_std','nearest_train_tanimoto']].describe())
