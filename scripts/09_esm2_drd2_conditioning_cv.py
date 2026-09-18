"""
DRD2 ESM-2 conditioning with 10-fold CV

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

def summarize_cv(df, model_name):
    """Summarize Train, Validation, and Test metrics across the 10 fold-trained models."""
    cols = [
        "ROC_AUC", "PR_AUC_active", "PR_AUC_inactive", "MCC", "BalancedAcc",
        "Recall_active", "Recall_inactive", "Precision_active", "Precision_inactive", "Brier"
    ]
    out = {"model": model_name, "n_folds": int(df["fold"].nunique())}

    for split_name, prefix in [
        ("Train", "CV_Train"),
        ("Validation", "CV_Validation"),
        ("Test", "Test"),
    ]:
        part = df[df["split"] == split_name]
        for c in cols:
            out[f"{prefix}_{c}_mean"] = part[c].mean()
            out[f"{prefix}_{c}_std"] = part[c].std(ddof=1)
    return out
# SECTION: Protein embedding and scientific limitation
# DRD2 is the same target for every compound. A single pooled DRD2 vector is therefore constant across rows.
# This experiment is a negative-control/conditioning demonstration, not evidence that protein information improves a one-target classifier.
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from transformers import AutoTokenizer,AutoModel
MODEL_ID="facebook/esm2_t12_35M_UR50D"; EPOCHS=25; BATCH_SIZE=128
torch.manual_seed(SEED)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends,"mps") and torch.backends.mps.is_available() else "cpu"))

def read_fasta(path):
    return "".join(x.strip() for x in path.read_text().splitlines() if x.strip() and not x.startswith(">"))
seq=read_fasta(ROOT/'data'/'DRD2_P14416-1.fasta')
tok=AutoTokenizer.from_pretrained(MODEL_ID); enc=AutoModel.from_pretrained(MODEL_ID).to(DEVICE); enc.eval()
b=tok(seq,return_tensors='pt',truncation=True); b={k:v.to(DEVICE) for k,v in b.items()}
with torch.no_grad(): protein=enc(**b).last_hidden_state.mean(1).squeeze(0).cpu().numpy().astype(np.float32)
lig=morgan_matrix(train_df.SMILES).astype(np.float32); lig_test=morgan_matrix(test_df.SMILES).astype(np.float32)
X=np.concatenate([lig,np.repeat(protein[None,:],len(lig),0)],1).astype(np.float32)
X_test=np.concatenate([lig_test,np.repeat(protein[None,:],len(lig_test),0)],1).astype(np.float32)
y=train_df.Activity.to_numpy(dtype=np.int64); y_test=test_df.Activity.to_numpy(dtype=np.int64)

# SECTION: Small neural head
class Head(nn.Module):
    def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Dropout(.25),nn.Linear(256,2))
    def forward(self,x): return self.net(x)
def cw(labels):
    c=np.bincount(labels,minlength=2).astype(np.float32); return torch.tensor(len(labels)/(2*np.maximum(c,1)),dtype=torch.float32,device=DEVICE)
def fit_fold(tr,va):
    m=Head(X.shape[1]).to(DEVICE); loss_fn=nn.CrossEntropyLoss(weight=cw(y[tr])); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    loader=DataLoader(TensorDataset(torch.from_numpy(X[tr]),torch.from_numpy(y[tr])),batch_size=BATCH_SIZE,shuffle=True)
    xv=torch.from_numpy(X[va]).to(DEVICE); yv=torch.from_numpy(y[va]).to(DEVICE)
    hist={'train_loss':[],'val_loss':[]}; best=None; best_loss=float('inf'); best_epoch=1
    for ep in range(1,EPOCHS+1):
        m.train(); total=0.; n=0
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad(set_to_none=True); loss=loss_fn(m(xb),yb); loss.backward(); opt.step(); total+=loss.item()*len(yb); n+=len(yb)
        hist['train_loss'].append(total/n); m.eval()
        with torch.no_grad(): logits=m(xv); vl=loss_fn(logits,yv).item()
        hist['val_loss'].append(vl)
        if vl<best_loss: best_loss=vl; best_epoch=ep; best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(best); m.eval()
    xt=torch.from_numpy(X[tr]).to(DEVICE)
    with torch.no_grad():
        p_train=torch.softmax(m(xt),1)[:,1].cpu().numpy()
        p_val=torch.softmax(m(xv),1)[:,1].cpu().numpy()
    return m,p_train,p_val,hist,best_epoch

# SECTION: 10-fold CV and learning curve
skf=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED); rows=[]; histories=[]; best_epochs=[]
for fold,(tr,va) in enumerate(skf.split(X,y),1):
    model,p_train,p_val,h,b=fit_fold(tr,va)

    train_row=evaluate(y[tr],p_train)
    train_row.update(model='Morgan_plus_constant_DRD2_ESM2',fold=fold,split="Train",best_epoch=b)

    val_row=evaluate(y[va],p_val)
    val_row.update(model='Morgan_plus_constant_DRD2_ESM2',fold=fold,split="Validation",best_epoch=b)

    with torch.no_grad():
        p_test=torch.softmax(model(torch.from_numpy(X_test).to(DEVICE)),1)[:,1].cpu().numpy()
    test_row=evaluate(y_test,p_test)
    test_row.update(model='Morgan_plus_constant_DRD2_ESM2',fold=fold,split="Test",best_epoch=b)

    rows.extend([train_row,val_row,test_row])
    histories.append(h); best_epochs.append(b)

    print(
        f"fold {fold:02d}: "
        f"Train ROC={train_row['ROC_AUC']:.3f} MCC={train_row['MCC']:.3f} | "
        f"Validation ROC={val_row['ROC_AUC']:.3f} MCC={val_row['MCC']:.3f} | "
        f"Test ROC={test_row['ROC_AUC']:.3f} MCC={test_row['MCC']:.3f}"
    )
a=np.array([h['train_loss'] for h in histories]); b=np.array([h['val_loss'] for h in histories]); ep=np.arange(1,EPOCHS+1)
plt.figure(figsize=(8,5)); plt.plot(ep,a.mean(0),label='Training loss'); plt.plot(ep,b.mean(0),label='Validation loss'); plt.xlabel('Epoch'); plt.ylabel('Cross-entropy loss'); plt.title('ESM-2 conditioning: mean learning curve across 10 CV folds'); plt.legend(); plt.tight_layout(); plt.savefig(RESULTS/'09_esm2_epoch_vs_loss.png',dpi=180); plt.show()
fold_df=pd.DataFrame(rows); summary=summarize_cv(fold_df,'Morgan_plus_constant_DRD2_ESM2'); summary['CV_best_epoch_median']=int(np.median(best_epochs))

# SECTION: Save fold-level and mean±SD results
fold_df.to_csv(RESULTS/'09_esm2_cv_folds.csv',index=False)
pd.DataFrame([summary]).to_csv(RESULTS/'09_esm2_summary.csv',index=False)
print(pd.DataFrame([summary]).to_string(index=False))
print('\nCaution: the same DRD2 vector is repeated for all compounds, so it is not sample-specific information.')
