"""
Small SMILES Transformer trained with 10-fold CV

Evaluation design
-----------------
1. D2_training_set_Ki.csv is the development dataset.
2. StratifiedKFold(n_splits=10, shuffle=True, random_state=42) is applied only to that training dataset.
3. In each fold, 90% of the training data are used to fit the model and 10% are used as validation.
4. D2_test_scaffold_split_Ki.csv is never part of cross-validation.
5. Each of the 10 fold-trained models also evaluates the same untouched held-out test set.
6. Threshold-based metrics use one fixed threshold: 0.50.
7. This file is self-contained and does not import project helper modules.
"""

# SECTION: Imports, settings, and data
from pathlib import Path
import random
import numpy as np
import pandas as pd

SEED = 42
N_SPLITS = 10
THRESHOLD = 0.50

random.seed(SEED)
np.random.seed(SEED)

def project_root():
    """Find repository root when run from root, scripts/, or a notebook."""
    candidates = [Path.cwd(), Path.cwd().parent]
    if "__file__" in globals():
        candidates.insert(0, Path(__file__).resolve().parents[1])
    for p in candidates:
        if (p/"data"/"D2_training_set_Ki.csv").exists():
            return p
    raise FileNotFoundError("Run from the project root, scripts/, or notebooks/.")

ROOT = project_root()
RESULTS = ROOT/"results"
RESULTS.mkdir(exist_ok=True)

train_df = pd.read_csv(ROOT/"data"/"D2_training_set_Ki.csv")
test_df = pd.read_csv(ROOT/"data"/"D2_test_scaffold_split_Ki.csv")

for name, df in [("training", train_df), ("test", test_df)]:
    if not {"SMILES","Activity"}.issubset(df.columns):
        raise ValueError(f"{name} file must contain SMILES and Activity.")
    if df["Activity"].isna().any():
        raise ValueError(f"{name} file contains missing Activity values.")

print(f"Training: {train_df.shape} | active fraction={train_df.Activity.mean():.4f}")
print(f"Test:     {test_df.shape} | active fraction={test_df.Activity.mean():.4f}")
# SECTION: Evaluation metrics
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, recall_score, precision_score,
    brier_score_loss, confusion_matrix
)

def evaluate(y_true, p_active, threshold=THRESHOLD):
    """Evaluate probabilities using the same 0.50 threshold for every classifier."""
    y_true = np.asarray(y_true, dtype=int)
    p_active = np.asarray(p_active, dtype=float)
    y_pred = (p_active >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        "ROC_AUC": roc_auc_score(y_true, p_active),
        "PR_AUC_active": average_precision_score(y_true, p_active),
        "PR_AUC_inactive": average_precision_score(1-y_true, 1-p_active),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "BalancedAcc": balanced_accuracy_score(y_true, y_pred),
        "Recall_active": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall_inactive": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "Precision_active": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Precision_inactive": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "Brier": brier_score_loss(y_true, p_active),
        "threshold": threshold,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }

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
# SECTION: Character tokenizer and compact Transformer
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

torch.manual_seed(SEED)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends,"mps") and torch.backends.mps.is_available() else "cpu"))
EPOCHS=18
MAX_LEN=160
BATCH_SIZE=96

all_smiles=list(train_df.SMILES.astype(str))
chars=sorted(set("".join(all_smiles)))
stoi={"<PAD>":0,"<UNK>":1}
stoi.update({c:i+2 for i,c in enumerate(chars)})

def encode(s):
    ids=[stoi.get(c,1) for c in str(s)[:MAX_LEN]]
    ids += [0]*(MAX_LEN-len(ids))
    return np.asarray(ids,dtype=np.int64)

X=np.stack([encode(s) for s in train_df.SMILES])
X_test=np.stack([encode(s) for s in test_df.SMILES])
y=train_df.Activity.to_numpy(dtype=np.int64); y_test=test_df.Activity.to_numpy(dtype=np.int64)

class SmilesTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d=96
        self.emb=nn.Embedding(len(stoi),d,padding_idx=0)
        self.pos=nn.Parameter(torch.zeros(1,MAX_LEN,d))
        layer=nn.TransformerEncoderLayer(d_model=d,nhead=4,dim_feedforward=192,dropout=.15,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=2)
        self.out=nn.Linear(d,2)
    def forward(self,tok):
        mask=(tok==0)
        x=self.emb(tok)+self.pos[:,:tok.size(1)]
        x=self.encoder(x,src_key_padding_mask=mask)
        keep=(~mask).unsqueeze(-1).float()
        pooled=(x*keep).sum(1)/keep.sum(1).clamp(min=1)
        return self.out(pooled)

def weights(labels):
    c=np.bincount(labels,minlength=2).astype(np.float32)
    return torch.tensor(len(labels)/(2*np.maximum(c,1)),dtype=torch.float32,device=DEVICE)

def fit_fold(tr,va):
    model=SmilesTransformer().to(DEVICE); loss_fn=nn.CrossEntropyLoss(weight=weights(y[tr]))
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
    loader=DataLoader(list(zip(torch.from_numpy(X[tr]),torch.from_numpy(y[tr]))),batch_size=BATCH_SIZE,shuffle=True)
    Xv=torch.from_numpy(X[va]).to(DEVICE); yv=torch.from_numpy(y[va]).to(DEVICE)
    hist={"train_loss":[],"val_loss":[]}; best=None; best_loss=float("inf"); best_epoch=1
    for ep in range(1,EPOCHS+1):
        model.train(); total=0.; n=0
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
            total+=loss.item()*len(yb); n+=len(yb)
        hist["train_loss"].append(total/n)
        model.eval()
        with torch.no_grad():
            logits=model(Xv); vl=loss_fn(logits,yv).item()
        hist["val_loss"].append(vl)
        if vl<best_loss:
            best_loss=vl; best_epoch=ep
            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best); model.eval()
    Xt=torch.from_numpy(X[tr]).to(DEVICE)
    with torch.no_grad():
        p_train=torch.softmax(model(Xt),1)[:,1].cpu().numpy()
        p_val=torch.softmax(model(Xv),1)[:,1].cpu().numpy()
    return model,p_train,p_val,hist,best_epoch

# SECTION: 10-fold CV
skf=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED)
rows=[]; histories=[]; best_epochs=[]
for fold,(tr,va) in enumerate(skf.split(X,y),1):
    model,p_train,p_val,h,b=fit_fold(tr,va)

    train_row=evaluate(y[tr],p_train)
    train_row.update(model="SMILES_Transformer",fold=fold,split="Train",best_epoch=b)

    val_row=evaluate(y[va],p_val)
    val_row.update(model="SMILES_Transformer",fold=fold,split="Validation",best_epoch=b)

    with torch.no_grad():
        p_test=torch.softmax(
            model(torch.from_numpy(X_test).to(DEVICE)),1
        )[:,1].cpu().numpy()
    test_row=evaluate(y_test,p_test)
    test_row.update(model="SMILES_Transformer",fold=fold,split="Test",best_epoch=b)

    rows.extend([train_row,val_row,test_row])
    histories.append(h); best_epochs.append(b)

    print(
        f"fold {fold:02d}: "
        f"Train ROC={train_row['ROC_AUC']:.3f} MCC={train_row['MCC']:.3f} | "
        f"Validation ROC={val_row['ROC_AUC']:.3f} MCC={val_row['MCC']:.3f} | "
        f"Test ROC={test_row['ROC_AUC']:.3f} MCC={test_row['MCC']:.3f}"
    )

# SECTION: Learning curve
a=np.array([h["train_loss"] for h in histories]); b=np.array([h["val_loss"] for h in histories])
ep=np.arange(1,EPOCHS+1)
plt.figure(figsize=(8,5)); plt.plot(ep,a.mean(0),label="Training loss"); plt.plot(ep,b.mean(0),label="Validation loss")
plt.xlabel("Epoch"); plt.ylabel("Cross-entropy loss"); plt.title("SMILES Transformer: mean learning curve across 10 CV folds")
plt.legend(); plt.tight_layout(); plt.savefig(RESULTS/"05_smiles_transformer_epoch_vs_loss.png",dpi=180); plt.show()

fold_df=pd.DataFrame(rows); summary=summarize_cv(fold_df,"SMILES_Transformer")
summary["CV_best_epoch_median"]=int(np.median(best_epochs))
fold_df.to_csv(RESULTS/"05_smiles_transformer_cv_folds.csv",index=False)
pd.DataFrame([summary]).to_csv(RESULTS/"05_smiles_transformer_cv_summary.csv",index=False)

# SECTION: Save fold-level and mean±SD results
pd.DataFrame([summary]).to_csv(RESULTS/"05_smiles_transformer_cv_summary.csv",index=False)

print("\nThis file focuses on readable supervised Transformer CV. Pretrained molecular encoders are covered separately.")
