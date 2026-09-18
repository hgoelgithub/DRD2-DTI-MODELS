"""
GNN variants with 10-fold CV and fast defaults

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
# SECTION: GNN settings
# 10-fold CV multiplies the training cost by 10. Keep the default comparison modest.
# Change RUN_EXTENDED_VARIANTS to True only when you intentionally want a longer run.
# D-MPNN is not run here by default because it was the main runtime bottleneck.
RUN_EXTENDED_VARIANTS=False
EPOCHS=12
BATCH_SIZE=128

import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv,SAGEConv,GATv2Conv,GINConv,global_mean_pool

torch.manual_seed(SEED)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends,"mps") and torch.backends.mps.is_available() else "cpu"))

# SECTION: Convert one molecule into one graph
ATOM_DIM=8
def atom_features(atom):
    return [
        atom.GetAtomicNum()/100.0,
        atom.GetTotalDegree()/6.0,
        atom.GetFormalCharge()/4.0,
        atom.GetTotalNumHs()/4.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        atom.GetMass()/250.0,
        int(atom.GetHybridization())/10.0,
    ]

def smiles_to_graph(smiles,label=None):
    mol=Chem.MolFromSmiles(str(smiles))
    if mol is None: raise ValueError(f"Invalid SMILES: {smiles}")
    x=torch.tensor([atom_features(a) for a in mol.GetAtoms()],dtype=torch.float32)
    src=[]; dst=[]
    for b in mol.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx()
        src += [i,j]; dst += [j,i]
    edge_index=torch.tensor([src,dst],dtype=torch.long) if src else torch.empty((2,0),dtype=torch.long)
    data=Data(x=x,edge_index=edge_index)
    if label is not None: data.y=torch.tensor([int(label)],dtype=torch.long)
    return data

train_graphs=[smiles_to_graph(s,y) for s,y in zip(train_df.SMILES,train_df.Activity)]
test_graphs=[smiles_to_graph(s,y) for s,y in zip(test_df.SMILES,test_df.Activity)]
y=train_df.Activity.to_numpy(dtype=np.int64)
y_test=test_df.Activity.to_numpy(dtype=np.int64)

# SECTION: Small, readable GNN architectures
class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=GCNConv(ATOM_DIM,64); self.c2=GCNConv(64,64); self.out=nn.Linear(64,2)
    def forward(self,data):
        x=F.relu(self.c1(data.x,data.edge_index))
        x=F.relu(self.c2(x,data.edge_index))
        return self.out(global_mean_pool(x,data.batch))

class GraphSAGE(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=SAGEConv(ATOM_DIM,64); self.c2=SAGEConv(64,64); self.out=nn.Linear(64,2)
    def forward(self,data):
        x=F.relu(self.c1(data.x,data.edge_index)); x=F.relu(self.c2(x,data.edge_index))
        return self.out(global_mean_pool(x,data.batch))

class GAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=GATv2Conv(ATOM_DIM,32,heads=2); self.c2=GATv2Conv(64,64,heads=1); self.out=nn.Linear(64,2)
    def forward(self,data):
        x=F.elu(self.c1(data.x,data.edge_index)); x=F.elu(self.c2(x,data.edge_index))
        return self.out(global_mean_pool(x,data.batch))

class GIN(nn.Module):
    def __init__(self):
        super().__init__()
        mlp1=nn.Sequential(nn.Linear(ATOM_DIM,64),nn.ReLU(),nn.Linear(64,64))
        mlp2=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,64))
        self.c1=GINConv(mlp1); self.c2=GINConv(mlp2); self.out=nn.Linear(64,2)
    def forward(self,data):
        x=F.relu(self.c1(data.x,data.edge_index)); x=F.relu(self.c2(x,data.edge_index))
        return self.out(global_mean_pool(x,data.batch))

models={"GCN":GCN,"GIN":GIN}
if RUN_EXTENDED_VARIANTS:
    models.update({"GraphSAGE":GraphSAGE,"GAT":GAT})

def weights(labels):
    c=np.bincount(labels,minlength=2).astype(np.float32)
    return torch.tensor(len(labels)/(2*np.maximum(c,1)),dtype=torch.float32,device=DEVICE)

def train_one(model_cls,tr_idx,va_idx):
    model=model_cls().to(DEVICE)
    loss_fn=nn.CrossEntropyLoss(weight=weights(y[tr_idx]))
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    tr_loader=DataLoader([train_graphs[i] for i in tr_idx],batch_size=BATCH_SIZE,shuffle=True)
    va_loader=DataLoader([train_graphs[i] for i in va_idx],batch_size=BATCH_SIZE,shuffle=False)

    hist={"train_loss":[],"val_loss":[]}; best=None; best_loss=float("inf"); best_epoch=1
    for epoch in range(1,EPOCHS+1):
        model.train(); total=0.; n=0
        for batch in tr_loader:
            batch=batch.to(DEVICE); opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(batch),batch.y.view(-1)); loss.backward(); opt.step()
            total+=loss.item()*batch.num_graphs; n+=batch.num_graphs
        hist["train_loss"].append(total/n)

        model.eval(); total=0.; n=0; probs=[]
        with torch.no_grad():
            for batch in va_loader:
                batch=batch.to(DEVICE); logits=model(batch)
                loss=loss_fn(logits,batch.y.view(-1))
                total+=loss.item()*batch.num_graphs; n+=batch.num_graphs
                probs.extend(torch.softmax(logits,1)[:,1].cpu().numpy())
        vloss=total/n; hist["val_loss"].append(vloss)
        if vloss<best_loss:
            best_loss=vloss; best_epoch=epoch
            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best)
    return model,hist,best_epoch

def predict_fold(model, indices):
    """Predict active probabilities for selected graph indices."""
    loader=DataLoader([train_graphs[i] for i in indices],batch_size=BATCH_SIZE,shuffle=False)
    probs=[]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            probs.extend(torch.softmax(model(batch.to(DEVICE)),1)[:,1].cpu().numpy())
    return np.asarray(probs)

# SECTION: CV for each selected GNN
skf=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED)
all_rows=[]; summaries=[]

for name,model_cls in models.items():
    print(f"\n===== {name} =====")
    rows=[]; histories=[]; best_epochs=[]
    splits=list(skf.split(np.zeros(len(y)),y))
    for fold,(tr,va) in enumerate(splits,1):
        model,hist,best_epoch=train_one(model_cls,tr,va)

        p_train=predict_fold(model,tr)
        p_val=predict_fold(model,va)

        train_row=evaluate(y[tr],p_train)
        train_row.update(model=name,fold=fold,split="Train",best_epoch=best_epoch)

        val_row=evaluate(y[va],p_val)
        val_row.update(model=name,fold=fold,split="Validation",best_epoch=best_epoch)

        # Same fold-trained GNN predicts the held-out test set.
        test_loader=DataLoader(test_graphs,batch_size=BATCH_SIZE,shuffle=False)
        probs=[]; model.eval()
        with torch.no_grad():
            for batch in test_loader:
                probs.extend(torch.softmax(model(batch.to(DEVICE)),1)[:,1].cpu().numpy())
        test_row=evaluate(y_test,np.asarray(probs))
        test_row.update(model=name,fold=fold,split="Test",best_epoch=best_epoch)

        rows.extend([train_row,val_row,test_row])
        all_rows.extend([train_row,val_row,test_row])
        histories.append(hist); best_epochs.append(best_epoch)

        print(
            f"fold {fold:02d}: "
            f"Train ROC={train_row['ROC_AUC']:.3f} MCC={train_row['MCC']:.3f} | "
            f"Validation ROC={val_row['ROC_AUC']:.3f} MCC={val_row['MCC']:.3f} | "
            f"Test ROC={test_row['ROC_AUC']:.3f} MCC={test_row['MCC']:.3f}"
        )

    fdf=pd.DataFrame(rows); summary=summarize_cv(fdf,name)
    summary["CV_best_epoch_median"]=int(np.median(best_epochs))

    # Learning curve averaged over folds.
    tr_loss=np.array([h["train_loss"] for h in histories])
    va_loss=np.array([h["val_loss"] for h in histories])
    ep=np.arange(1,EPOCHS+1)
    plt.figure(figsize=(8,5))
    plt.plot(ep,tr_loss.mean(0),label="Training loss")
    plt.plot(ep,va_loss.mean(0),label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("Cross-entropy loss")
    plt.title(f"{name}: mean learning curve across 10 CV folds")
    plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS/f"04_{name}_epoch_vs_loss.png",dpi=180); plt.show()

    summaries.append(summary)

pd.DataFrame(all_rows).to_csv(RESULTS/"04_gnn_cv_folds.csv",index=False)
pd.DataFrame(summaries).to_csv(RESULTS/"04_gnn_summary.csv",index=False)
print("\nD-MPNN is intentionally excluded from the default 10-fold run because it is much slower.")
print("Use this notebook to compare the faster core architectures first.")
