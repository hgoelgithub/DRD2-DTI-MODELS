"""
PyTorch MLP with 10-fold CV and learning curves

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

# Workflow guide:
# Build Morgan fingerprints and train a new MLP in each fold. Class weights come from that
# fold’s training labels. Record completed epochs, restore the lowest-validation-loss checkpoint,
# and then evaluate training, validation, and test predictions.

# SECTION: Imports, settings, and data
from pathlib import Path
import random
import numpy as np
import pandas as pd

# Set reproducible pseudo-random seeds; hardware and library differences can still affect results.
SEED = 42
N_SPLITS = 10
# Use the same active-class cutoff throughout this workflow; this is not a tuned threshold.
THRESHOLD = 0.50

random.seed(SEED)
np.random.seed(SEED)

# Locate the project from script or notebook execution; notebooks may not define __file__.
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
# SECTION: Morgan fingerprints
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

FP_SIZE = 2048
MORGAN_RADIUS = 2
# Initialize the shared fingerprint generator; identical settings are used for development and test molecules.
fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=FP_SIZE)

# Create one fixed-length binary fingerprint per SMILES, preserving row order.
# Radius 2 describes local atom neighborhoods; hashed bits can represent multiple fragments.
# Fail on invalid SMILES instead of silently training on an all-zero placeholder.
def morgan_matrix(smiles):
    X = np.zeros((len(smiles), FP_SIZE), dtype=np.uint8)
    invalid = []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            invalid.append(i)
            continue
        fp = fp_gen.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, X[i])
    if invalid:
        raise ValueError(f"Invalid SMILES at rows {invalid[:10]}")
    return X
# SECTION: Evaluation metrics
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, recall_score, precision_score,
    brier_score_loss, confusion_matrix
)

# Compare binary labels (0 inactive, 1 active) with P(active).
# ROC AUC measures ranking; PR_AUC keys use average precision, not trapezoidal area.
# MCC and balanced accuracy summarize label predictions; Brier measures probability error.
# Reversing labels/probabilities lets the same metrics describe the inactive class.
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

# Aggregate metric means and sample standard deviations (ddof=1).
# When split-specific rows exist, keep Train, Validation, and Test summaries separate.
# Repeated test predictions come from different models on the same molecules.
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
# SECTION: PyTorch model and training utilities
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

torch.manual_seed(SEED)
# Prefer CUDA, then Apple MPS when available, and otherwise use the CPU.
DEVICE=torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends,"mps") and torch.backends.mps.is_available() else "cpu"))
EPOCHS = 500
BATCH_SIZE=128

X=morgan_matrix(train_df.SMILES).astype(np.float32)
y=train_df.Activity.to_numpy(dtype=np.int64)
X_test=morgan_matrix(test_df.SMILES).astype(np.float32)
y_test=test_df.Activity.to_numpy(dtype=np.int64)

# Map fingerprint features through two hidden layers to two raw class scores (logits).
# CrossEntropyLoss accepts logits directly; use softmax only when probabilities are needed.
class MLP(nn.Module):
    def __init__(self,input_dim=FP_SIZE):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_dim,512),nn.ReLU(),nn.Dropout(0.25),
            nn.Linear(512,128),nn.ReLU(),nn.Dropout(0.20),
            nn.Linear(128,2)
        )
    def forward(self,x): return self.net(x)

# Calculate inverse-frequency loss weights from the training fold; validation labels do not set weights.
def class_weights(labels):
    counts=np.bincount(labels,minlength=2).astype(np.float32)
    return torch.tensor(len(labels)/(2*np.maximum(counts,1)),dtype=torch.float32,device=DEVICE)

# Initialize a new network and optimizer, train on this fold, and select the checkpoint by validation loss.
# Stop after three epochs without a meaningful validation-loss improvement.
# EPOCHS remains the maximum budget; test metrics never control stopping.
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-4

class EarlyStopping:
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA):
        if patience < 1 or min_delta < 0:
            raise ValueError("patience must be positive and min_delta nonnegative")
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.wait = 0

    def update(self, loss):
        if not np.isfinite(loss):
            raise RuntimeError("Non-finite validation loss")
        if loss < self.best - self.min_delta:
            self.best = loss
            self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience

def mean_learning_curve(histories, key):
    # Average only folds that actually reached an epoch; never invent later losses.
    # Later points can therefore represent fewer folds than earlier points.
    length = max(len(h[key]) for h in histories)
    padded = np.full((len(histories), length), np.nan)
    for i, history in enumerate(histories):
        padded[i, :len(history[key])] = history[key]
    return np.nanmean(padded, axis=0)

def run_fold(Xtr,ytr,Xva,yva,epochs=EPOCHS):
    model=MLP().to(DEVICE)
    loss_fn=nn.CrossEntropyLoss(weight=class_weights(ytr))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)

    tr_loader=DataLoader(
        TensorDataset(torch.from_numpy(Xtr.astype(np.float32)),torch.from_numpy(ytr.astype(np.int64))),
        batch_size=BATCH_SIZE,shuffle=True
    )
    Xva_t=torch.from_numpy(Xva.astype(np.float32)).to(DEVICE)
    yva_t=torch.from_numpy(yva.astype(np.int64)).to(DEVICE)

    history={"train_loss":[],"val_loss":[]}
    best_state=None; best_loss=float("inf"); best_epoch=1

    stopper = EarlyStopping()
    for epoch in range(1,epochs+1):
        model.train(); total=0.0; n=0
        for xb,yb in tr_loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            # Calculate supervised loss, backpropagate gradients, and update model parameters.
            loss=loss_fn(model(xb),yb)
            loss.backward(); opt.step()
            total+=loss.item()*len(yb); n+=len(yb)
        train_loss=total/n

        model.eval()
        with torch.no_grad():
            logits=model(Xva_t)
            val_loss=loss_fn(logits,yva_t).item()
            # Score the development validation subset with the current fold model.
            p_val=torch.softmax(logits,dim=1)[:,1].cpu().numpy()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        # Save the parameters at the lowest validation loss observed so far.
        if val_loss<best_loss:
            best_loss=val_loss; best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        # Checkpoint selection above still keeps the absolute lowest loss.
        if stopper.update(val_loss):
            print(f"Early stopping after {len(history['val_loss'])} epochs; restoring best checkpoint.")
            break

    # Use the lowest-validation-loss model, which may precede the final epoch.
    model.load_state_dict(best_state)
    model.eval()

    # Probabilities from the best checkpoint for BOTH CV splits.
    Xtr_t=torch.from_numpy(Xtr.astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        # Score the original training subset; these fitted-data metrics are not estimates of unseen performance.
        p_train=torch.softmax(model(Xtr_t),dim=1)[:,1].cpu().numpy()
        # Score the development validation subset with the current fold model.
        p_val=torch.softmax(model(Xva_t),dim=1)[:,1].cpu().numpy()

    return model,p_train,p_val,history,best_epoch

# SECTION: 10-fold CV
# Preserve class proportions when splitting development rows; this is not scaffold-grouped CV.
skf=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED)
fold_rows=[]; histories=[]; best_epochs=[]

for fold,(tr,va) in enumerate(skf.split(X,y),1):
    print(f"\nFold {fold}/{N_SPLITS}")
    model,p_train,p_val,history,best_epoch=run_fold(X[tr],y[tr],X[va],y[va])

    train_row=evaluate(y[tr],p_train)
    train_row.update(model="PyTorch_MLP",fold=fold,split="Train",best_epoch=best_epoch)

    val_row=evaluate(y[va],p_val)
    val_row.update(model="PyTorch_MLP",fold=fold,split="Validation",best_epoch=best_epoch)

    with torch.no_grad():
        p_test=torch.softmax(
            model(torch.from_numpy(X_test.astype(np.float32)).to(DEVICE)),dim=1
        )[:,1].cpu().numpy()
    test_row=evaluate(y_test,p_test)
    test_row.update(model="PyTorch_MLP",fold=fold,split="Test",best_epoch=best_epoch)

    fold_rows.extend([train_row,val_row,test_row])
    histories.append(history); best_epochs.append(best_epoch)

    print(
        f"Train ROC={train_row['ROC_AUC']:.3f} MCC={train_row['MCC']:.3f} | "
        f"Validation ROC={val_row['ROC_AUC']:.3f} MCC={val_row['MCC']:.3f} | "
        f"Test ROC={test_row['ROC_AUC']:.3f} MCC={test_row['MCC']:.3f} | "
        f"best_epoch={best_epoch}"
    )

fold_df=pd.DataFrame(fold_rows)
summary=summarize_cv(fold_df,"PyTorch_MLP")
summary["CV_best_epoch_median"]=int(np.median(best_epochs))

# SECTION: Plot epoch vs training/validation loss
train_losses=mean_learning_curve(histories, 'train_loss')
val_losses=mean_learning_curve(histories, 'val_loss')
epochs=np.arange(1, max(len(h["train_loss"]) for h in histories) + 1)
plt.figure(figsize=(8,5))
plt.plot(epochs,train_losses,label="Training loss")
plt.plot(epochs,val_losses,label="Validation loss")
plt.xlabel("Epoch"); plt.ylabel("Cross-entropy loss")
plt.title("MLP: mean learning curve across 10 CV folds")
plt.legend(); plt.tight_layout()
plt.savefig(RESULTS/"03_mlp_epoch_vs_loss.png",dpi=180)
plt.show()

# SECTION: Save fold-level and mean±SD results
# Export fold-level results without adding a pandas index column.
fold_df.to_csv(RESULTS/"03_mlp_cv_folds.csv",index=False)
pd.DataFrame([summary]).to_csv(RESULTS/"03_mlp_summary.csv",index=False)
print(pd.DataFrame([summary]).to_string(index=False))
