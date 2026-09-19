"""
SELFormer frozen embeddings + MLP head with 10-fold CV

Evaluation design
-----------------
1. D2_training_set_Ki.csv is the development dataset.
2. StratifiedKFold(n_splits=10, shuffle=True, random_state=42) is applied only to that training dataset.
3. Each fold trains on 9/10 of the development data and validates on 1/10.
4. D2_test_scaffold_split_Ki.csv is never part of cross-validation.
5. Each fold-trained model evaluates the same held-out test set; there is no final full-data refit.
6. Threshold-based metrics use one fixed threshold: 0.50.
7. This file is self-contained and does not import project helper modules.
"""

# Workflow guide:
# Convert SMILES to SELFIES before tokenization, extract frozen SELFormer embeddings, and
# train an MLP head independently per fold. The encoder stays fixed; validation loss selects
# the head checkpoint used for evaluation.

# SECTION: Imports, settings, and data
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
# SECTION: Metrics
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
# SECTION: Pretrained encoder
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from transformers import AutoTokenizer,AutoModel

MODEL_ID='HUBioDataLab/SELFormer'
TRUST_REMOTE_CODE=False
EPOCHS = 500
BATCH_SIZE=128
EMBED_BATCH_SIZE=16
MAX_LENGTH=256
torch.manual_seed(SEED)
# Prefer CUDA, then Apple MPS when available, and otherwise use the CPU.
DEVICE=torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends,"mps") and torch.backends.mps.is_available() else "cpu"))

import selfies as sf
# Convert SMILES into SELFIES, the representation expected by this encoder.
def prepare_text(smiles):
    return sf.encoder(str(smiles))

# Average token vectors over non-padding positions.
# The output contains one vector per molecule; unmasked special tokens are included.
def mean_pool(hidden,mask):
    m=mask.unsqueeze(-1).float(); return (hidden*m).sum(1)/m.sum(1).clamp(min=1e-9)

# Reuse cached features only if the row count matches. This does not check molecule order
# or encoder settings: changed inputs/settings require regenerating the cache.
def extract_embeddings(smiles,cache_path):
    if cache_path.exists():
        arr=np.load(cache_path)
        if len(arr)==len(smiles):
            print("Loaded cached embeddings:",cache_path.name); return arr.astype(np.float32)
    texts=[prepare_text(s) for s in smiles]
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=TRUST_REMOTE_CODE)
    # Load the pretrained encoder in evaluation mode; only the downstream head is optimized in this workflow.
    encoder=AutoModel.from_pretrained(MODEL_ID,trust_remote_code=TRUST_REMOTE_CODE).to(DEVICE); encoder.eval()
    chunks=[]
    # Encode a small batch at a time to limit memory while retaining input row order.
    for start in range(0,len(texts),EMBED_BATCH_SIZE):
        b=texts[start:start+EMBED_BATCH_SIZE]
        tok=tokenizer(b,padding=True,truncation=True,max_length=MAX_LENGTH,return_tensors="pt")
        tok={k:v.to(DEVICE) for k,v in tok.items()}
        # Disable gradient recording for frozen feature extraction and pool token outputs.
        with torch.no_grad(): out=encoder(**tok); pooled=mean_pool(out.last_hidden_state,tok["attention_mask"])
        chunks.append(pooled.cpu().numpy().astype(np.float32))
        print(f"encoded {min(start+EMBED_BATCH_SIZE,len(texts))}/{len(texts)}")
    # Stack batches into a [molecules, embedding dimensions] array and save it for reuse.
    arr=np.concatenate(chunks); np.save(cache_path,arr); return arr

X=extract_embeddings(train_df.SMILES,RESULTS/'08_SELFormer_train_embeddings.npy')
X_test=extract_embeddings(test_df.SMILES,RESULTS/'08_SELFormer_test_embeddings.npy')
y=train_df.Activity.to_numpy(dtype=np.int64); y_test=test_df.Activity.to_numpy(dtype=np.int64)

# SECTION: Supervised classification head
# Train the small supervised classifier on feature vectors; its two outputs are logits.
class Head(nn.Module):
    def __init__(self,d):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Dropout(.25),nn.Linear(256,2))
    def forward(self,x): return self.net(x)

# Calculate inverse-frequency loss weights from the training fold; validation labels do not set weights.
def class_weights(labels):
    c=np.bincount(labels,minlength=2).astype(np.float32)
    return torch.tensor(len(labels)/(2*np.maximum(c,1)),dtype=torch.float32,device=DEVICE)

# Keep training and checkpoint selection within the current development fold.
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

def train_fold(tr,va):
    model=Head(X.shape[1]).to(DEVICE); loss_fn=nn.CrossEntropyLoss(weight=class_weights(y[tr])); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    loader=DataLoader(TensorDataset(torch.from_numpy(X[tr]),torch.from_numpy(y[tr])),batch_size=BATCH_SIZE,shuffle=True)
    xv=torch.from_numpy(X[va]).to(DEVICE); yv=torch.from_numpy(y[va]).to(DEVICE)
    hist={"train_loss":[],"val_loss":[]}; best=None; best_loss=float("inf"); best_epoch=1
    stopper = EarlyStopping()
    for ep in range(1,EPOCHS+1):
        model.train(); total=0.; n=0
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad(set_to_none=True)
            # Calculate supervised loss, backpropagate gradients, and update model parameters.
            loss=loss_fn(model(xb),yb); loss.backward(); opt.step(); total+=loss.item()*len(yb); n+=len(yb)
        hist["train_loss"].append(total/n)
        model.eval()
        with torch.no_grad(): logits=model(xv); vl=loss_fn(logits,yv).item()
        hist["val_loss"].append(vl)
        # Retain a copy of the best validation checkpoint, not merely a reference to live parameters.
        if vl<best_loss:
            best_loss=vl; best_epoch=ep; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        # Checkpoint selection above still keeps the absolute lowest loss.
        if stopper.update(vl):
            print(f"Early stopping after {len(hist['val_loss'])} epochs; restoring best checkpoint.")
            break
    # Restore the chosen checkpoint before reporting predictions.
    model.load_state_dict(best); model.eval()
    xt=torch.from_numpy(X[tr]).to(DEVICE)
    with torch.no_grad():
        # Score the original training subset; these fitted-data metrics are not estimates of unseen performance.
        p_train=torch.softmax(model(xt),1)[:,1].cpu().numpy()
        # Score the development validation subset with the current fold model.
        p_val=torch.softmax(model(xv),1)[:,1].cpu().numpy()
    return model,p_train,p_val,hist,best_epoch

# SECTION: 10-fold cross-validation
# Preserve class proportions when splitting development rows; this is not scaffold-grouped CV.
skf=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED)
rows=[]; histories=[]; best_epochs=[]
for fold,(tr,va) in enumerate(skf.split(X,y),1):
    model,p_train,p_val,h,b=train_fold(tr,va)

    train_row=evaluate(y[tr],p_train)
    train_row.update(model='SELFormer',fold=fold,split="Train",best_epoch=b)

    val_row=evaluate(y[va],p_val)
    val_row.update(model='SELFormer',fold=fold,split="Validation",best_epoch=b)

    with torch.no_grad():
        p_test=torch.softmax(model(torch.from_numpy(X_test).to(DEVICE)),1)[:,1].cpu().numpy()
    test_row=evaluate(y_test,p_test)
    test_row.update(model='SELFormer',fold=fold,split="Test",best_epoch=b)

    # Store one row per split per fold, keeping model and fold identifiers for later summaries.
    rows.extend([train_row,val_row,test_row])
    histories.append(h); best_epochs.append(b)

    print(
        f"fold {fold:02d}: "
        f"Train ROC={train_row['ROC_AUC']:.3f} MCC={train_row['MCC']:.3f} | "
        f"Validation ROC={val_row['ROC_AUC']:.3f} MCC={val_row['MCC']:.3f} | "
        f"Test ROC={test_row['ROC_AUC']:.3f} MCC={test_row['MCC']:.3f} | "
        f"best_epoch={b}"
    )
fold_df=pd.DataFrame(rows); summary=summarize_cv(fold_df,'SELFormer'); summary["CV_best_epoch_median"]=int(np.median(best_epochs))

# SECTION: Epoch vs loss plot
tr_loss=mean_learning_curve(histories, 'train_loss'); va_loss=mean_learning_curve(histories, 'val_loss'); ep=np.arange(1, max(len(h["train_loss"]) for h in histories) + 1)
plt.figure(figsize=(8,5)); plt.plot(ep,tr_loss,label="Training loss"); plt.plot(ep,va_loss,label="Validation loss")
plt.xlabel("Epoch"); plt.ylabel("Cross-entropy loss"); plt.title('SELFormer: mean learning curve across 10 CV folds'); plt.legend(); plt.tight_layout()
plt.savefig(RESULTS/'08_SELFormer_epoch_vs_loss.png',dpi=180); plt.show()

# SECTION: Save fold-level and mean±SD results
# Export fold-level results without adding a pandas index column.
fold_df.to_csv(RESULTS/'08_SELFormer_cv_folds.csv',index=False)
pd.DataFrame([summary]).to_csv(RESULTS/'08_SELFormer_summary.csv',index=False)
print(pd.DataFrame([summary]).to_string(index=False))
