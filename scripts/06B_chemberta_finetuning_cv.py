"""
06B | DRD2: ChemBERTa Partial Fine-Tuning

Evaluate task adaptation of ChemBERTa for DRD2 activity classification.

Method
------
Initialize DeepChem/ChemBERTa-100M-MLM independently for each fold, update its final encoder
blocks and a LayerNorm/MLP head, and select checkpoints using validation loss.

Evaluation Design
-----------------
Use D2_training_set_Ki.csv for development and reserve D2_test_scaffold_split_Ki.csv for held-
out testing. Ten shuffled, stratified folds (seed 42) split development rows into 90% training
and 10% validation; these folds are not scaffold-grouped. Each fold model evaluates the same
held-out test molecules. Threshold-based metrics use 0.50, and summaries report means and
standard deviations across fold models. Test variability describes different fitted models on
one fixed test set, not independent test datasets.

Training Control
----------------
Validation loss controls early stopping (patience 3; minimum improvement 0.0001). Each fold
restores its lowest-loss checkpoint. Learning curves average only folds that completed a given
epoch.

Outputs
-------
Saved under results/:
- 06B_ChemBERTa_finetuned_cv_folds.csv
- 06B_ChemBERTa_finetuned_summary.csv
- 06B_ChemBERTa_finetuned_epoch_vs_loss.png

Outcome and Interpretation
--------------------------
Compare these outputs with workflow 06 under the same evaluation design to assess whether
partial fine-tuning improves validation performance. This workflow tokenizes molecules directly
and does not reuse frozen embedding caches.
"""

# SECTION: Configuration and Input Data
from pathlib import Path
import random
import numpy as np
import pandas as pd

# Step 1: Configure repeatability, the number of CV folds, and the decision cutoff.
# Activity is assumed to be binary: 0 = inactive, 1 = active. Seeds help
# repeatability, but do not guarantee identical results on every device.
SEED=42
N_SPLITS=10
THRESHOLD=0.50
random.seed(SEED); np.random.seed(SEED)

# Find the data directory from either a script or a notebook working directory.
# Notebooks generally do not define __file__, so the current directory is also checked.
def project_root():
    candidates=[Path.cwd(),Path.cwd().parent]
    if "__file__" in globals(): candidates.insert(0,Path(__file__).resolve().parents[1])
    for p in candidates:
        if (p/"data"/"D2_training_set_Ki.csv").exists(): return p
    raise FileNotFoundError("Run from the project root, scripts/, or notebooks/.")

# Create the output folder and load the supplied development and held-out datasets.
# The column check below verifies names; it does not validate all label values.
ROOT=project_root(); RESULTS=ROOT/"results"; RESULTS.mkdir(exist_ok=True)
train_df=pd.read_csv(ROOT/"data"/"D2_training_set_Ki.csv")
test_df=pd.read_csv(ROOT/"data"/"D2_test_scaffold_split_Ki.csv")
for name,df in [("training",train_df),("test",test_df)]:
    if not {"SMILES","Activity"}.issubset(df.columns): raise ValueError(f"{name} file must contain SMILES and Activity")
print(f"Training: {train_df.shape} | active fraction={train_df.Activity.mean():.4f}")
print(f"Test:     {test_df.shape} | active fraction={test_df.Activity.mean():.4f}")
# SECTION: Classification Metrics
from sklearn.metrics import roc_auc_score,average_precision_score,matthews_corrcoef,balanced_accuracy_score,recall_score,precision_score,brier_score_loss,confusion_matrix

# Step 2: Score predicted probabilities and thresholded class predictions.
# ROC AUC measures ranking; average precision summarizes precision/recall
# (the PR_AUC keys here use average precision, not trapezoidal integration).
# MCC summarizes the confusion matrix; balanced accuracy averages class recalls.
# Recall measures recovery of a class, precision measures prediction correctness.
# Brier is mean squared probability error; lower values are better.
def evaluate(y_true,p_active,threshold=THRESHOLD):
    y_true=np.asarray(y_true,dtype=int); p_active=np.asarray(p_active,dtype=float)
    # Convert active probabilities to labels using the same fixed cutoff for every fold.
    y_pred=(p_active>=threshold).astype(int)
    # With labels [0, 1], entries are true negatives, false positives,
    # false negatives, and true positives. Inactive metrics treat class 0 as positive.
    tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    return {"ROC_AUC":roc_auc_score(y_true,p_active),"PR_AUC_active":average_precision_score(y_true,p_active),"PR_AUC_inactive":average_precision_score(1-y_true,1-p_active),"MCC":matthews_corrcoef(y_true,y_pred),"BalancedAcc":balanced_accuracy_score(y_true,y_pred),"Recall_active":recall_score(y_true,y_pred,pos_label=1,zero_division=0),"Recall_inactive":recall_score(y_true,y_pred,pos_label=0,zero_division=0),"Precision_active":precision_score(y_true,y_pred,pos_label=1,zero_division=0),"Precision_inactive":precision_score(y_true,y_pred,pos_label=0,zero_division=0),"Brier":brier_score_loss(y_true,p_active),"threshold":threshold,"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp)}

def summarize_cv(df, model_name):
    """Summarize Train, Validation, and Test metrics across the 10 fold-trained models."""
    cols = [
        "ROC_AUC", "PR_AUC_active", "PR_AUC_inactive", "MCC", "BalancedAcc",
        "Recall_active", "Recall_inactive", "Precision_active", "Precision_inactive", "Brier"
    ]
    # Report each metric separately for Train, Validation, and Test.
    # The standard deviation uses ddof=1 (sample SD across fold models).
    # Test rows reuse the same molecules, so these are not independent test datasets.
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
# SECTION: Pretrained Encoder and SMILES Tokenization
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

# Step 3: Fine-tune only the last two transformer blocks. Earlier blocks stay frozen.
# Use a smaller encoder learning rate to avoid large changes to pretrained weights.
MODEL_ID = "DeepChem/ChemBERTa-100M-MLM"
TRUST_REMOTE_CODE = False
UNFREEZE_LAST_N = 2
ENCODER_LR = 2e-5
HEAD_LR = 1e-3
HEAD_HIDDEN = 256
HEAD_DROPOUT = 0.25
WEIGHT_DECAY = 1e-4
EPOCHS = 500
BATCH_SIZE = 16
MAX_LENGTH = 256
MODEL_NAME = "ChemBERTa_finetuned_LayerNorm"
OUTPUT_PREFIX = "06B_ChemBERTa_finetuned"
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else (
    "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))

# Tokenization has no fitted statistics and may be shared across folds.
# Old frozen-embedding caches cannot be used: embeddings now change during training.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=TRUST_REMOTE_CODE)

def tokenize(smiles):
    return tokenizer(list(smiles.astype(str)), padding="max_length", truncation=True,
                     max_length=MAX_LENGTH, return_tensors="pt")

train_tokens = tokenize(train_df.SMILES)
test_tokens = tokenize(test_df.SMILES)
y = train_df.Activity.to_numpy(dtype=np.int64)
y_test = test_df.Activity.to_numpy(dtype=np.int64)

def mean_pool(hidden, mask):
    # Exclude padding from the token average; unmasked special tokens are included.
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

# SECTION: Normalized Classifier and Fold Training
class Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        # Step 4: Normalize each molecule's pooled embedding across its features.
        # LayerNorm needs no dataset-wide statistics. Its affine parameters are
        # learned only on the training fold, together with the classifier.
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, HEAD_HIDDEN),
                                 nn.ReLU(), nn.Dropout(HEAD_DROPOUT),
                                 nn.Linear(HEAD_HIDDEN, 2))

    def forward(self, x):
        return self.net(x)

class ChemBERTaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Reload the original pretrained weights for EVERY fold, preventing
        # training on one fold from leaking into the next fold's validation set.
        self.encoder = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=TRUST_REMOTE_CODE)
        blocks = self.encoder.encoder.layer  # RoBERTa blocks in this ChemBERTa model.
        if not 1 <= UNFREEZE_LAST_N <= len(blocks):
            raise ValueError("UNFREEZE_LAST_N must be between 1 and the number of encoder layers")
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        for block in blocks[-UNFREEZE_LAST_N:]:
            for param in block.parameters():
                param.requires_grad_(True)
        self.head = Head(self.encoder.config.hidden_size)

    def train(self, mode=True):
        super().train(mode)
        # Keep dropout disabled in frozen blocks, while allowing dropout in the
        # trainable final blocks and head during training.
        self.encoder.eval()
        if mode:
            for block in self.encoder.encoder.layer[-UNFREEZE_LAST_N:]:
                block.train()
        return self

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Do not use no_grad here: final encoder blocks must receive gradients.
        return self.head(mean_pool(hidden, attention_mask))

def class_weights(labels):
    # Preserve the existing inverse-frequency class weights, using training labels only.
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    return torch.tensor(len(labels) / (2 * np.maximum(counts, 1)),
                        dtype=torch.float32, device=DEVICE)

def make_loader(tokens, labels, indices, shuffle=False):
    indices = torch.as_tensor(indices, dtype=torch.long)
    dataset = TensorDataset(tokens["input_ids"][indices], tokens["attention_mask"][indices],
                            torch.as_tensor(labels, dtype=torch.long)[indices])
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

def predict(model, loader, loss_fn=None):
    # Step 5: Batch ALL evaluation, including the test set, to bound device memory.
    model.eval()
    probabilities = []
    loss_sum = weight_sum = 0.0
    with torch.no_grad():
        for ids, mask, labels in loader:
            ids, mask, labels = ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
            logits = model(ids, mask)
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            if loss_fn is not None:
                # Exact weighted mean over the complete split, independent of batch sizes.
                batch_weight = loss_fn.weight[labels].sum().item()
                loss_sum += loss_fn(logits, labels).item() * batch_weight
                weight_sum += batch_weight
    return np.concatenate(probabilities), loss_sum / weight_sum if weight_sum else None

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

def train_fold(tr, va):
    # Step 6: Fresh encoder/head/optimizer and validation-selected checkpoint per fold.
    model = ChemBERTaClassifier().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y[tr]))
    opt = torch.optim.AdamW([
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": ENCODER_LR},
        {"params": model.head.parameters(), "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)
    loader = make_loader(train_tokens, y, tr, shuffle=True)
    val_loader = make_loader(train_tokens, y, va)
    hist = {"train_loss": [], "val_loss": []}
    best = None
    best_loss = float("inf")
    best_epoch = 1
    stopper = EarlyStopping()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = weight_total = 0.0
        for ids, mask, labels in loader:
            ids, mask, labels = ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(ids, mask), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            batch_weight = loss_fn.weight[labels].sum().item()
            total += loss.item() * batch_weight
            weight_total += batch_weight
        _, val_loss = predict(model, val_loader, loss_fn)
        hist["train_loss"].append(total / weight_total)
        hist["val_loss"].append(val_loss)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch:02d}/{EPOCHS}: train={hist['train_loss'][-1]:.4f}, val={val_loss:.4f}")
        # Checkpoint selection above still keeps the absolute lowest loss.
        if stopper.update(val_loss):
            print(f"Early stopping after {len(hist['val_loss'])} epochs; restoring best checkpoint.")
            break
    if best is None:
        raise RuntimeError("No finite validation checkpoint was produced")
    model.load_state_dict(best)
    p_train, _ = predict(model, make_loader(train_tokens, y, tr))
    p_val, _ = predict(model, val_loader)
    return model, p_train, p_val, hist, best_epoch

# SECTION: Stratified 10-Fold Evaluation
# Step 7: Split only the development data into 10 folds, preserving class proportions.
# Each run trains on 9 folds and validates on the remaining fold. These CV splits
# are stratified by activity, not grouped by molecular scaffold.
skf=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED)
rows=[]; histories=[]; best_epochs=[]
for fold,(tr,va) in enumerate(skf.split(np.zeros(len(y)),y),1):
    # Keep the fitted fold model, its predictions, loss history, and selected epoch.
    model,p_train,p_val,h,b=train_fold(tr,va)

    train_row=evaluate(y[tr],p_train)
    train_row.update(model=MODEL_NAME,fold=fold,split="Train",best_epoch=b)

    val_row=evaluate(y[va],p_val)
    val_row.update(model=MODEL_NAME,fold=fold,split="Validation",best_epoch=b)

    # Test labels are used only for reporting, never for checkpoint selection.
    p_test, _ = predict(model, make_loader(test_tokens, y_test, np.arange(len(y_test))))
    test_row=evaluate(y_test,p_test)
    test_row.update(model=MODEL_NAME,fold=fold,split="Test",best_epoch=b)

    # Save three metric rows per fold and retain histories for the learning-curve plot.
    rows.extend([train_row,val_row,test_row])
    histories.append(h); best_epochs.append(b)
    del model  # Release this fold before constructing the next encoder.

    print(
        f"fold {fold:02d}: "
        f"Train ROC={train_row['ROC_AUC']:.3f} MCC={train_row['MCC']:.3f} | "
        f"Validation ROC={val_row['ROC_AUC']:.3f} MCC={val_row['MCC']:.3f} | "
        f"Test ROC={test_row['ROC_AUC']:.3f} MCC={test_row['MCC']:.3f} | "
        f"best_epoch={b}"
    )
# Summarize fold means and sample SDs. Median best epoch is reported only;
# the code does not use it to train another model.
fold_df=pd.DataFrame(rows); summary=summarize_cv(fold_df,MODEL_NAME); summary["CV_best_epoch_median"]=int(np.median(best_epochs))

# SECTION: Training and Validation Learning Curves
# Step 8: Form [folds, epochs] loss arrays and average across folds at each epoch.
# These curves show completed epochs only; later points may average fewer folds.
tr_loss=mean_learning_curve(histories, 'train_loss'); va_loss=mean_learning_curve(histories, 'val_loss'); ep=np.arange(1, max(len(h["train_loss"]) for h in histories) + 1)
plt.figure(figsize=(8,5)); plt.plot(ep,tr_loss,label="Training loss"); plt.plot(ep,va_loss,label="Validation loss")
plt.xlabel("Epoch"); plt.ylabel("Cross-entropy loss"); plt.title('Fine-tuned ChemBERTa: mean learning curve across 10 CV folds'); plt.legend(); plt.tight_layout()
# Save the learning-curve PNG before displaying it in the notebook.
plt.savefig(RESULTS/f"{OUTPUT_PREFIX}_epoch_vs_loss.png",dpi=180); plt.show()

# SECTION: Results Export
# Step 9: Export per-fold Train/Validation/Test metrics and one summary row.
# Separate filenames preserve the frozen baseline. Model weights are not saved.
fold_df.to_csv(RESULTS/f"{OUTPUT_PREFIX}_cv_folds.csv",index=False)
pd.DataFrame([summary]).to_csv(RESULTS/f"{OUTPUT_PREFIX}_summary.csv",index=False)
print(pd.DataFrame([summary]).to_string(index=False))