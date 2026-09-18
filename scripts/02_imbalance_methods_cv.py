"""
Imbalance-handling methods with 10-fold CV

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
# SECTION: Morgan fingerprints
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

FP_SIZE = 2048
MORGAN_RADIUS = 2
fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=FP_SIZE)

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
# SECTION: Compare resampling strategies without leakage
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

X=morgan_matrix(train_df.SMILES)
y=train_df.Activity.to_numpy(dtype=np.int64)
X_test=morgan_matrix(test_df.SMILES)
y_test=test_df.Activity.to_numpy(dtype=np.int64)

samplers={
    "No_resampling":None,
    "RandomOverSampler":RandomOverSampler(random_state=SEED),
    "RandomUnderSampler":RandomUnderSampler(random_state=SEED),
    "SMOTE":SMOTE(random_state=SEED,k_neighbors=5),
    "ADASYN":ADASYN(random_state=SEED,n_neighbors=5),
    "SMOTE_Tomek":SMOTETomek(random_state=SEED),
}
skf=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED)
fold_rows=[]; summaries=[]

for method,sampler in samplers.items():
    print(f"\n===== {method} =====")
    rows=[]
    for fold,(tr,va) in enumerate(skf.split(X,y),1):
        Xtr,ytr=X[tr],y[tr]
        if sampler is not None:
            Xtr,ytr=sampler.fit_resample(Xtr,ytr)  # IMPORTANT: only resample the 9 training folds
        model=LogisticRegression(max_iter=3000,solver="liblinear",random_state=SEED)
        model.fit(Xtr,ytr)
        # Training metrics are calculated on the ORIGINAL 90% fold,
        # not on duplicated or synthetic resampled rows.
        p_train=model.predict_proba(X[tr])[:,1]
        p_val=model.predict_proba(X[va])[:,1]

        train_row=evaluate(y[tr],p_train)
        train_row.update(model=method,fold=fold,split="Train")

        val_row=evaluate(y[va],p_val)
        val_row.update(model=method,fold=fold,split="Validation")

        p_test=model.predict_proba(X_test)[:,1]
        test_row=evaluate(y_test,p_test)
        test_row.update(model=method,fold=fold,split="Test")

        rows.extend([train_row,val_row,test_row])
        fold_rows.extend([train_row,val_row,test_row])

        print(
            f"fold {fold:02d}: "
            f"Train MCC={train_row['MCC']:.3f} | "
            f"Validation MCC={val_row['MCC']:.3f} | "
            f"Test MCC={test_row['MCC']:.3f}"
        )

    fdf=pd.DataFrame(rows)
    summary=summarize_cv(fdf,method)
    summaries.append(summary)

pd.DataFrame(fold_rows).to_csv(RESULTS/"02_imbalance_cv_folds.csv",index=False)
pd.DataFrame(summaries).to_csv(RESULTS/"02_imbalance_summary.csv",index=False)
print("\nImportant caveat: SMOTE/ADASYN interpolate fingerprint vectors; those interpolated vectors are not literal molecules.")
print(pd.DataFrame(summaries).to_string(index=False))
