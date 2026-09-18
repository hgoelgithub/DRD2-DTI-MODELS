"""
Traditional machine-learning models with 10-fold CV

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
# SECTION: Define models
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from xgboost import XGBClassifier

X = morgan_matrix(train_df.SMILES)
y = train_df.Activity.to_numpy(dtype=np.int64)
X_test = morgan_matrix(test_df.SMILES)
y_test = test_df.Activity.to_numpy(dtype=np.int64)

models = {
    "LogisticRegression_balanced": LogisticRegression(
        max_iter=3000, solver="liblinear", class_weight="balanced", random_state=SEED),
    "RandomForest_balanced": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
        n_jobs=-1, random_state=SEED),
    "XGBoost_balanced": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        eval_metric="logloss", n_jobs=-1, random_state=SEED),
    "BernoulliNB": BernoulliNB(alpha=1.0),
    "MultinomialNB": MultinomialNB(alpha=1.0),
}

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
all_folds, summaries = [], []

def balanced_sample_weights(labels):
    labels=np.asarray(labels)
    n=len(labels)
    n0=max((labels==0).sum(),1); n1=max((labels==1).sum(),1)
    return np.where(labels==0, n/(2*n0), n/(2*n1))

# SECTION: 10-fold cross-validation
for model_name, template in models.items():
    print(f"\n===== {model_name} =====")
    fold_rows=[]
    for fold,(tr,va) in enumerate(skf.split(X,y),1):
        model=clone(template)
        if model_name=="XGBoost_balanced":
            model.fit(X[tr],y[tr],sample_weight=balanced_sample_weights(y[tr]))
        else:
            model.fit(X[tr],y[tr])
        # Evaluate both the 90% training portion and the 10% validation portion.
        p_train=model.predict_proba(X[tr])[:,1]
        p_val=model.predict_proba(X[va])[:,1]

        train_row=evaluate(y[tr],p_train)
        train_row.update(model=model_name,fold=fold,split="Train")

        val_row=evaluate(y[va],p_val)
        val_row.update(model=model_name,fold=fold,split="Validation")

        # The SAME fold-trained model also predicts the untouched held-out test set.
        p_test=model.predict_proba(X_test)[:,1]
        test_row=evaluate(y_test,p_test)
        test_row.update(model=model_name,fold=fold,split="Test")

        fold_rows.extend([train_row,val_row,test_row])
        all_folds.extend([train_row,val_row,test_row])

        print(
            f"fold {fold:02d}: "
            f"Train ROC={train_row['ROC_AUC']:.3f} MCC={train_row['MCC']:.3f} | "
            f"Validation ROC={val_row['ROC_AUC']:.3f} MCC={val_row['MCC']:.3f} | "
            f"Test ROC={test_row['ROC_AUC']:.3f} MCC={test_row['MCC']:.3f}"
        )

    fold_df=pd.DataFrame(fold_rows)
    summary=summarize_cv(fold_df,model_name)
    summaries.append(summary)

pd.DataFrame(all_folds).to_csv(RESULTS/"01_traditional_ml_cv_folds.csv",index=False)
summary_df=pd.DataFrame(summaries)
summary_df.to_csv(RESULTS/"01_traditional_ml_summary.csv",index=False)
print("\nCV mean±SD and held-out test results")
print(summary_df.to_string(index=False))
