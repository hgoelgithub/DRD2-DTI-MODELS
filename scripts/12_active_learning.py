"""
Pool-based active-learning simulation

Workflow
--------
Simulate pool-based acquisition within the development data: start with 80 active and 40 inactive labels, fit a forest, and acquire up to 100 predictions closest to 0.50 per round. Labels already exist in the dataset and are revealed by moving rows into the labeled pool. No CV or held-out performance evaluation runs here.
This file is self-contained and does not import project helper modules.
"""

# Workflow guide:
# Simulate pool-based acquisition within the development data: start with 80 active and 40
# inactive labels, fit a forest, and acquire up to 100 predictions closest to 0.50 per round.
# Labels already exist in the dataset and are revealed by moving rows into the labeled pool.
# No CV or held-out performance evaluation runs here.

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
# SECTION: Morgan fingerprints
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
# SECTION: Active-learning loop
from sklearn.ensemble import RandomForestClassifier
X=morgan_matrix(train_df.SMILES); y=train_df.Activity.to_numpy(dtype=np.int64); rng=np.random.default_rng(SEED)
# Initialize a labeled subset containing both classes; the sample sizes assume enough rows per class.
pos=np.where(y==1)[0]; neg=np.where(y==0)[0]; initial=np.concatenate([rng.choice(pos,80,replace=False),rng.choice(neg,40,replace=False)])
# Keep labeled and unlabeled index sets disjoint; all acquisition occurs inside development data.
labeled=set(initial.tolist()); pool=set(range(len(y)))-labeled; rows=[]
# Refit on the growing labeled set each round before scoring the remaining pool.
for round_id in range(10):
    idx=np.array(sorted(labeled)); pool_idx=np.array(sorted(pool)); model=RandomForestClassifier(n_estimators=200,class_weight='balanced_subsample',n_jobs=-1,random_state=SEED+round_id); model.fit(X[idx],y[idx]); p=model.predict_proba(X[pool_idx])[:,1]
    # Acquire the predictions closest to 0.50 (uncertainty sampling).
    # Record pool sizes before acquisition, then reveal the chosen rows’ existing labels.
    take=pool_idx[np.argsort(np.abs(p-.5))[:min(100,len(pool_idx))]]; rows.append({'round':round_id,'labeled':len(labeled),'pool':len(pool)}); labeled.update(take.tolist()); pool.difference_update(take.tolist())
    if not pool: break
out=pd.DataFrame(rows); out.to_csv(RESULTS/'12_active_learning_history.csv',index=False); print(out.to_string(index=False))
