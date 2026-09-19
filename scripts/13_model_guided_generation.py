"""
13 | DRD2: Model-Guided Molecular Candidate Generation

Generate and prioritize DRD2 analog candidates using fragment recombination and a learned
activity scorer.

Method
------
Fit a random forest on all development molecules, derive BRICS fragments from sampled active
compounds, and recombine them at limited depth. Retain molecular weights from 150 to 650 and
score up to 500 candidates.

Outputs
-------
Saved under results/:
- 13_generated_candidates.csv (when candidates are generated)

Outcome and Interpretation
--------------------------
The ranked table contains candidate SMILES and predicted active-class probabilities. Scores
support prioritization but do not establish experimental activity, novelty, or synthetic
feasibility. If no candidate passes generation and filtering, no candidate CSV is written.
"""

# SECTION: Configuration and Input Data
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
# SECTION: Molecular Representation: Morgan Fingerprints
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
# SECTION: BRICS Generation and Activity Ranking
from sklearn.ensemble import RandomForestClassifier
from rdkit.Chem import BRICS,Descriptors
X=morgan_matrix(train_df.SMILES); y=train_df.Activity.to_numpy(dtype=np.int64); model=RandomForestClassifier(n_estimators=300,class_weight='balanced_subsample',n_jobs=-1,random_state=SEED); model.fit(X,y)
# Choose active development molecules as fragment sources; test molecules are not used as seeds.
seed_smiles=train_df.loc[train_df.Activity==1,'SMILES'].sample(min(150,(train_df.Activity==1).sum()),random_state=SEED); fragments=set()
for s in seed_smiles:
    mol=Chem.MolFromSmiles(s)
    if mol: fragments.update(BRICS.BRICSDecompose(mol))
# Parse the unique BRICS fragments into molecules that can be recombined.
frag_mols=[Chem.MolFromSmiles(x) for x in fragments if Chem.MolFromSmiles(x) is not None]; candidates=[]
# Enumerate limited-depth fragment recombinations; candidates are not guaranteed novel or synthesizable.
for mol in BRICS.BRICSBuild(frag_mols,maxDepth=2):
    s=Chem.MolToSmiles(mol,canonical=True)
    # Keep candidates in the specified molecular-weight range and stop at the candidate budget.
    if 150<=Descriptors.MolWt(mol)<=650: candidates.append(s)
    if len(candidates)>=500: break
if candidates:
    # Score generated fingerprints and sort by predicted activity; these predictions need experimental validation.
    Xc=morgan_matrix(pd.Series(candidates)); p=model.predict_proba(Xc)[:,1]; out=pd.DataFrame({'SMILES':candidates,'predicted_active_probability':p}).sort_values('predicted_active_probability',ascending=False); out.to_csv(RESULTS/'13_generated_candidates.csv',index=False); print(out.head(20).to_string(index=False))
else: print('No candidates generated.')