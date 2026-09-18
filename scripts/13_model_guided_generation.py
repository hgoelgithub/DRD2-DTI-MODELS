"""
Simple model-guided analog generation

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
# SECTION: Fit scorer and build a limited BRICS candidate set
from sklearn.ensemble import RandomForestClassifier
from rdkit.Chem import BRICS,Descriptors
X=morgan_matrix(train_df.SMILES); y=train_df.Activity.to_numpy(dtype=np.int64); model=RandomForestClassifier(n_estimators=300,class_weight='balanced_subsample',n_jobs=-1,random_state=SEED); model.fit(X,y)
seed_smiles=train_df.loc[train_df.Activity==1,'SMILES'].sample(min(150,(train_df.Activity==1).sum()),random_state=SEED); fragments=set()
for s in seed_smiles:
    mol=Chem.MolFromSmiles(s)
    if mol: fragments.update(BRICS.BRICSDecompose(mol))
frag_mols=[Chem.MolFromSmiles(x) for x in fragments if Chem.MolFromSmiles(x) is not None]; candidates=[]
for mol in BRICS.BRICSBuild(frag_mols,maxDepth=2):
    s=Chem.MolToSmiles(mol,canonical=True)
    if 150<=Descriptors.MolWt(mol)<=650: candidates.append(s)
    if len(candidates)>=500: break
if candidates:
    Xc=morgan_matrix(pd.Series(candidates)); p=model.predict_proba(Xc)[:,1]; out=pd.DataFrame({'SMILES':candidates,'predicted_active_probability':p}).sort_values('predicted_active_probability',ascending=False); out.to_csv(RESULTS/'13_generated_candidates.csv',index=False); print(out.head(20).to_string(index=False))
else: print('No candidates generated.')
