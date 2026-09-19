"""
DRD2 data audit and split checks

Workflow
--------
Audit the supplied splits without training a model: parse molecules, canonicalize SMILES, count duplicate structures, and compare scaffold keys across files. An empty Murcko scaffold is replaced with canonical SMILES, so the scaffold-key overlap is not a pure scaffold-overlap count for acyclic molecules.
This file is self-contained and does not import project helper modules.
"""

# Workflow guide:
# Audit the supplied splits without training a model: parse molecules, canonicalize SMILES,
# count duplicate structures, and compare scaffold keys across files. An empty Murcko
# scaffold is replaced with canonical SMILES, so the scaffold-key overlap is not a pure
# scaffold-overlap count for acyclic molecules.

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
# SECTION: Validate molecules and inspect overlap
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# Parse and standardize the molecular representation so equivalent input strings can match.
def canonical_smiles(s):
    mol = Chem.MolFromSmiles(str(s))
    return None if mol is None else Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

# Extract a ring/linker scaffold; use canonical SMILES when the scaffold is empty.
def scaffold_key(s):
    mol = Chem.MolFromSmiles(str(s))
    if mol is None:
        return None
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    # Acyclic molecules have an empty Murcko scaffold; use canonical SMILES as a unique key.
    return scaf if scaf else Chem.MolToSmiles(mol, canonical=True)

rows=[]
cache={}
for name, df in [("train",train_df),("test",test_df)]:
    can = df.SMILES.map(canonical_smiles)
    scaf = df.SMILES.map(scaffold_key)
    cache[name]=(can,scaf)
    rows.append({
        "dataset":name, "n":len(df),
        "active":int((df.Activity==1).sum()),
        "inactive":int((df.Activity==0).sum()),
        "active_fraction":df.Activity.mean(),
        "invalid_smiles":int(can.isna().sum()),
        "canonical_duplicates":int(can.duplicated().sum()),
        "unique_scaffold_keys":int(scaf.nunique(dropna=True))
    })

summary=pd.DataFrame(rows)
print("\nData audit")
print(summary.to_string(index=False))

train_can,train_scaf=cache["train"]
test_can,test_scaf=cache["test"]
print("\nCross-file overlap")
print("Canonical SMILES overlap:", len(set(train_can.dropna()) & set(test_can.dropna())))
print("Scaffold-key overlap:", len(set(train_scaf.dropna()) & set(test_scaf.dropna())))
print("\nThe supplied files are not re-split or modified by this script.")

summary.to_csv(RESULTS/"00_data_audit.csv",index=False)
