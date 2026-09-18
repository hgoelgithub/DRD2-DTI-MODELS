# DRD2 Scientific AI — 10-fold CV 

## Core design

This version uses only two modeling files:

- `data/D2_training_set_Ki.csv` — development data.
- `data/D2_test_scaffold_split_Ki.csv` — held-out final test data.

There is **no separate validation CSV**. Each supervised model uses:

```python
StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
```

Within each fold, 90% of the development data are used for training and 10% for validation. The held-out test file is not used in cross-validation.

## Self-contained files

The main notebooks/scripts deliberately repeat the small amount of code they need. They do not import project-specific `src` or `utils` modules. This keeps data loading, features, metrics, CV, model definition, training, and saving together in one place.

## Learning curves

The MLP, GNN, SMILES Transformer, ChemBERTa, MoLFormer, SELFormer, and ESM-2 conditioning workflows plot and save **epoch vs training loss and validation loss**. This makes overfitting/underfitting visible.

- training and validation loss both high -> likely underfitting
- both decrease and remain close -> healthier fit
- training loss keeps dropping while validation loss rises -> overfitting

## GNN runtime

`04_gnn_variants_cv` runs **GCN + GIN** by default with a compact 25-epoch schedule. Set `RUN_EXTENDED_VARIANTS=True` to add GraphSAGE and GAT. D-MPNN is intentionally left out of the default 10-fold run because it was the major runtime bottleneck; 10-fold CV would multiply that cost.

## Pretrained encoders

ChemBERTa (`06`) fine-tunes its final two transformer blocks in each CV fold and uses LayerNorm before its classifier head. Each fold starts from the pretrained encoder; earlier blocks remain frozen. Tokenization is shared, but cached frozen embeddings are not used. Its outputs use the `06_ChemBERTa_finetuned` prefix to preserve the earlier frozen baseline. Fine-tuning takes substantially more compute than training on cached embeddings.

MoLFormer and SELFormer are used as frozen molecular encoders. Their embeddings are computed once per dataset and cached by that same file; a small PyTorch classifier head is then evaluated with 10-fold CV. No credential is hard-coded.

## Suggested order

`00` audit -> `01` traditional ML -> `02/02B` imbalance -> `03` MLP -> `04` GNN -> `05` SMILES Transformer -> `06-09` pretrained/protein models -> `10` benchmark -> `11-15` analyses/report.
