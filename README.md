# DRD2 Scientific AI

**Molecular activity classification, representation learning, and model-guided analysis for the dopamine D2 receptor.**

This project benchmarks fingerprint classifiers, graph neural networks, SMILES transformers, and pretrained molecular encoders on binary DRD2 activity labels. It combines a shared evaluation protocol with downstream analyses of prediction uncertainty, structural coverage, active learning, and molecular candidate generation.

The repository contains **18 notebooks and 18 matching Python scripts**. Each workflow is self-contained, with its data loading, model configuration, analysis, and result export in one file.

## Project Structure

```text
.
├── data/          # Development data, scaffold-test data, and DRD2 protein sequence
├── notebooks/     # Interactive workflows with explanations and saved outputs
├── scripts/       # Corresponding standalone Python workflows
├── results/       # Metric tables, learning curves, embedding caches, and report
├── requirements.txt
└── README.md
```

## Data and Prediction Task

| File | Purpose |
| --- | --- |
| [D2_training_set_Ki.csv](data/D2_training_set_Ki.csv) | Development dataset used for training and cross-validation |
| [D2_test_scaffold_split_Ki.csv](data/D2_test_scaffold_split_Ki.csv) | Held-out scaffold test dataset |
| [DRD2_P14416-1.fasta](data/DRD2_P14416-1.fasta) | DRD2 protein sequence used by the ESM-2 conditioning workflow |

The molecular CSVs must contain `SMILES` and `Activity`. The supplied activity labels are interpreted as **0 = inactive** and **1 = active**; model outputs represent the probability of the active class. These workflows perform classification rather than continuous Ki prediction.

Run the data audit first to inspect class balance, invalid SMILES, duplicate structures, and overlap between the supplied datasets. The audit preserves the input files and reports overlap separately from the per-dataset summary.

## Setup and Execution

From the project root, create an environment and install the dependencies in [requirements.txt](requirements.txt):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell. The dependency file pins core numerical and modeling libraries, including PyTorch `2.2.2` and Transformers `4.40.2`.

### Interactive Notebooks

```bash
jupyter lab
```

Open a notebook in `notebooks/`, select the project environment as its kernel, and execute its cells in order. Each notebook describes its purpose, method, outputs, and interpretation.

### Python Scripts

```bash
python scripts/00_data_audit.py
python scripts/01_traditional_ml_cv.py
```

Choose the modeling workflows relevant to the experiment. After generating model summaries, assemble the benchmark and report:

```bash
python scripts/10_benchmark_results.py
python scripts/15_technical_report.py
```

Configurations are defined within each file. Neural workflows prefer CUDA, then Apple MPS when available, and otherwise CPU. Pretrained workflows require access to Hugging Face for files that are not already cached; initial downloads and embedding extraction add to runtime.

## Workflow Catalog

### Data Audit and Predictive Models

| ID | Workflow | Method and primary outcome | Files |
| --- | --- | --- | --- |
| 00 | Data quality and split audit | Molecular validity, class distribution, duplicate counts, and structural overlap | [Notebook](notebooks/00_data_audit.ipynb) · [Script](scripts/00_data_audit.py) |
| 01 | Traditional machine learning | Morgan fingerprints with logistic regression, random forest, XGBoost, and two naive Bayes classifiers | [Notebook](notebooks/01_traditional_ml_cv.ipynb) · [Script](scripts/01_traditional_ml_cv.py) |
| 02 | Resampling methods | Six sampling strategies with a fixed logistic-regression classifier | [Notebook](notebooks/02_imbalance_methods_cv.ipynb) · [Script](scripts/02_imbalance_methods_cv.py) |
| 02B | Cost-sensitive learning | Random forests with increasing inactive-class penalties | [Notebook](notebooks/02B_cost_sensitive_cv.ipynb) · [Script](scripts/02B_cost_sensitive_cv.py) |
| 03 | Fingerprint MLP | PyTorch classifier on Morgan fingerprints, with learning curves | [Notebook](notebooks/03_pytorch_mlp_cv.ipynb) · [Script](scripts/03_pytorch_mlp_cv.py) |
| 04 | Graph neural networks | GCN and GIN by default; optional GraphSAGE and GAT | [Notebook](notebooks/04_gnn_variants_cv.ipynb) · [Script](scripts/04_gnn_variants_cv.py) |
| 05 | SMILES transformer | Character-level transformer trained from scratch | [Notebook](notebooks/05_smiles_transformer_cv.ipynb) · [Script](scripts/05_smiles_transformer_cv.py) |
| 06 | Frozen ChemBERTa | Cached molecular embeddings with a LayerNorm/MLP classifier | [Notebook](notebooks/06_chemberta_cv.ipynb) · [Script](scripts/06_chemberta_cv.py) |
| 06B | ChemBERTa fine-tuning | Adapt the final encoder blocks and normalized classifier per fold | [Notebook](notebooks/06B_chemberta_finetuning_cv.ipynb) · [Script](scripts/06B_chemberta_finetuning_cv.py) |
| 07 | Frozen MoLFormer | IBM molecular embeddings with an MLP classifier | [Notebook](notebooks/07_molformer_cv.ipynb) · [Script](scripts/07_molformer_cv.py) |
| 08 | Frozen SELFormer | SELFIES-based molecular embeddings with an MLP classifier | [Notebook](notebooks/08_selformer_cv.ipynb) · [Script](scripts/08_selformer_cv.py) |
| 09 | ESM-2 conditioning | DRD2 protein embedding appended to ligand fingerprints | [Notebook](notebooks/09_esm2_drd2_conditioning_cv.ipynb) · [Script](scripts/09_esm2_drd2_conditioning_cv.py) |

### Benchmarking and Downstream Analysis

| ID | Workflow | Primary outcome | Files |
| --- | --- | --- | --- |
| 10 | Consolidated benchmark | Combine available model summaries with source-file provenance | [Notebook](notebooks/10_benchmark_results.ipynb) · [Script](scripts/10_benchmark_results.py) |
| 11 | Uncertainty and applicability domain | Ensemble disagreement and nearest-development fingerprint similarity | [Notebook](notebooks/11_uncertainty_applicability_domain.ipynb) · [Script](scripts/11_uncertainty_applicability_domain.py) |
| 12 | Active-learning simulation | Track labeled and unlabeled pool sizes during uncertainty sampling | [Notebook](notebooks/12_active_learning.ipynb) · [Script](scripts/12_active_learning.py) |
| 13 | Model-guided generation | Rank BRICS-generated molecular candidates by predicted activity | [Notebook](notebooks/13_model_guided_generation.ipynb) · [Script](scripts/13_model_guided_generation.py) |
| 14 | Interpretation and error analysis | Test prediction errors and random-forest fingerprint-bit importance | [Notebook](notebooks/14_interpretability_error_analysis.ipynb) · [Script](scripts/14_interpretability_error_analysis.py) |
| 15 | Technical report | Assemble the existing benchmark and interpretation notes into Markdown | [Notebook](notebooks/15_technical_report.ipynb) · [Script](scripts/15_technical_report.py) |

Start with **00**, then run selected models from **01–09**, including the optional **02B** and **06B** comparisons. Run **10** after producing model summaries and **15** after refreshing the benchmark. Workflows **11–14** perform their own analyses and can be run independently of the benchmark collector.

## Evaluation Protocol

The supervised benchmark workflows **01–09**, including **02B** and **06B**, share the following design:

1. Split the development dataset using `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`.
2. Fit each fold model on 90% of the development rows and validate on the remaining 10%. No separate validation CSV is required.
3. Evaluate each fold model on the same held-out test dataset. Test molecules do not participate in fitting or validation-based checkpoint selection.
4. Use a fixed active-class probability threshold of **0.50** for label-based metrics.
5. Export separate training, validation, and test metrics, with means and standard deviations across fold models. These benchmark workflows do not perform a final full-development-data refit.

Development folds are stratified by label, **not grouped by scaffold**. The test set is the supplied scaffold split, whose overlap is inspected in workflow 00. Test standard deviations describe variability between fitted models on one fixed dataset; they are not estimates from ten independent test sets. Use development validation results for model and hyperparameter selection.

Metrics include ROC AUC, active- and inactive-class average precision, Matthews correlation coefficient (MCC), balanced accuracy, class-specific precision and recall, Brier score, and confusion-matrix counts. Columns named `PR_AUC_active` and `PR_AUC_inactive` contain **average precision**, rather than trapezoidal precision–recall area. Higher Brier scores indicate greater probability error.

### Neural Training and Learning Curves

Workflows **03–09**, including **06B**, monitor validation loss with early-stopping patience **3** and minimum improvement **0.0001**. Each fold restores its lowest-validation-loss checkpoint. Epoch settings are maximum budgets, and loss plots average only the folds that reached each epoch.

A growing gap between decreasing training loss and increasing validation loss can indicate overfitting. Interpret loss curves together with validation metrics and class balance. The GNN workflow enables GCN and GIN by default; set `RUN_EXTENDED_VARIANTS=True` to include GraphSAGE and GAT.

## Pretrained Models and Embedding Caches

| Workflow | Checkpoint | Training scope |
| --- | --- | --- |
| 06 | `DeepChem/ChemBERTa-100M-MLM` | Frozen encoder; train the normalized head |
| 06B | `DeepChem/ChemBERTa-100M-MLM` | Fine-tune the final two encoder blocks by default, plus the head |
| 07 | `ibm/MoLFormer-XL-both-10pct` | Frozen encoder; train the MLP head |
| 08 | `HUBioDataLab/SELFormer` | Frozen encoder on SELFIES; train the MLP head |
| 09 | `facebook/esm2_t12_35M_UR50D` | Frozen DRD2 protein embedding; train the conditioned classifier |

MoLFormer pins revision `7b12d946c181a37f6012b9dc3b002275de070314` for the project's Transformers version and loads the serialized tokenizer directly. Keep the model and tokenizer revision settings together when reproducing this workflow.

Frozen molecular encoder workflows cache embeddings under `results/`. Cache reuse checks row count, but does not fully verify molecule identity, row order, or encoder configuration. Regenerate the affected cache files after changing those inputs. Workflow 07 uses IBM-specific cache filenames, and workflow 06B trains from tokenized molecules without using frozen embeddings.

## Results and Interpretation

All workflows write their artifacts to `results/`. Individual notebooks and script headers list the exact output filenames.

| Artifact | Contents |
| --- | --- |
| `*_cv_folds.csv` | Metrics for individual models, folds, and evaluation splits |
| `*summary.csv` | Aggregated model metrics, typically mean and standard deviation |
| `*_epoch_vs_loss.png` | Neural training and validation learning curves |
| `*_embeddings.npy` | Reusable frozen molecular features |
| `10_final_benchmark.csv` | Combined summaries with source filenames |
| `11_uncertainty_applicability_test.csv` | Test probabilities, model disagreement, and structural similarity |
| `12_active_learning_history.csv` | Pool sizes before each acquisition round |
| `13_generated_candidates.csv` | Candidate SMILES and activity scores, when generation succeeds |
| `14_test_error_analysis.csv` | Compound-level predictions and error categories |
| `14_rf_feature_importance.csv` | Ranked fingerprint-bit importance |
| `15_technical_report.md` | Report assembled from available benchmark results |

The benchmark collector reads every available `*summary.csv` containing a `model` column. Older experiment summaries may therefore appear alongside current outputs; inspect `source_file` before comparing models. Missing metric columns remain empty, and rerunning a workflow replaces its corresponding output files.

The downstream analyses have distinct interpretations:

- **ESM-2 conditioning:** Every ligand receives the same DRD2 vector. This is a single-target experiment, not evidence of generalization across protein targets.
- **Uncertainty:** Ensemble disagreement is descriptive and is not a calibrated prediction interval. Fingerprint similarity measures structural coverage.
- **Active learning:** The simulation reveals existing development labels and records acquisition progress. It does not measure held-out performance gains or compare against random acquisition.
- **Candidate generation:** Activity scores prioritize candidates; they do not establish experimental activity, novelty, or synthetic feasibility.
- **Feature importance:** A hashed fingerprint bit can represent multiple chemical environments. Its importance does not uniquely identify a substructure.

Random seeds support repeatability, but hardware, library versions, and cached artifacts can affect results. Preserve the environment and experiment settings alongside exported metrics when comparing runs.
