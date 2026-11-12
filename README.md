# Batik-CSPLiteNeXt: code, splits and per-fold results

Supporting material for the manuscript "Batik-CSPLiteNeXt" submitted to *Computers, Materials & Continua* (manuscript 94861).
It contains everything needed to reproduce the experiments and the statistical tests reported in the paper.

## Contents

| Folder | What it holds |
|---|---|
| `code/` | `cspconvnext.py` (model: partitioned stages, stepped stem, BN blocks, ESE, partition ratio), `batik_train.py` (training, five-fold cross-validation, EMA weights, metrics, efficiency measurement), `batik_dataset_tool.py` with `class_map.json` (curation: perceptual-hash deduplication and frozen dataset version) and `list_removed_duplicates.py` (regenerates the list of removed images). |
| `notebooks/` | The Kaggle notebooks used for every experiment (augmentation regimes, repeated CV with fold seed 2026, partition-ratio sweep, component ablation, partial pretrained initialisation, MobileViT-S, ImageWoof, misclassified-image figure). |
| `splits/` | `folds_seed42.csv` (main protocol) and `folds_seed2026.csv` (repeated CV): for each of the 766 curated images, its index, relative path, class index and test fold. |
| `curation/` | `manifest.csv` (source path, class and original split of every retained image) and `removed_duplicates.csv` (each of the 34 removed images, its duplicate cluster, the retained image and their Hamming distance). |
| `results/` | Per-fold scores (`results_*.csv`) and per-model summaries (`ringkasan_*.csv`, "summary"). |
| `predictions/` | Per-fold test predictions (`*_fold{k}.npz`: `test_idx`, `files`, `y_true`, `y_prob`) used for the paired tests, McNemar tests, bootstrap intervals and ROC curves. ImageWoof predictions are named `woof_*_seed{s}.npz`. |
| `figures/` | Misclassified-image figure and the confusion matrix of the r = 0.16 model. |

## Data

- Batik Nusantara (original, 800 images, 20 classes): public on Kaggle (reference [21] of the paper).
- Curated version used in the paper (766 images after perceptual-hash deduplication, Hamming distance 5): Kaggle dataset `hertiyani/batik-clean-v1`.
- DTD and ImageWoof: available from their official sources.

## Reproducing

1. Attach the curated dataset to a Kaggle notebook (GPU enabled) or place it locally.
2. Put `code/cspconvnext.py` and `code/batik_train.py` in the working directory (the notebooks also write them with `%%writefile`).
3. Run the notebook of the experiment of interest. Fold assignments are produced by `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)` over the pooled file list; `splits/` records the result so it can be checked.

Requirements: Python 3, PyTorch, torchvision, timm, scikit-learn, thop, imagehash, Pillow.

## License

Code: MIT License (see `LICENSE`). Splits, results, predictions and figures: CC BY 4.0. The images themselves remain under the license of the original Batik Nusantara dataset and are not redistributed here.
