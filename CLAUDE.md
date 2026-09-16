# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Deep-learning pipeline for detecting and classifying breast lesions in DICOM mammography images (INC — Instituto Nacional de Cancerológico). Two stages, trained independently and chained at inference time:

1. **Detection**: a YOLO (Ultralytics) model finds lesion ROIs in a mammogram.
2. **Classification**: a hybrid model (frozen ResNet50 image backbone + frozen clinical-data MLP → trainable fusion MLP) predicts malignancy probability for each detected ROI.

There are actually four independently-trainable model variants under `src/models/`, plus a root-level combined model used only for inference:

- `classification_data_clinic/` — MLP trained on clinical/tabular data only.
- `classification_images/` — CNN (ResNet50/DenseNet121/InceptionV3 backbone + MLP head) trained on image patches only.
- `classification_final/` — the hybrid model: loads the *already-trained* image backbone and clinical MLP (both frozen) and trains only the fusion MLP on concatenated features.
- `detection/` — YOLO training via Ultralytics.

Each of these four is a self-contained mini-project with its own `main.py`, `options.py`, `training.py`/`train.py`, dataloader, and `run.sh` example command, and each has its own README with the authoritative details on data formats and CLI flags — read the relevant subproject README before changing training code there.

## Environment setup

```bash
conda env create -f environment.yml
conda activate inc-combined
```

Key deps: Python 3.10, PyTorch 2.9.1 (CUDA 12.8 wheels), `ultralytics` (YOLO), `pydicom`, `scikit-image`, `scikit-learn`, `opencv`, PySide6 (GUI libs, currently unused by any script in this repo). Training scripts log to **Weights & Biases**. Each `training.py`/`train.py` has a local `_wandb_credentials_cached()` helper (same idea as `FedMammoBench`'s `src/tracking.py`) that checks `~/.netrc` for a cached `wandb login` session and passes `mode="online"` if found, else `mode="offline"` — so runs never block on an interactive login prompt or fail outright when no account is configured; run `wandb login` first if you want runs actually uploaded. `wandb.init()` no longer hardcodes an `entity` — it logs to whatever account is cached locally.

There is no lint config, no test suite, and no packaging (`setup.py`/`pyproject.toml`) in this repo — nothing to run for "build" or "test".

## Running things

### End-to-end inference (detection → classification) on a single DICOM

```bash
python scripts/pipeline-inferencia.py \
    --dcm-path /path/to/image.dcm \
    --output-dir /path/to/output \
    --yolo-path-model /path/to/yolo_model.pt \
    --model-path /path/to/classification_model.pth
```

This is implemented by `PipelineInferencia` in `scripts/pipeline-inferencia.py`, which: reads + windows the DICOM, crops to the breast region, pads to square, resizes to 1024×1024, runs YOLO detection, crops each detected ROI from the full-res image, resizes/normalizes each patch to 224×224×3, and runs `models.model.MLP_Final_Model` (image patch + clinical features) to get a malignancy probability per lesion. **Clinical data is currently hardcoded** as a sample dict inside `ejecutar_pipeline()` (~line 107) — real patient data requires editing that dict directly.

`models/model.py` defines a second, parameter-free copy of the hybrid architecture (`MLP_Final_Model`, `ResNetModel`, `MLP`) with dimensions hardcoded to match a specific pretrained checkpoint. This is deliberately decoupled from `src/models/classification_final/models/get_model.py` (the configurable version used for training) — if you retrain the hybrid model with different hyperparameters, `models/model.py` must be updated to match or the pipeline's `torch.load(...).load_state_dict(...)` will fail on shape mismatch.

### Training a subproject model

Each subproject's imports are relative to its own directory (e.g. `classification_final/models/get_model.py` does `from models.image_models import get_image_model`, `main.py` does `from training import TrainModel`), so **you must `cd` into the subproject directory first** — running `python src/models/classification_final/main.py` from the repo root will fail with `ModuleNotFoundError`.

```bash
cd src/models/classification_data_clinic && python3 main.py --train ...   # clinical-only MLP
cd src/models/classification_images     && python3 main.py --train ...   # image-only CNN+MLP
cd src/models/classification_final      && python3 main.py --train ...   # hybrid fusion model
cd src/models/detection                 && python3 main.py ...          # YOLO detection
```

Each directory's `run.sh` has a filled-in example command (with placeholder or real historical paths) — copy and adapt it rather than guessing flags; `options.py` in each directory is the source of truth for available CLI arguments and defaults. `--train` triggers training + automatic test-set evaluation using the best checkpoint; omitting it runs test-only against an already-trained checkpoint.

Common per-experiment output layout (`<result_dir>/<exp_name>/`): `config.txt` (saved CLI args), `Saved_Models/Best_Model*.pth`, `Results/` (confusion matrix, ROC curve, `test_summary.csv`). YOLO training instead writes Ultralytics' own layout under `<path_project>/<experiment_name>/weights/{best,last}.pt`.

For `classification_final`, the clinical-MLP checkpoint passed via `--path_clinic_model` must have been trained with architecture flags (`--clinic_input_size`, `--clinic_hidden_layers`, `--clinic_activation`, `--clinic_dropout`) matching exactly what's passed to `classification_final`, or `load_state_dict` will fail — cross-check against the `classification_data_clinic` run that produced it.

### Data format expectations (shared across the three classification subprojects)

- `--csv_data_path` / `--data_clinic_path` points to a **directory**, not a file, and must contain exactly `train_*.csv`, `val_*.csv`, `test_*.csv` (exact filenames differ slightly per subproject — check its dataloader/README). Feature columns must be identical and same-order across the three splits, numeric only (no NaN, categoricals pre-encoded).
- `--images_dir` (image-based subprojects) contains `.npy` patch arrays; the CSV's ID/filename column must match `.npy` filenames exactly. Patches are expected pre-resized (no resize in the loader for `classification_final`/`classification_images`).

## Reference: metrics/evaluation/W&B in the sibling FedMammoBench project

`../FedMammoBench` (a separate federated + centralized mammography-classifier project, not part of this
repo) has a more structured metrics/tracking layer than this project's four subprojects — each of which
reimplements its own metrics/eval loop from scratch. Worth consulting as a reference pattern if asked to
make this repo's training/eval code less duplicated, since this repo's `_wandb_credentials_cached()`
helper (see "Environment setup" above) is already a direct copy of FedMammoBench's `src/tracking.py`:

- **Metrics**: `src/metrics.py` builds a `torchmetrics.MetricCollection` (accuracy, AUROC,
  sensitivity/recall, specificity, F1, F1-macro via a custom `BinaryMacroF1Score`, precision), computed
  once per split by `evaluate()`/`evaluate_checkpoint()` (`src/train/evaluation.py`) — always reloading
  the *best* checkpoint, never the model's end-of-training state. `f1_macro` (not `f1`) is used as the
  checkpoint-selection / early-stopping metric, because plain `f1` (positive class only) can sit at
  exactly `0.0` for several epochs while the backbone is frozen, and `EarlyStopping` never resets its
  patience counter on a flat `0.0`.
- **Extra diagnostic metrics**: `src/reporting.py:compute_confusion_matrix_metrics()` derives everything
  else computable from the confusion matrix (NPV, MCC, Cohen's kappa, likelihood ratios, balanced
  accuracy, etc., all prefixed `cm_`) once at the end, over full-test predictions from
  `predict_on_loader()` — not inside the per-epoch training loop.
- **Persistence** (`src/tracking.py:MetricsLogger`): one class writes to three destinations per run —
  `metrics.csv` (one row per epoch; the file/writer open lazily on the first `log()` call specifically so
  that re-evaluating an already-trained run via `evaluate.py` doesn't truncate its committed history),
  TensorBoard event files, and W&B (`log_summary()` for one-shot end-of-run values instead of `log()`, so
  they land in the run's summary rather than polluting the epoch time series). `run_dir/<split>/`
  (`val/`, `test/`) additionally gets `metrics.json`, `confusion_matrix_metrics.json`, `predictions.csv`
  (`y_true,y_pred,y_prob`), `confusion_matrix.png`, `roc_curve.png` — plus `metrics_by_database.json` and
  two more plots when per-database manifests are configured. `metrics.csv`/`metrics.json`/
  `predictions.csv`/`plots/*.png` are committed as the results record; checkpoints and TensorBoard event
  files are gitignored.
- **W&B config**: a single `train.wandb_project` config field (`None` disables W&B entirely — `wandb` is
  never even imported in that case). Auth is a shared team service account cached in `~/.netrc`;
  `MetricsLogger` checks `"api.wandb.ai" in ~/.netrc` and silently falls back to `mode="offline"` instead
  of blocking a run — exactly the "never block on missing credentials" behavior this repo's own
  `_wandb_credentials_cached()` reproduces.

`classification_images` and `classification_final` now replicate several of these pieces (added on top of
the existing training loop without changing its decisions — same loss, optimizer, early-stopping
criterion, and existing wandb keys/values as before): a new `reporting_extras.py` in each subproject dir
(`EpochLogger` for `metrics.csv` + TensorBoard per epoch, `compute_extra_metrics()` for AUC/F1-macro/NPV/
MCC/kappa/likelihood ratios, `save_metrics_json()`/`save_predictions_csv()`). `training.py`'s old
`test_model()` was factored into a shared `_evaluate_and_report(loader, stage, dir_name)` (test's own
output — `Results/test_summary.csv`, `confusion_matrix.png`, `roc_curve.png`, the wandb keys logged — is
byte-identical to before); a new `validate_model_full()` calls the same helper on `val_loader`, writing
to `Results_Val/`. Both also now upload confusion-matrix/ROC images and the predictions table to the W&B
run's summary (`wandb.summary.update()` / `wandb.Image()` / `wandb.Table()`), which the original code
didn't do. `classification_data_clinic` and `detection` were intentionally left untouched. This closed
several — not all — of the gaps in the comparison above; `classification_images`/`classification_final`
still don't have `metrics.csv` compared against a `run_dir`/experiment-tracking abstraction the way
FedMammoBench's `eval_pipeline.py` orchestrates it, and per-database breakdown was not added.
`environment.yml` gained `wandb`, `seaborn`, `torchmetrics`, `tensorboard` as pip deps — all four were
already imported by this code (directly or via the pre-existing `metrics.py`) but were missing from the
env file.

## Known gaps / inconsistencies to be aware of

- The root `README.md` documents a `src/data/` package (`dataset_generator.py`, `data_validator.py`, `dataset_splitter.py`) that **does not exist in this checkout**. `scripts/preprocesar-datos.py` imports from it and will fail (`ModuleNotFoundError`) until that package is added/restored.
- The README also references `test.py`, `run2.sh`, and `models/classifier_model.py` files under `classification_images`/`classification_final` that aren't present — treat the root README's file tree as aspirational/partially stale; each subproject's own README is more current for that subproject.
- `notebooks/` (`dividir-conjunto-datos.ipynb`, `procesar-datos-clinicos.ipynb`) hold exploratory data-prep work adjacent to the (missing) `src/data/` pipeline.
