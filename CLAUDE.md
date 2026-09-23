# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Deep-learning pipeline for detecting and classifying breast lesions in DICOM mammography images
(INC — Instituto Nacional de Cancerológico). Two stages, trained independently and chained at
inference time: **detection** (YOLO finds lesion ROIs) → **classification** (a hybrid frozen-backbone
+ frozen-clinic-MLP + trainable fusion-MLP model scores malignancy per ROI).

**This file is a router, not the full context.** Load the sheet that matches the task before working;
don't load them all.

---

## Where to go

### Code branch — `.claude/context/code/`

| if you're going to… | read |
|---|---|
| train the YOLO detector | [DETECTION.md](.claude/context/code/DETECTION.md) |
| train the clinical-only MLP | [CLASSIFICATION_CLINIC.md](.claude/context/code/CLASSIFICATION_CLINIC.md) |
| train the image-only CNN+MLP | [CLASSIFICATION_IMAGES.md](.claude/context/code/CLASSIFICATION_IMAGES.md) |
| train the hybrid fusion model | [CLASSIFICATION_FINAL.md](.claude/context/code/CLASSIFICATION_FINAL.md) |
| touch `scripts/pipeline-inferencia.py` or root `models/model.py` (end-to-end inference) | [INFERENCE_PIPELINE.md](.claude/context/code/INFERENCE_PIPELINE.md) |
| touch `manifest/`, splits, or `scripts/preprocesar-datos.py` | [DATA.md](.claude/context/code/DATA.md) |

Each subproject under `src/models/*/` also has its own README with the authoritative CLI-flag list —
read it alongside the matching sheet above, not instead of it: the sheets carry the *why* and the
cross-subproject gotchas (e.g. two different subprojects reusing the same CSV filenames for
incompatible formats), the READMEs carry the full flag reference.

Each of the four subprojects (`classification_data_clinic/`, `classification_images/`,
`classification_final/`, `detection/`) is self-contained: its own `main.py`, `options.py`,
`training.py`/`train.py`, dataloader, and `run.sh` example command. Imports inside each are relative
to its own directory, so **`cd` into the subproject directory before running its script** — invoking
`python src/models/classification_final/main.py` from the repo root fails with `ModuleNotFoundError`.

---

## Environment setup

```bash
conda env create -f environment.yml
conda activate inc-combined
```

Python 3.10, PyTorch 2.9.1 (CUDA 12.8 wheels), `ultralytics` (YOLO), `pydicom`, `scikit-image`,
`scikit-learn`, `opencv`, PySide6 (GUI libs, currently unused by any script here). No lint config, no
test suite, no packaging (`setup.py`/`pyproject.toml`) — nothing to run for "build" or "test".

Training scripts log to **Weights & Biases**. Each `training.py`/`train.py` has a local
`_wandb_credentials_cached()` helper (a direct copy of `FedMammoBench`'s `src/tracking.py` idea) that
checks `~/.netrc` for a cached `wandb login` session and passes `mode="online"` if found, else
`mode="offline"` — runs never block on an interactive login prompt or fail outright with no account
configured. Run `wandb login` first if you want runs actually uploaded. `wandb.init()` doesn't
hardcode an `entity` — it logs to whatever account is cached locally.

---

## Reference: metrics/evaluation/W&B pattern in the sibling FedMammoBench project

`../FedMammoBench` (a separate federated + centralized mammography-classifier project, not part of
this repo) has a more structured metrics/tracking layer than this repo's four subprojects — each of
which reimplements its own metrics/eval loop from scratch. Worth consulting if asked to make this
repo's training/eval code less duplicated:

- **Metrics**: `src/metrics.py` builds a `torchmetrics.MetricCollection` (accuracy, AUROC,
  sensitivity/recall, specificity, F1, F1-macro via a custom `BinaryMacroF1Score`, precision), computed
  once per split by `evaluate()`/`evaluate_checkpoint()` — always reloading the *best* checkpoint,
  never the model's end-of-training state. `f1_macro` (not `f1`) is used as the
  checkpoint-selection/early-stopping metric, because plain `f1` can sit at exactly `0.0` for several
  epochs while the backbone is frozen, and `EarlyStopping` never resets patience on a flat `0.0`.
- **Extra diagnostic metrics**: `src/reporting.py:compute_confusion_matrix_metrics()` derives NPV,
  MCC, Cohen's kappa, likelihood ratios, balanced accuracy, etc. (all prefixed `cm_`) once at the end
  over full-test predictions — not inside the per-epoch loop.
- **Persistence** (`src/tracking.py:MetricsLogger`): writes `metrics.csv` (file/writer opened lazily
  on first `log()`, so re-evaluating an already-trained run doesn't truncate committed history),
  TensorBoard event files, and W&B (`log_summary()` for one-shot end-of-run values, kept out of the
  epoch time series). `run_dir/<split>/` also gets `metrics.json`, `confusion_matrix_metrics.json`,
  `predictions.csv`, plots — committed as the results record; checkpoints and TensorBoard files are
  gitignored.
- **W&B config**: a single `train.wandb_project` field (`None` disables it entirely). Auth is a
  shared team service-account cached in `~/.netrc`, exactly the "never block on missing credentials"
  behavior this repo's `_wandb_credentials_cached()` reproduces.

`classification_images` and `classification_final` now replicate several of these pieces on top of
their original training loops (same loss/optimizer/early-stopping decisions, same existing wandb
keys/values as before): a `reporting_extras.py` in each subproject dir (`EpochLogger` for
`metrics.csv`+TensorBoard, `compute_extra_metrics()`, `save_metrics_json()`/`save_predictions_csv()`).
`training.py`'s old `test_model()` was factored into `_evaluate_and_report(loader, stage, dir_name)`
(test output is byte-identical to before); a new `validate_model_full()` calls the same helper on
`val_loader`, writing `Results_Val/`. Both now also upload confusion-matrix/ROC images and the
predictions table to the W&B run summary, which the original code didn't do.
`classification_data_clinic` and `detection` were intentionally left untouched — this closed several,
not all, of the gaps above; per-database breakdown was not added, and neither subproject has a
`run_dir`/experiment-tracking abstraction the way FedMammoBench's `eval_pipeline.py` orchestrates it.
`environment.yml` gained `wandb`, `seaborn`, `torchmetrics`, `tensorboard` as pip deps — all four were
already imported by this code but were missing from the env file.

---

## Known gaps / inconsistencies

- The root `README.md` documents a `src/data/` package that **does not exist in this checkout** —
  see [DATA.md](.claude/context/code/DATA.md).
- The README also references `test.py`, `run2.sh`, and `models/classifier_model.py` files under
  `classification_images`/`classification_final` that aren't present — treat the root README's file
  tree as aspirational/partially stale; each subproject's own README is more current for that
  subproject.
