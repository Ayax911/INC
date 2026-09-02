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

Key deps: Python 3.10, PyTorch 2.9.1 (CUDA 12.8 wheels), `ultralytics` (YOLO), `pydicom`, `scikit-image`, `scikit-learn`, `opencv`, PySide6 (GUI libs, currently unused by any script in this repo). Training scripts log to **Weights & Biases** — a wandb account/API key must be configured (`wandb login`) before running any `training.py`/`train.py`, or logging calls will fail/prompt.

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

## Known gaps / inconsistencies to be aware of

- The root `README.md` documents a `src/data/` package (`dataset_generator.py`, `data_validator.py`, `dataset_splitter.py`) that **does not exist in this checkout**. `scripts/preprocesar-datos.py` imports from it and will fail (`ModuleNotFoundError`) until that package is added/restored.
- The README also references `test.py`, `run2.sh`, and `models/classifier_model.py` files under `classification_images`/`classification_final` that aren't present — treat the root README's file tree as aspirational/partially stale; each subproject's own README is more current for that subproject.
- `notebooks/` (`dividir-conjunto-datos.ipynb`, `procesar-datos-clinicos.ipynb`) hold exploratory data-prep work adjacent to the (missing) `src/data/` pipeline.
