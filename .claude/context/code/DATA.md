# DATA.md — manifests, splits, the missing `src/data/`

## `scripts/preprocesar-datos.py` is dead code

It imports `from src.data.dataset_generator import DatasetGenerator`,
`from src.data.data_validator import DataValidator`, `from src.data.dataset_splitter import
DatasetSplitter`. **`src/data/` does not exist in this checkout** — this script raises
`ModuleNotFoundError` until that package is restored or added. The root `README.md`'s file tree
describing it is aspirational, not current.

## What's actually present instead: `manifest/generar_splits.py`

Pure-stdlib (`csv` module, no pandas). Reads `manifest/fedmammobench_norm_neg1_1.csv` — columns
`ID_image, source_dataset, laterality, view, preprocessed_image_path, classification, density,
BIRADS, abnormality, molecular_subtype, raw_image_path, mask_path, ROI_path, x, y, radius,
subject_age, source_subjectID, original_source_path, patient_id, split`.

For each `split` value it writes `manifest/splits/{train,val,test}_clinical_data.csv`, prepending two
**positional** columns 0–1:

- col 0: `image_path` — a copy of `preprocessed_image_path`.
- col 1: `label` — 0 (Benign) / 1 (Malignant), derived from `classification`.

These land at columns 0–1 by *position*, because
[`classification_images`](CLASSIFICATION_IMAGES.md)'s `ImageDataset.__getitem__`
(`dataloaders/dataloader_images.py`) reads them via `.iloc[sample, 0]` / `.iloc[sample, 1]` — **not by
column name**. This is the actual generator behind the `train/val/test_clinical_data.csv` format
consumed by `classification_images`, whose filenames collide (but not the contents) with
`classification_final`'s `train/val/test_clinical_data.csv`, which instead carries `ID`, `Etiqueta`,
and real clinical feature columns. Don't conflate the two — check which subproject a given
`*_clinical_data.csv` belongs to before editing it.

## Notebooks — exploratory, not wired into any pipeline

- `notebooks/dividir-conjunto-datos.ipynb` — dataset splitting exploration.
- `notebooks/procesar-datos-clinicos.ipynb` — clinical-data imputation/cleaning ("Imputación y
  limpieza de datos").

Neither is imported by or invoked from any script. They sit adjacent to, not as a replacement for, the
missing `src/data/` pipeline.

## Cross-reference

The sibling `../FedMammoBench` project (a separate repo, not part of this one) uses the same
`fedmammobench_norm_*.csv` manifest family and has a more structured `Dataset`/split/registry layer —
worth consulting if asked to build out `src/data/` properly instead of patching around its absence.
