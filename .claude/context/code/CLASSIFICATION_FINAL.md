# CLASSIFICATION_FINAL.md — `src/models/classification_final/`

The hybrid fusion model. Both the image backbone and the clinical MLP are loaded **frozen and
pretrained**; only the final fusion `classifier` MLP trains.

## Forward pass

Image backbone → image features. Clinical MLP → `get_hidden_layer_features(model, x, layer_idx)`
pulls an **intermediate-layer activation** (index via `--clinic_idx_hidden_layer`, default matches
`models/model.py`'s hardcoded `layer_idx=2` — see [INFERENCE_PIPELINE.md](INFERENCE_PIPELINE.md)).
Image features + clinic hidden features are concatenated, then passed through the final MLP →
2 logits.

## Running it

```bash
cd src/models/classification_final
python3 main.py --train --csv_data_path <dir> \
    --path_image_model <backbone.pth> --path_clinic_model <clinic.pth> \
    --clinic_input_size <n> --clinic_hidden_layers <...> --clinic_activation <fn> \
    --clinic_dropout <p> --clinic_idx_hidden_layer <idx> \
    --final_input_size <concat_dim> --final_hidden_layers <...> --final_activation <fn> \
    --final_dropout <p> --loss BCE --class_balance --pos_weight <w> --neg_weight <w>
```

`--loss` accepts `"BCE"` or `"Focal"` (with `--gamma`).

## Critical invariant

`--path_clinic_model` must have been trained with architecture flags
(`--clinic_input_size`, `--clinic_hidden_layers`, `--clinic_activation`, `--clinic_dropout`) matching
**exactly** what's passed here, or `load_state_dict` fails on shape mismatch (stated explicitly in the
subproject README). Cross-check against the `classification_data_clinic` run that produced the
checkpoint — see [CLASSIFICATION_CLINIC.md](CLASSIFICATION_CLINIC.md). Same applies to
`--path_image_model` against [CLASSIFICATION_IMAGES.md](CLASSIFICATION_IMAGES.md).

`--final_input_size` is the concat dimension (image feature dim + clinic hidden-layer dim at
`clinic_idx_hidden_layer`) — get this wrong and the first fusion-MLP layer's shape won't match.

## Data format

`--csv_data_path` dir needs `train_clinical_data.csv` / `val_clinical_data.csv` /
`test_clinical_data.csv` with columns `ID`, `Etiqueta`, plus clinical feature columns — **this
subproject's version of these filenames DOES carry clinical features**, unlike `classification_images`'
same-named files (see [DATA.md](DATA.md)). Don't assume the format from the filename alone.

## Early stopping — differs from the other two subprojects

Monitors `Val_F1-Score`, not the metric used by `classification_data_clinic`/`classification_images`.

## Output

Adds `Best_Model_Final.pth` (final fusion MLP weights only) alongside the full `Best_Model.pth`.

## Relationship to the standalone inference pipeline

`scripts/pipeline-inferencia.py` + root `models/model.py` reimplement this architecture from scratch,
parameter-free, for inference-only use — they are **not** imported from here and can silently drift
out of sync. See [INFERENCE_PIPELINE.md](INFERENCE_PIPELINE.md) before retraining with different
hyperparameters if the inference pipeline needs to keep working.
