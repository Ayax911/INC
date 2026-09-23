# INFERENCE_PIPELINE.md — `scripts/pipeline-inferencia.py` + root `models/model.py`

The only end-to-end (detection → classification) entrypoint. Root-level `models/model.py` (not under
`src/`) is a second, independent, parameter-free reimplementation of the `classification_final`
architecture, built specifically to be loaded by this script.

## `PipelineInferencia.ejecutar_pipeline()` — order of operations

1. `_leer_imagen_dicom` — `pydicom` read + `apply_windowing`, MONOCHROME1 inversion.
2. `_recortar_mama` — binarize (threshold 10) → `binary_fill_holes` → `remove_small_objects(min_size=1000)`
   → `binary_closing(disk(15))` → bbox + 10px margin crop.
3. `_agregar_padding_cuadrado` — zero-pad to square.
4. `_obtener_imagen_1024` — resize to 1024×1024, `rescale_intensity` to uint8 `[0,255]`, save as JPG.
5. YOLO `.predict(conf=0.25, iou=0.5, imgsz=1024, device="cpu")` on the 1024 JPG.
6. Crop each detected box from the **full-res square image** (not the 1024 JPG) via `yolo_to_corners`
   + a scale factor back from the 1024 detection space.
7. `_procesar_parche` per detected ROI — resize 224×224, `rescale_intensity(in_range=(0,4095),
   out_range=(-1,1))`, stack ×3 to fake RGB.
8. Run `MLP_Final_Model` (image patch + clinical features) → malignancy probability per lesion.

`self.device = torch.device("cpu")` is hardcoded at line 29 (GPU line commented out).

## Hardcoded clinical data — no CLI/form input path

Lines 107–130 inside `ejecutar_pipeline()`: a literal dict `datos_clinicos`, followed by
`_procesar_datos_clinicos` (hardcoded mean/std z-score norm for 2 continuous columns) and fixed
one-hot category lists at lines 157–212. **Using real patient data means hand-editing this dict** —
there is no other input path.

## `models/model.py` — how it relates to, and diverges from, `classification_final`

`MLP_Final_Model.__init__` hardcodes, with zero CLI args:

- `ResNetModel()` for the image branch.
- Clinic branch: `MLP(input_size=51, hidden_layers=[2048,2048,1024,1024,512,256,128,64,32],
  dropout=0.2)`.
- Fusion branch: `MLP(input_size=4096, hidden_layers=[2048,1024,256,128], dropout=0.2)`.
- `get_hidden_layer_features` hardcodes `layer_idx=2` (matches the README's
  `--clinic_idx_hidden_layer 2` default in [CLASSIFICATION_FINAL.md](CLASSIFICATION_FINAL.md)).

This is deliberately decoupled from `src/models/classification_final/models/get_model.py` (the
configurable version driven by `--clinic_*`/`--final_*` flags, used for training).

**Consequence:** if `classification_final` is retrained with different hyperparameters (different
hidden-layer sizes, different `clinic_idx_hidden_layer`), `models/model.py`'s hardcoded shapes go
stale, and `PipelineInferencia.__init__` (`self.model.load_state_dict(torch.load(model_path))`,
line 38) raises a shape-mismatch error. You must hand-edit `models/model.py` to match the new
architecture before this script will load that checkpoint.

## Command

```bash
python scripts/pipeline-inferencia.py \
    --dcm-path /path/to/image.dcm --output-dir /path/to/output \
    --yolo-path-model /path/to/yolo_model.pt --model-path /path/to/classification_model.pth
```
