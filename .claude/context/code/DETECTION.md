# DETECTION.md — `src/models/detection/`

YOLO (Ultralytics) lesion-ROI detector. Standalone subproject: `main.py`, own `run.sh`, own `Readme.md`.

## Running it

```bash
cd src/models/detection
python3 main.py --path_data data_1024.yaml --img_size 1024 --batch_size 16 --epochs 300 \
    --device 0 --pretrained --path_project <dir> --experiment_name <name> --optimizer auto \
    --seed 42 --learning_rate 1e-3 --freeze_layers 8 --box_weight 4.0 --classification_weight 0.5 \
    --dropout_rate 0.0 --model_path yolo11_m_digitaleye.pt --save_period 20
```

`options.py` is the source of truth for the full flag list; copy `run.sh` rather than guessing.

## Data format

`<root>/images/{train,val,test}/` + `<root>/labels/{train,val,test}/`, one `.txt` per image with lines
`class_id x_center y_center width height` normalized to `[0,1]` (standard YOLO format). The dataset
YAML (`data_640.yaml`/`data_1024.yaml`) declares `path`/`train`/`val`/`test`/`nc`/`names`.

## Output layout — differs from the three classification subprojects

Ultralytics writes its own tree, **not** the `Saved_Models/`+`Results/` convention the other three
subprojects share:

```
<path_project>/<experiment_name>/weights/{best,last}.pt
<path_project>/<experiment_name>/results.csv
<path_project>/<experiment_name>/confusion_matrix.png
<path_project>/<experiment_name>/PR_curve.png
```

## Gotcha

`prediction.py` has hardcoded paths per the README's own words ("típicamente tiene rutas hardcodeadas:
debes editar") — edit them before running standalone prediction; it's not driven by `options.py` flags
the way `main.py` is.

`scripts/pipeline-inferencia.py` (see [INFERENCE_PIPELINE.md](INFERENCE_PIPELINE.md)) is the real
end-to-end consumer of a trained detection checkpoint — it loads `best.pt` via `--yolo-path-model` and
calls `.predict(conf=0.25, iou=0.5, imgsz=1024, device="cpu")` directly, bypassing `prediction.py`.
