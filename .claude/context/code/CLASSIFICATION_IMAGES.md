# CLASSIFICATION_IMAGES.md — `src/models/classification_images/`

Image-only CNN+MLP: `ResNetModel` (ResNet50 minus final FC, ResNet18/DenseNet121/InceptionV3/
`CustomCNNModel` also supported by `models/get_model.py`) → `MLP` head, wired as
`nn.Sequential(image_model, classifier)`.

`CustomCNNModel` (`models/image_models.py`) is a from-scratch CNN backbone — 4 blocks of
`Conv2d(3x3, padding="same") -> BatchNorm2d -> ReLU` x2 (channels 32/64/128/256), `MaxPool2d(2)`
after blocks 1-3 (block 4 does not downsample), `AdaptiveAvgPool2d(1)` → 256 features. It has no
pretrained weights: `--image_model CustomCNN` requires `--from_scratch` (or any other flag that
makes `get_model.py` pass `weigths_file=None`) — passing `--path_image_model` with it raises a
`ValueError` instead of a confusing `load_state_dict` shape mismatch against the default
`ResNet50.pt`, and `--pretrained` also raises. It's a parallel, non-shared reimplementation of
`FedMammoBench/src/models/custom_cnn.py::CustomCNNBackbone` (same design, same parameter count —
verified: 1,174,176 trainable params in both repos), paired for cross-repo comparison the same way
`ResNet18Model` is paired with FMB's `resnet18_scratch`.

## Running it

```bash
cd src/models/classification_images
python3 main.py --train --images_dir <dir> --csv_data_path <dir> --path_image_model <weights.pth> \
    --augmentation ...
```

`options.py` is authoritative for flags; copy `run.sh`.

## Data format

- `--images_dir` holds `.npy` patch arrays (`H×W×3` float32 recommended). `Dataset.__getitem__`
  (`dataloaders/dataloader_images.py`) does `np.load` + `ToTensor()` (HWC→CHW). **No resize in the
  loader** — patches must arrive pre-resized.
- `--csv_data_path` dir needs exactly `train_clinical_data.csv` / `val_clinical_data.csv` /
  `test_clinical_data.csv` — **same filenames as the clinical subproject's format, but a different
  shape**: here col 0 = `.npy` filename string, col 1 = label int, **no header row**, no clinical
  feature columns at all. `ImageDataset.__getitem__` reads these **positionally** via
  `.iloc[sample, 0]` / `.iloc[sample, 1]`, not by column name — see
  [DATA.md](DATA.md) for the generator (`manifest/generar_splits.py`) that produces this exact format.

## Backbone weights

`--path_image_model` loads a pretrained backbone via `image_models.py`:
`base_model.load_state_dict(torch.load(weights_file, map_location=device))`.

## Final classifier head

`models/mlp_models.py::MLP` also has `--input_dropout` (default `0.0`, port of FedMammoBench's
`ConfigurableMLPHead.input_dropout`): a `Dropout` applied right after the input, before any hidden
layer. Exists because plain `--dropout` is only inserted *inside* the hidden-layer loop, so with
`--hidden_layers` empty (a single `Linear(features, output_size)`, no hidden layer) there was no way
to request dropout at all — needed for `CustomCNN`'s "GAP → Dropout(0.4) → Dense(1)" design. Default
preserves every existing config's behavior unchanged.

## Augmentation

`--augmentation` adds (train split only) `RandomHorizontalFlip(0.5)`, `RandomRotation(15)`,
`RandomVerticalFlip(0.2)`, random `GaussianBlur`; val/test always get bare `ToTensor()`.

## Output

`<result_dir>/<exp_name>/{config.txt, batch_images.png, Saved_Models/{Best_Model.pth,
Best_Model_image.pth, Best_Model_classifier.pth, Model_NNN.pth}, Results/{test_summary.csv,
confusion_matrix.png, roc_curve.png}}`.

## Downstream dependency

The trained backbone (`Best_Model_image.pth` or equivalent) is loaded **frozen** by
`classification_final` via `--path_image_model` — see [CLASSIFICATION_FINAL.md](CLASSIFICATION_FINAL.md).

## Tracking layer

Per [`../../../CLAUDE.md`](../../../CLAUDE.md) §"Reference: metrics/evaluation/W&B", this subproject
has a `reporting_extras.py` (added on top of the original training loop, same loss/optimizer/
early-stopping decisions) providing `EpochLogger`, `compute_extra_metrics()`
(AUC/F1-macro/NPV/MCC/kappa/likelihood ratios), `save_metrics_json()`/`save_predictions_csv()`, plus a
`validate_model_full()` writing `Results_Val/` and W&B image/table uploads on top of the original
`test_model()` (now `_evaluate_and_report()`).
