# CLASSIFICATION_CLINIC.md — `src/models/classification_data_clinic/`

Clinical/tabular-only MLP. Files: `main.py`, `options.py`, `training.py`, `losses.py` (weighted
CrossEntropy for BCE), `metrics.py`, `early_stopping.py` (saves `Best_Model.pth`),
`dataloaders/data_loader_data.py`, `models/get_model.py` + `mlp_models.py`.

## Running it

```bash
cd src/models/classification_data_clinic
python3 main.py --train --data_clinic_path <dir> \
    --input_size_clinic_model 106 --hidden_layers_clinic_model 2048 512 128 \
    --output_size_clinic_model <n> --activation_clinic_model <fn> --dropout_clinic_model <p> \
    --loss bce --neg_weight <w> --pos_weight <w> --patience_early <n> --min_lr <lr>
```

`--train` triggers training + automatic test-set evaluation against the best checkpoint; omit it to
run test-only against an already-trained checkpoint.

## Data format

`--data_clinic_path` points to a **directory**, not a file, containing exactly
`train_clinical_data_processed.csv`, `val_clinical_data_processed.csv`, `test_clinical_data_processed.csv`.
Columns: `ID` (dropped), `Diagnostico` (0/1 label), then numeric feature columns — identical set and
order across the three splits, no NaN, categoricals pre-encoded.

## Output

`<result_dir>/<exp_name>/{config.txt, Saved_Models/Best_Model.pth, Results/{confusion_matrix.png,
roc_curve.png, test_summary.csv}}`.

## Downstream dependency

This checkpoint is loaded **frozen** by `classification_final` (see
[CLASSIFICATION_FINAL.md](CLASSIFICATION_FINAL.md)) via `--path_clinic_model` — the
`--clinic_input_size`/`--clinic_hidden_layers`/`--clinic_activation`/`--clinic_dropout` flags passed
there must exactly match the architecture flags used here, or `load_state_dict` fails downstream. When
you retrain this subproject with different hyperparameters, note the new architecture flags for
whoever retrains `classification_final` against this checkpoint.
