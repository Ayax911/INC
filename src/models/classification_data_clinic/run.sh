#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

BASE_DIR="/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/13-PregradoJulian/Federal Learning/infraestructura federada"
INC_DIR="$BASE_DIR/INC"

data_clinic_path="$INC_DIR/manifest/splits"
result_dir="$INC_DIR/results/data_clinic"

PYTHON="${PYTHON:-/home/imagenesmedicas/miniconda3/envs/inc-combined/bin/python3}"

"$PYTHON" main.py --exp_name "Classification_Clinical_Data" \
                 --data_clinic_path "$data_clinic_path" \
                 --result_dir "$result_dir" \
                 --tag_exp "Clinical Data" \
                 --input_size_clinic_model 106 \
                 --hidden_layers_clinic_model 2048 \
                 --output_size_clinic_model 1 \
                 --activation_clinic_model "LeakyReLU" \
                 --dropout_clinic_model 0.5 \
                 --n_epochs 100 \
                 --batch_size 32 \
                 --lr 1e-4 \
                 --patience_early 20 \
                 --class_balance \
                 --neg_weight 2.0 \
                 --pos_weight 1.0 \
                 --train