#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

BASE_DIR="/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/13-PregradoJulian/Federal Learning/infraestructura federada"
FEDMAMMOBENCH_DIR="$BASE_DIR/FedMammoBench"
INC_DIR="$BASE_DIR/INC"

path_image_model_resnet="$FEDMAMMOBENCH_DIR/weights/ResNet50.pt"
images_dir="/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench/preproccesed_julian"
csv_data_path="$INC_DIR/manifest/splits"
result_dir="$INC_DIR/results"

PYTHON="${PYTHON:-/home/imagenesmedicas/miniconda3/envs/inc-combined/bin/python3}"

"$PYTHON" main.py --exp_name "Classification Images" \
                --images_dir "$images_dir" \
                --csv_data_path "$csv_data_path" \
                --result_dir "$result_dir" \
                --tag_exp "Classification Images" \
                --activation_image_model "Gelu" \
                --image_model "ResNet" \
                --path_image_model "$path_image_model_resnet" \
                --num_freeze 80 \
                --hidden_layers 2048 1024 256 \
                --output_size 2 \
                --activation "Gelu" \
                --dropout 0.5 \
                --augmentation \
                --n_epochs 200 \
                --batch_size 64 \
                --lr 5e-4 \
                --patience_early 50 \
                --min_lr 1e-6 \
                --loss "BCE" \
                --train \
                "$@"

