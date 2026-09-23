#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

BASE_DIR="/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/13-PregradoJulian/Federal Learning/infraestructura federada"
FEDMAMMOBENCH_DIR="$BASE_DIR/FedMammoBench"
INC_DIR="$BASE_DIR/INC"

path_image_model_resnet="$FEDMAMMOBENCH_DIR/weights/ResNet50.pt"
path_data_clinic="$INC_DIR/models/pretrained_models/data_clinic.pth"
images_dir="/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench/preproccesed_julian"
csv_data_path="$INC_DIR/manifest/splits"
result_dir="$INC_DIR/results/final_classification"

PYTHON="${PYTHON:-/home/imagenesmedicas/miniconda3/envs/inc-combined/bin/python3}"

"$PYTHON" main.py --exp_name FinalClassification-hL-2048-1024-256-128_dropout-0.2_bs-64_lr-5e-4_BCE_without_balanced \
                --images_dir "$images_dir" \
                --csv_data_path "$csv_data_path" \
                --result_dir "$result_dir" \
                --tag_exp "Classification Images" \
                --activation_image_model "Gelu" \
                --image_model "ResNet" \
                --path_image_model "$path_image_model_resnet" \
                --path_clinic_model "$path_data_clinic" \
                --final_input_size 4096 \
                --final_hidden_layers 2048 1024 256 128 \
                --final_output_size 2 \
                --final_activation "Gelu" \
                --final_dropout 0.2 \
                --clinic_input_size 51 \
                --clinic_hidden_layers 2048 2048 1024 1024 512 256 128 64 32 \
                --clinic_activation "LeakyReLU" \
                --clinic_dropout 0.2 \
                --clinic_idx_hidden_layer 2 \
                --augmentation \
                --n_epochs 200 \
                --batch_size 64 \
                --lr 5e-4 \
                --patience_early 50 \
                --min_lr 1e-6 \
                --loss "BCE" \
                --train \
                "$@"


# python3 main.py --exp_name Inception-100f_hL-1024-512-256-128-64-32_dropout-0.2_bs-64_lr-1e-4_BCE-pos-0.80-neg-1.30 \
#                 --images_dir "/home/kevin-osorno-castillo/Documentos/parches_224x224/imagenes_npy" \
#                 --csv_data_path "/home/kevin-osorno-castillo/Documentos/parches_224x224/splits" \
#                 --result_dir $result_dir \
#                 --tag_exp "Classification Images" \
#                 --activation_image_model "LeakyReLU" \
#                 --image_model "Inception" \
#                 --path_image_model $path_image_model_inception \
#                 --num_freeze 100 \
#                 --hidden_layers 1024 512 256 128 64 32 \
#                 --output_size 2 \
#                 --activation "LeakyReLU" \
#                 --dropout 0.2 \
#                 --augmentation \
#                 --n_epochs 200 \
#                 --batch_size 64 \
#                 --lr 1e-4 \
#                 --patience_early 20 \
#                 --min_lr 1e-6 \
#                 --loss "BCE" \
#                 --class_balance \
#                 --pos_weight 0.80 \
#                 --neg_weight 1.30 \
#                 --train


# python3 main.py --exp_name DenseNet-120f_hL-1024-512-256-128-64-32_dropout-0.2_bs-64_lr-1e-4_BCE-pos-0.80-neg-1.30 \
#                 --images_dir "/home/kevin-osorno-castillo/Documentos/parches_224x224/imagenes_npy" \
#                 --csv_data_path "/home/kevin-osorno-castillo/Documentos/parches_224x224/splits" \
#                 --result_dir $result_dir \
#                 --tag_exp "Classification Images" \
#                 --activation_image_model "LeakyReLU" \
#                 --image_model "DenseNet" \
#                 --path_image_model $path_image_model_densenet \
#                 --num_freeze 120 \
#                 --hidden_layers 1024 512 256 128 64 32 \
#                 --output_size 2 \
#                 --activation "LeakyReLU" \
#                 --dropout 0.2 \
#                 --augmentation \
#                 --n_epochs 200 \
#                 --batch_size 64 \
#                 --lr 1e-4 \
#                 --patience_early 20 \
#                 --min_lr 1e-6 \
#                 --loss "BCE" \
#                 --class_balance \
#                 --pos_weight 0.80 \
#                 --neg_weight 1.30 \
#                 --train