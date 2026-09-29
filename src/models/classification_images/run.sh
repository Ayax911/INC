#!/usr/bin/env bash
# ==============================================================================
# exp64_custom_cnn_inc -- backbone CNN custom (INC). Par del lado INC de
# FedMammoBench/configs/exp64_custom_cnn.yaml.
#
# exp61_resnet50_scratch_inc, exp62_resnet18_scratch_inc y
# exp63_resnet18_imagenet_inc (ResNet50/18 desde cero, ResNet18 ImageNet) se
# quitaron de este script: ya tienen resultados reales publicados acá, así
# que no hace falta re-correrlos. Recuperables con
# `git log --all --oneline -- src/models/classification_images/run.sh` si
# hiciera falta reproducirlos.
# ==============================================================================
set -uo pipefail
cd "$(dirname "$0")"

BASE="/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/13-PregradoJulian/Federal Learning/infraestructura federada"
INC_DIR="$BASE/INC"
REPRO="$BASE/inc_repro_data"
RESULTS_DIR="$INC_DIR/results"
PYTHON="/home/imagenesmedicas/miniconda3/envs/inc-combined/bin/python3"

IMGROOT="/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench/preproccesed_julian"

mkdir -p "$RESULTS_DIR"
status=0

# ==============================================================================
# exp64 -- backbone CNN custom, desde cero (todo entrenable)
#
#    --output_size 2 (no 1): "--loss BCE" acá es en realidad nn.CrossEntropyLoss
#    ponderada de 2 clases (ver losses.py), no BCEWithLogitsLoss de 1 logit --
#    no puede reproducir el "Dense(1)+sigmoid" literal del diseño original ni
#    con --output_size 1. FedMammoBench/configs/exp64_custom_cnn.yaml sí usa 1
#    logit real (loss: bce); esta es la comparación más cercana que el código
#    de pérdida de este repo permite.
#
#    TODOS los flags de options.py que aplican están explícitos, aunque su
#    valor coincida con el default. Los que quedan afuera y por qué:
#      --pretrained: store_true, default False -- CustomCNN además lo rechaza
#        con ValueError si se pasa (ver CustomCNNModel en image_models.py).
#      --path_image_model: --from_scratch lo hace ignorarse por completo en
#        get_model.py -- pasar su default (una ruta de otra máquina) sería
#        más confuso que omitirlo.
#      --normalize_mean/--normalize_std/--backbone_lr/--dataloader_seed:
#        quedan en None -- tipados float/int, no existe forma de pasarles
#        "None" explícito por CLI; la ausencia ES el valor explícito.
#      --gamma: solo lo lee focal_loss.py -- irrelevante con --loss BCE.
# ==============================================================================
echo "=== running exp64_custom_cnn_inc ==="
if ! "$PYTHON" main.py \
    --exp_name       "exp64_custom_cnn_inc" \
    --images_dir     "$IMGROOT" \
    --csv_data_path  "$REPRO/csvs_norm_neg1_1" \
    --result_dir     "$RESULTS_DIR" \
    --wandb_project  "fedmammobench2.0" \
    --wandb_group    "custom_cnn" \
    --tag_exp        "inc" "custom_cnn" \
    --seed           42 \
    --image_model    "CustomCNN" \
    --activation_image_model "ReLU" \
    --from_scratch \
    --num_freeze     0 \
    --hidden_layers \
    --output_size    2 \
    --activation     "LeakyReLU" \
    --dropout        0.0 \
    --input_dropout  0.4 \
    --channels       3 \
    --augmentation \
    --img_size       224,224 \
    --init_epoch     0 \
    --n_epochs       100 \
    --batch_size     16 \
    --lr             1e-3 \
    --b1             0.9 \
    --b2             0.999 \
    --weight_decay   1e-4 \
    --label_smoothing 0.0 \
    --patience_early 50 \
    --min_lr         1e-6 \
    --loss           "BCE" \
    --class_balance \
    --neg_weight     0.7593 \
    --pos_weight     1.4642 \
    --train \
    "$@" \
    2>&1 | tee "$REPRO/exp64_custom_cnn_inc.log"; then
  echo "=== exp64_custom_cnn_inc falló ==="
  status=1
fi

if [ "$status" -ne 0 ]; then
  echo "=== Al menos una corrida falló -- revisa los logs en $REPRO ==="
fi

exit "$status"
