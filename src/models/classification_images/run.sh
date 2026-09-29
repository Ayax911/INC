#!/usr/bin/env bash
# ==============================================================================
# Comparación de backbones: ResNet50/18 desde cero vs ResNet18 ImageNet vs CNN
# custom (INC). Par del lado INC de FedMammoBench (exp61, exp62, exp63, exp64)
#
# Corridas incluidas:
#   1. exp61_resnet50_scratch_inc: ResNet50 inicializado desde cero (pesos
#      aleatorios, todo entrenable: --from_scratch --num_freeze 0), sobre norm_neg1_1.
#   2. exp62_resnet18_scratch_inc: ResNet18 inicializado desde cero (pesos
#      aleatorios, todo entrenable: --from_scratch --num_freeze 0), sobre norm_neg1_1.
#   3. exp63_resnet18_imagenet_inc: ResNet18 con pesos ImageNet (torchvision
#      IMAGENET1K_V1), backbone congelado por default (--num_freeze 60), sobre
#      norm_0_1 con normalización (0.449 / 0.226).
#   4. exp64_custom_cnn_inc: backbone CNN custom (4 bloques Conv-BN-ReLUx2 +
#      MaxPool, canales 32/64/128/256, GAP), entrenado desde cero, sobre
#      norm_neg1_1. Par de FedMammoBench/configs/exp64_custom_cnn.yaml -- ver
#      ese archivo para la justificación completa de cada hiperparámetro.
#      exp64 y no exp61 porque exp61/62/63 ya tienen resultados reales
#      publicados acá (este mismo commit los trae) -- no se reutiliza un
#      número con resultados existentes para un experimento distinto.
#
# Justificación de norm/normalize para exp63:
#   ImageNet espera píxeles en [0, 1] + estadísticas de ImageNet promediadas a
#   un canal (0.449 / 0.226) debido a que las mamografías son en escala de grises
#   y se replican a 3 canales.
#
# Ejecución:
#   Las 4 corridas se ejecutan EN SECUENCIA porque con backbone entrenable una
#   sola corrida ya ocupa la GPU. Si una corrida falla, el script continúa con
#   la siguiente y al final retorna un código de salida distinto de 0 si alguna falló.
#
#   Para cambiar el congelamiento de ResNet18 ImageNet (default 60 = 100% congelado):
#     NUM_FREEZE_R18=0 ./run.sh
# ==============================================================================
set -uo pipefail
cd "$(dirname "$0")"

BASE="/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/13-PregradoJulian/Federal Learning/infraestructura federada"
INC_DIR="$BASE/INC"
REPRO="$BASE/inc_repro_data"
RESULTS_DIR="$INC_DIR/results"
PYTHON="/home/imagenesmedicas/miniconda3/envs/inc-combined/bin/python3"

IMGROOT="/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench/preproccesed_julian"

# NUM_FREEZE_R18: 60 = backbone ResNet18 100% congelado (default), 0 = todo entrenable.
# Valores intermedios congelan los primeros N tensores de parámetros.
NUM_FREEZE_R18="${NUM_FREEZE_R18:-60}"

mkdir -p "$RESULTS_DIR"
status=0

# ==============================================================================
# 1. exp61 -- ResNet50 desde cero (todo entrenable)
# ==============================================================================
echo "=== running exp61_resnet50_scratch_inc ==="
if ! "$PYTHON" main.py \
    --exp_name       "exp61_resnet50_scratch_inc" \
    --images_dir     "$IMGROOT" \
    --csv_data_path  "$REPRO/csvs_norm_neg1_1" \
    --result_dir     "$RESULTS_DIR" \
    --wandb_project  "fedmammobench2.0" \
    --wandb_group    "resnet_scratch_vs_resnet18" \
    --tag_exp        "inc" "resnet_scratch_vs_resnet18" \
    --image_model    "ResNet" \
    --from_scratch \
    --num_freeze     0 \
    --hidden_layers  1024 \
    --output_size    2 \
    --activation     "LeakyReLU" \
    --dropout        0.2 \
    --img_size       224,224 \
    --augmentation \
    --n_epochs       200 \
    --batch_size     64 \
    --lr             1e-3 \
    --b1             0.5 \
    --b2             0.999 \
    --patience_early 100 \
    --min_lr         1e-6 \
    --loss           "BCE" \
    --class_balance \
    --neg_weight     0.7593 \
    --pos_weight     1.4642 \
    --train \
    "$@" \
    2>&1 | tee "$REPRO/exp61_resnet50_scratch_inc.log"; then
  echo "=== exp61_resnet50_scratch_inc falló ==="
  status=1
fi

# ==============================================================================
# 2. exp62 -- ResNet18 desde cero (todo entrenable)
# ==============================================================================
echo "=== running exp62_resnet18_scratch_inc ==="
if ! "$PYTHON" main.py \
    --exp_name       "exp62_resnet18_scratch_inc" \
    --images_dir     "$IMGROOT" \
    --csv_data_path  "$REPRO/csvs_norm_neg1_1" \
    --result_dir     "$RESULTS_DIR" \
    --wandb_project  "fedmammobench2.0" \
    --wandb_group    "resnet_scratch_vs_resnet18" \
    --tag_exp        "inc" "resnet_scratch_vs_resnet18" \
    --image_model    "ResNet18" \
    --from_scratch \
    --num_freeze     0 \
    --hidden_layers  1024 \
    --output_size    2 \
    --activation     "LeakyReLU" \
    --dropout        0.2 \
    --img_size       224,224 \
    --augmentation \
    --n_epochs       200 \
    --batch_size     64 \
    --lr             1e-3 \
    --b1             0.5 \
    --b2             0.999 \
    --patience_early 100 \
    --min_lr         1e-6 \
    --loss           "BCE" \
    --class_balance \
    --neg_weight     0.7593 \
    --pos_weight     1.4642 \
    --train \
    "$@" \
    2>&1 | tee "$REPRO/exp62_resnet18_scratch_inc.log"; then
  echo "=== exp62_resnet18_scratch_inc falló ==="
  status=1
fi

# ==============================================================================
# 3. exp63 -- ResNet18 ImageNet (backbone congelado configurable)
# ==============================================================================
echo "=== running exp63_resnet18_imagenet_inc ==="
if ! "$PYTHON" main.py \
    --exp_name       "exp63_resnet18_imagenet_inc" \
    --images_dir     "$IMGROOT" \
    --csv_data_path  "$REPRO/csvs_norm_0_1" \
    --result_dir     "$RESULTS_DIR" \
    --wandb_project  "fedmammobench2.0" \
    --wandb_group    "resnet_scratch_vs_resnet18" \
    --tag_exp        "inc" "resnet_scratch_vs_resnet18" \
    --image_model    "ResNet18" \
    --pretrained \
    --num_freeze     "$NUM_FREEZE_R18" \
    --normalize_mean 0.449 \
    --normalize_std  0.226 \
    --hidden_layers  1024 \
    --output_size    2 \
    --activation     "LeakyReLU" \
    --dropout        0.2 \
    --img_size       224,224 \
    --augmentation \
    --n_epochs       200 \
    --batch_size     64 \
    --lr             1e-3 \
    --b1             0.5 \
    --b2             0.999 \
    --patience_early 100 \
    --min_lr         1e-6 \
    --loss           "BCE" \
    --class_balance \
    --neg_weight     0.7593 \
    --pos_weight     1.4642 \
    --train \
    "$@" \
    2>&1 | tee "$REPRO/exp63_resnet18_imagenet_inc.log"; then
  echo "=== exp63_resnet18_imagenet_inc falló ==="
  status=1
fi

# ==============================================================================
# 4. exp64 -- backbone CNN custom, desde cero (todo entrenable)
#
#    --output_size 2 (no 1): "--loss BCE" acá es en realidad nn.CrossEntropyLoss
#    ponderada de 2 clases (ver losses.py), no BCEWithLogitsLoss de 1 logit --
#    no puede reproducir el "Dense(1)+sigmoid" literal del diseño original ni
#    con --output_size 1. FedMammoBench/configs/exp64_custom_cnn.yaml sí usa 1
#    logit real (loss: bce); esta es la comparación más cercana que el código
#    de pérdida de este repo permite.
#
#    TODOS los flags de options.py que aplican están explícitos, aunque su
#    valor coincida con el default (misma convención que los 3 bloques de
#    arriba). Los que quedan afuera y por qué:
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
