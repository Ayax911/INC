#!/usr/bin/env bash
# Réplica INC de la ablación de pretraining exp17/18/19 de FedMammoBench --
# el par del lado INC de FedMammoBench/run_pretrain_ablation.sh, mismo criterio
# que exp04-vs-expA y exp05: se corre lo mismo en los dos proyectos para que la
# comparación sea pareada.
#
# Las tres corridas leen exactamente los mismos TIFF que FedMammoBench
# (preproccesed_julian/{norm_0_1,norm_neg1_1}), no una segunda conversión: lo
# único traducido es el índice, vía inc_repro_data/prepare_manifest_csvs_for_inc.py.
#
# En W&B van al proyecto de FedMammoBench (fedmammobench2.0) y al mismo
# wandb_group que los configs originales -- los grupos de W&B son por proyecto,
# así que loguear al proyecto del INC las dejaría fuera del grupo. El nombre de
# cada corrida es el experiment_id del YAML equivalente + sufijo "_inc", así que
# en la UI el grupo queda con los 6 runs pareados de a dos. training.py además
# duplica cada métrica bajo el nombre que usa FedMammoBench (train_loss,
# val_f1_macro, etc.) para que los paneles del grupo se puedan comparar
# directamente -- ver el comentario junto a esos wandb.log() en training.py.
#
# Los tres se lanzan EN PARALELO (cada uno en su propio proceso, los tres a la
# vez en la misma GPU) en vez de en cola -- con el backbone 100% congelado
# (--num_freeze 159) un solo run no satura la GPU, así que correr los tres
# a la vez termina el trío antes que uno-tras-otro, a cambio de que cada uno
# individualmente tarde más por época al repartir la GPU. La salida de la
# terminal sale entreverada entre los tres (cada uno imprime su propia barra de
# progreso) -- para seguir uno a la vez usá su log:
#   tail -f "$REPRO/<exp_name>.log"
#
# Correr:
#   ./run_pretrain_ablation_inc.sh
set -uo pipefail   # sin -e: al correr en paralelo, un solo `&` que falle no
                   # debe matar a los otros dos -- los códigos de salida se
                   # revisan a mano en el `wait` de más abajo.
cd "$(dirname "$0")"

BASE="/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/13-PregradoJulian/Federal Learning/infraestructura federada"
REPRO="$BASE/inc_repro_data"
PYTHON="/home/imagenesmedicas/miniconda3/envs/inc-combined/bin/python3"

# Mismo data.image_root que los YAML de exp17/18/19. Las rutas de los CSV ya
# traen el prefijo norm_0_1/ o norm_neg1_1/, así que el image_root es el padre.
IMGROOT="/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench/preproccesed_julian"

# Pesos de RadImageNet -- el mismo archivo que architecture.weights_path de exp18.
RADIMAGENET="$BASE/FedMammoBench/weights/ResNet50.pt"

# --num_freeze 159 = número exacto de tensores de parámetros del backbone
# ResNet50 recortado (children()[:9]), o sea el equivalente de
# `unfreeze_from: none` de los YAML: backbone 100% congelado.
run_exp () {
  local exp_name="$1"; shift

  echo "=== running $exp_name ==="
  "$PYTHON" main.py \
      --exp_name       "$exp_name" \
      --images_dir     "$IMGROOT" \
      --result_dir     "$REPRO/results" \
      --wandb_project  "fedmammobench2.0" \
      --wandb_group    "pretrain_ablation_imagenet_vs_radimagenet" \
      --tag_exp        "inc" "pretrain_ablation" \
      --image_model    "ResNet" \
      --num_freeze     159 \
      --hidden_layers  1024 \
      --output_size    2 \
      --activation     "LeakyReLU" \
      --dropout        0.2 \
      --img_size       224,224 \
      --augmentation \
      --n_epochs       200 \
      --batch_size     64 \
      --lr             1e-3 --b1 0.5 --b2 0.999 \
      --patience_early 100 \
      --min_lr         1e-6 \
      --loss           "BCE" \
      --class_balance --neg_weight 0.7593 --pos_weight 1.4642 \
      --train \
      "$@" \
      2>&1 | tee "$REPRO/${exp_name}.log"
}

# exp17 -- ImageNet con su normalización correcta: píxeles en [0, 1] (norm_0_1)
# normalizados con las estadísticas de ImageNet promediadas a 1 canal
# (0.449/0.226), porque el TIFF llega en 1 canal y la réplica a 3 ocurre después
# de Normalize. --pretrained ignora --path_image_model y usa los pesos de
# torchvision.
run_exp exp17_pretrain_ablation_imagenet_inc \
    --csv_data_path "$REPRO/csvs_norm_0_1" \
    --pretrained \
    --normalize_mean 0.449 --normalize_std 0.226 &
pid17=$!

# exp18 -- RadImageNet con su normalización nativa: píxeles ya en [-1, 1]
# (norm_neg1_1) y sin Normalize adicional.
run_exp exp18_pretrain_ablation_radimagenet_inc \
    --csv_data_path "$REPRO/csvs_norm_neg1_1" \
    --path_image_model "$RADIMAGENET" &
pid18=$!

# exp19 -- control "mal alimentado" a propósito: backbone de ImageNet pero con
# la normalización de RadImageNet sin corregir ([-1, 1], sin Normalize). No es
# la corrida que se cita como "el resultado de ImageNet" -- esa es exp17.
run_exp exp19_pretrain_ablation_imagenet_mismatched_norm_inc \
    --csv_data_path "$REPRO/csvs_norm_neg1_1" \
    --pretrained &
pid19=$!

status=0
for pid in "$pid17" "$pid18" "$pid19"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [ "$status" -ne 0 ]; then
  echo "=== al menos una corrida falló -- revisá $REPRO/exp1{7,8,9}_*.log ==="
fi
exit "$status"
