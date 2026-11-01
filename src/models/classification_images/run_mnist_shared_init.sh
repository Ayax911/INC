#!/usr/bin/env bash
# ==============================================================================
# MNIST smoke test -- variante de run_mnist_smoketest.sh con el backbone y la cabeza
# inicializados desde los checkpoints compartidos con FedMammoBench, en vez de que cada
# framework dibuje su propia inicialización aleatoria. Ver
# FedMammoBench/scripts/generate_shared_init_mnist.py para el porqué completo.
#
# Comparación de paridad puntual -- no un modo de uso general de este proyecto. Todas las
# demás flags quedan idénticas a run_mnist_smoketest.sh (ver ese archivo para la
# justificación de cada una); las únicas diferencias son:
#   - Se quita --from_scratch (que ignoraría --path_image_model) y se agrega
#     --path_image_model apuntando a backbone.pt -- ver get_image_model() en
#     models/image_models.py: sin --pretrained ni --from_scratch, carga el checkpoint dado.
#   - Se invoca run_mnist_shared_init.py en vez de main.py: ese script inyecta head_inc.pt
#     en trainer.model[1] (la cabeza) justo después de construir TrainModel, porque
#     options.py no tiene una flag equivalente a --path_image_model para la cabeza.
#   - Se agrega --dataloader_seed 42: con init compartido, la corrida anterior seguía
#     divergiendo de FedMammoBench en la clase par (TN=4900/FP=26 en FMB vs TN=4905/FP=21
#     acá) porque el train DataLoader de este proyecto no tenía generator= propio y leía del
#     RNG global (ver dataloaders/dataloader_images.py:Loader.train_dataloader), mientras que
#     FedMammoBench siempre desacopla el shuffle en su propio torch.Generator
#     (FedMammoBench/src/seed.py:make_generator). Este flag replica eso acá. Resultado medido:
#     sincronizó exacto la clase par (TN/FP) pero la impar divergió más (TP/FN) -- el residuo
#     total no bajó, se movió. Sigue sin cubrir dropout (ver el siguiente punto) ni floats.
#   - --dropout 0 (no 0.3): dropout lee del RNG global de cada proceso, sin generator propio
#     -- otra fuente de divergencia no cubierta por --dataloader_seed. En 0, ni este proyecto
#     ni FedMammoBench instancian la capa Dropout (`if dropout > 0:` en ambos MLP builders),
#     así que deja de aportar ruido. Requirió regenerar head_fmb.pt/head_inc.pt con
#     HEAD_KWARGS["dropout"]=0 (ver generate_shared_init_mnist.py) porque quitar la capa
#     Dropout corre los índices de los Linear subsiguientes en el Sequential.
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="/home/akira/snap/steam/federallearning/FedMammoBench/.venv/bin/python"
IMAGES_DIR="/home/akira/snap/steam/mnist/images_tif64"
CSV_DATA_PATH="../../../manifest/mnist_smoketest"
RESULT_DIR="../../../results"
BACKBONE_INIT="/home/akira/snap/steam/federallearning/FedMammoBench/runs/shared_init_mnist/backbone.pt"

"$PYTHON" run_mnist_shared_init.py \
  --exp_name        inc_mnist_smoketest_shared_init \
  --images_dir      "$IMAGES_DIR" \
  --csv_data_path   "$CSV_DATA_PATH" \
  --result_dir      "$RESULT_DIR" \
  --wandb_project   "inc-mnist-smoketest" \
  --tag_exp         smoketest mnist shared_init \
  --image_model     ResNet18 \
  --path_image_model "$BACKBONE_INIT" \
  --num_freeze      0 \
  --hidden_layers   128 \
  --output_size     2 \
  --activation      LeakyReLU \
  --dropout         0 \
  --img_size        64,64 \
  --n_epochs        3 \
  --batch_size      128 \
  --lr              0.001 --b1 0.9 --b2 0.999 \
  --patience_early  100 \
  --min_lr          1e-6 \
  --loss            BCE \
  --class_balance --pos_weight 1.0 --neg_weight 1.0 \
  --dataloader_seed 42 \
  --train \
  2>&1 | tee "$RESULT_DIR/inc_mnist_smoketest_shared_init.log"
