#!/usr/bin/env bash
# ==============================================================================
# MNIST smoke test -- directly comparable to FedMammoBench's
# FedMammoBench/configs/exp_mnist_smoketest.yaml. Same 64x64 TIFF files
# (mnist/scripts/make_smoketest_tif64.py), same even/odd binary label, same
# optimizer/scheduler/backbone/head hyperparameters -- every value below is
# pinned explicitly instead of relying on options.py defaults, so this file
# is the single source of truth for what "comparable" means here.
#
# Backbone: ResNet18, random init, num_freeze=0 (100% trainable) -- matches
#   FedMammoBench's resnet18_scratch + unfreeze_from: conv1.
# Head: MLP [128], LeakyReLU (this script's default negative_slope 0.2 in
#   models/mlp_models.py), dropout 0.3 -- FedMammoBench's head:configurable_mlp
#   was built to reproduce this exact class.
# Optimizer: plain Adam (this script's TrainModel hardcodes torch.optim.Adam,
#   no weight decay) -- FedMammoBench side set to optimizer.name: adam to match
#   (not its own default adamw).
# Scheduler: CosineAnnealingLR(T_max=n_epochs, eta_min=min_lr) -- hardcoded in
#   training.py, not optional here -- FedMammoBench side set to scheduler.name:
#   cosine with matching T_max/eta_min instead of disabled.
# Normalization: --normalize_mean/--normalize_std deliberately NOT passed --
#   this dataloader never calls ToTensor()/Normalize(), images stay raw uint8
#   0..255 cast to float. FedMammoBench side compensates with
#   Normalize(mean=0, std=1/255) to reproduce this same raw 0..255 scale,
#   since its own image path always divides by 255 first via TF.to_tensor().
# Resolution: --img_size 64,64 is informational only -- this dataloader never
#   resizes, it assumes images already match on disk, which they do
#   (images_tif64/ is pre-resized to 64x64, shared byte-for-byte with FedMammoBench).
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="/home/akira/snap/steam/federallearning/FedMammoBench/.venv/bin/python"
IMAGES_DIR="/home/akira/snap/steam/mnist/images_tif64"
CSV_DATA_PATH="../../../manifest/mnist_smoketest"
RESULT_DIR="../../../results"

"$PYTHON" main.py \
  --exp_name        inc_mnist_smoketest \
  --images_dir      "$IMAGES_DIR" \
  --csv_data_path   "$CSV_DATA_PATH" \
  --result_dir      "$RESULT_DIR" \
  --wandb_project   "inc-mnist-smoketest" \
  --tag_exp         smoketest mnist \
  --image_model     ResNet18 \
  --from_scratch \
  --num_freeze      0 \
  --hidden_layers   128 \
  --output_size     2 \
  --activation      LeakyReLU \
  --dropout         0.3 \
  --img_size        64,64 \
  --n_epochs        3 \
  --batch_size      128 \
  --lr              0.001 --b1 0.9 --b2 0.999 \
  --patience_early  100 \
  --min_lr          1e-6 \
  --loss            BCE \
  --class_balance --pos_weight 1.0 --neg_weight 1.0 \
  --train \
  2>&1 | tee "$RESULT_DIR/inc_mnist_smoketest.log"
