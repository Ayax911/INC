#!/usr/bin/env bash
set -euo pipefail

# Obtener directorio del script (raíz del proyecto INC)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Ejecutando entrenamiento de Clasificación de Imágenes en INC ==="
cd "$ROOT_DIR/src/models/classification_images"
./run.sh "$@"
