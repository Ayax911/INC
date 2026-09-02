# Clasificación final (Imágenes + datos clínicos) — INC

Este repositorio entrena y evalúa un **clasificador binario** que combina:

- **Features de imágenes** extraídas por un backbone (ResNet/DenseNet/Inception) **preentrenado y congelado**.
- **Features clínicas** extraídas desde una **capa intermedia** de un MLP clínico **preentrenado y congelado**.

Luego concatena ambos vectores y entrena un **MLP final** (trainable) para producir **2 logits** (clase 0 y clase 1).

> Importante: en el código actual **NO se entrena** el modelo de imágenes ni el modelo clínico (ambos se congelan). Solo se entrena el **MLP final**.

---

## Estructura del repositorio

```bash
classification_final/
├─ main.py                       # Punto de entrada (train + test)
├─ options.py                    # Argumentos CLI
├─ training.py                   # Entrenamiento, validación y test (genera métricas + ROC + confusión)
├─ losses.py                     # Pérdidas: "BCE" (CrossEntropy ponderada) y "Focal"
├─ focal_loss.py                 # Implementación de focal loss
├─ metrics.py                    # Métricas (accuracy, sensitivity, specificity, F1, VPP, etc.)
├─ early_stopping.py             # EarlyStopping (monitoriza Val_F1-Score, guarda Best_Model.pth)
├─ dataloaders/
│  └─ dataloader_images.py       # Loader que espera 3 CSV con nombres fijos (train/val/test)
├─ models/
│  ├─ get_model.py               # Define el modelo final (imagen + clínica + MLP final)
│  ├─ image_models.py            # Backbones (ResNet50, DenseNet121, InceptionV3) congelados
│  └─ mlp_models.py              # Definición genérica de MLP
├─ run.sh                        # Ejemplo de comando real usado en entrenamiento
└─ results/                      # Ejemplo de salida (Test_Code/)
```


## Preparación de los datos

### 1) Directorio de imágenes (--images_dir)

Debe contener archivos .npy (uno por muestra).

El loader hace:

```python
np.load(os.path.join(images_dir, f"{ID}"))
```

Por lo tanto, la columna ID del CSV debe coincidir exactamente con el nombre del archivo (incluyendo extensión si aplica), por ejemplo:

```
ID = "P001_patch_0001.npy" → existe images_dir/P001_patch_0001.npy
```

**Formato del .npy:**

Debe ser compatible con torchvision.transforms.ToTensor().

Recomendado: float32 con forma (H, W, C) o (H, W) (si es 1 canal).

Si vas a usar ResNet estándar, lo más seguro es guardar como 3 canales (H, W, 3) (por ejemplo replicando el canal gris).

```
Nota: el loader NO hace resize. Tus .npy deben ya estar en el tamaño correcto (por defecto se asume 224×224 en el proyecto).
```

### 2) Directorio de CSV (--csv_data_path)

El loader NO recibe un CSV individual, sino un directorio que debe contener exactamente estos archivos (nombres fijos):

```bash
<csv_data_path>/
├─ train_clinical_data.csv
├─ val_clinical_data.csv
└─ test_clinical_data.csv
```

Formato esperado de cada CSV

Cada CSV debe incluir al menos:

- ID: identificador (string) usado para ubicar el .npy.
- Etiqueta: etiqueta (0/1).
- El resto de columnas: features clínicas numéricas (int/float).

El modelo clínico usa:

**Reglas importantes:**

- Las columnas (features clínicas) deben ser idénticas en train/val/test, con el mismo orden.
- No debe haber NaN. Imputa o elimina antes.
- Variables categóricas deben convertirse a numéricas antes (one-hot, ordinal, etc.).
- --clinic_input_size debe coincidir con el número de columnas clínicas (todas excepto ID y Etiqueta).

### Modelos preentrenados requeridos:

Este repo necesita dos checkpoints para poder correr:

- Backbone de imágenes (--path_image_model)
- MLP clínico preentrenado (--path_clinic_model)

**Importante:** el checkpoint clínico debe ser compatible con la arquitectura definida por:
--clinic_input_size, --clinic_hidden_layers, --clinic_activation, --clinic_dropout.
Si no coincide, load_state_dict fallará.

### Cómo entrenar

main.py cuando ejecutas entrenamiento:

Entrena el MLP final.

Al terminar, ejecuta test automáticamente con Best_Model.pth.

Ejemplo completo:

```bash
python3 main.py --exp_name FinalClassification_v1 \
                --images_dir "/ruta/a/imagenes_npy" \
                --csv_data_path "/ruta/a/csv_dir" \
                --result_dir "/ruta/a/resultados" \
                --tag_exp INC Final Classification \
                --image_model "ResNet" \
                --path_image_model "models/pretrained_models/ResNet50.pt" \
                --path_clinic_model "models/pretrained_models/data_clinic.pth" \
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
                --b1 0.5 \
                --b2 0.999 \
                --patience_early 50 \
                --min_lr 1e-6 \
                --loss "BCE" \
                --class_balance \
                --pos_weight 1.0 \
                --neg_weight 2.0 \
                --train

```

## Parámetros principales

### Datos

- --images_dir: carpeta con .npy.
- --csv_data_path: carpeta con train_clinical_data.csv, val_clinical_data.csv, test_clinical_data.csv.

### Modelos preentrenados

- --path_image_model: pesos del backbone de imágenes.
- --path_clinic_model: checkpoint del MLP clínico.

### Modelo clínico (para extracción de features)

- --clinic_input_size: número de features clínicas.
- --clinic_hidden_layers: arquitectura usada para construir el MLP clínico (debe coincidir con el checkpoint).
- --clinic_activation, --clinic_dropout: deben coincidir con el checkpoint.
- --clinic_idx_hidden_layer: índice de la capa (en nn.Sequential) desde la cual se extraen features.

### Modelo final

- --final_input_size: dimensión de la concatenación [image_features || clinic_features].
- --final_hidden_layers: arquitectura del MLP final.
- --final_output_size: por defecto 2.
- --final_activation, --final_dropout.

### Entrenamiento

- --augmentation: activa flips/rotaciones/blur en train.
- --n_epochs, --batch_size, --lr, --b1, --b2.
- --patience_early: early stopping (monitoriza Val_F1-Score).
- --min_lr: mínimo para CosineAnnealingLR.

### Pérdida y balanceo

- --loss: "BCE" (CrossEntropy ponderada) o "Focal".
- --class_balance: si está activo usa pesos pos_weight/neg_weight.
- --pos_weight, --neg_weight: pesos de clase (para CE ponderada o Focal).
- --gamma: parámetro de focal loss.


## Salidas del experimento

Se crean carpetas dentro de:

```bash
<result_dir>/<exp_name>/
```

Estructura típica:

```bash
results/EXP01/
  config.txt
  batch_images.png
  Saved_Models/
    Best_Model.pth               # estado completo del modelo (imagen+clínica+final, aunque imagen/clínica están congelados)
    Best_Model_Final.pth         # pesos del MLP final únicamente
    classifier_010.pth ...
    Model_010.pth ...
  Results/
    test_summary.csv
    confusion_matrix.png
    roc_curve.png
```

- config.txt: argumentos CLI guardados.
- batch_images.png: ejemplo de batch (útil para validar transforms).
- test_summary.csv: media y std de métricas en test.
- confusion_matrix.png: matriz de confusión.
- roc_curve.png: curva ROC y AUC (usando probabilidad de la clase 1).

