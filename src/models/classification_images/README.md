# Classification de regiones de interes (INC)

Repositorio para **entrenar y evaluar un clasificador binario (0/1)** a partir de **parches/imágenes guardadas como `.npy`** y *splits* en CSV.

## Arquitectura

- Backbone de imagen: `ResNetModel` (ResNet50 sin FC).
- Cabeza: `MLP` (capas definidas por `--hidden_layers`).
- Modelo final: `nn.Sequential(image_model, classifier)`.



## 2) Estructura del código

```bash
classification_images/                     # Paquete/proyecto principal (código fuente)
    ├── dataloaders/                           # Módulos de carga y preparación de datos
    │   └── dataloader_images.py               # DataLoader/Dataset para cargar imágenes (y sus labels) desde la ruta definida
    ├── models/                                # Definición y construcción de modelos
    │   ├── pretrained_model/                  # Carpeta destinada a pesos/modelos preentrenados (checkpoints, pesos descargados, etc.)
    │   ├── get_model.py                       # Factory/selector: construye el modelo según argumentos (ResNet/DenseNet/Inception + head MLP, etc.)
    │   ├── image_models.py                    # Backbones de visión (arquitecturas CNN/transfer learning)
    │   └── mlp_models.py                      # Heads/MLPs (capas fully-connected) para clasificación (salida final)
    │
    ├── early_stopping.py                      # Implementación de Early Stopping (controla overfitting con paciencia/monitoreo)
    ├── focal_loss.py                          # Implementación de Focal Loss (útil para desbalance de clases)
    ├── losses.py                              # Wrapper/selector de funciones de pérdida (BCE, CE, focal, etc.)
    ├── main.py                                # Punto de entrada: parsea args, prepara data/modelo y lanza train/test
    ├── metrics.py                             # Métricas de evaluación (accuracy, AUC, sensibilidad, especificidad, etc.)
    ├── options.py                             # Definición centralizada de argumentos CLI (argparse): paths, hiperparámetros, flags train/test
    │
    ├── Readme.md                              # README alterno (posible versión anterior o duplicada)
    ├── README.md                              # README principal del proyecto (documentación)
    ├── run.sh                                 # Script bash con comandos de ejemplo para ejecutar entrenamientos/pruebas
    └── training.py                            # Lógica de entrenamiento/validación: loops, optimizador, scheduler, guardado de checkpoints
```


## 3) Formato de datos

### 3.1. Carpeta de imágenes (`--images_dir`)

El `Dataset` carga cada muestra con:

```python
np.load(os.path.join(images_dir, name_image))
```

Por tanto:

- `--images_dir` debe contener archivos `.npy`.
- El CSV debe contener el **nombre exacto del archivo de la imagen**.

**Formato recomendado del array**:

- Shape: `H x W x 3` (por ejemplo `224 x 224 x 3`)
- dtype: `float32` (recomendado)

El loader aplica `torchvision.transforms.ToTensor()`, que asume un array tipo imagen (HWC) y lo convierte a tensor `C x H x W`.

Si tus `.npy` son **grises** (`H x W` o `H x W x 1`), debes convertir a 3 canales (Sección 9.3).

### 3.2. CSV de splits (`--csv_data_path`)

El directorio `--csv_data_path` debe contener **exactamente**:

- `train_clinical_data.csv`
- `val_clinical_data.csv`
- `test_clinical_data.csv`

Cada CSV se interpreta así:

- Columna 0: nombre del archivo `.npy` (string)
- Columna 1: etiqueta (int 0 o 1)

Ejemplo mínimo (sin encabezado es lo más seguro):

```csv
P001_patch_0001.npy,0
P001_patch_0002.npy,1
P002_patch_0001.npy,0
```


## 4) Pesos preentrenados del backbone

El backbone carga pesos si pasas:

- `--path_image_model /ruta/al/peso.pt`

En `models/image_models.py` se hace:

```python
base_model.load_state_dict(torch.load(weigths_file, map_location=device))
```

## 6) Entrenamiento

Ajusta rutas a tus datos y pesos:

```bash

python3 main.py --exp_name "exp_resnet50" \
                --images_dir "/ruta/a/imagenes_npy" \
                --csv_data_path "/ruta/a/splits" \
                --result_dir "/ruta/a/resultados" \
                --tag_exp exp_resnet50 \
                --image_model "ResNet" \
                --path_image_model "/ruta/a/ResNet50.pt" \
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
                --train
```

## Explicación de hiperparámetros (CLI) — `main.py`

Esta sección documenta **únicamente** los hiperparámetros y flags que afectan el **comportamiento del entrenamiento** (optimización, regularización, augmentación y early stopping).

### `--augmentation`

**Tipo:** flag (booleano)  
**Ejemplo:** `--augmentation`

Activa el **data augmentation** en el `DataLoader` (normalmente aplicado **solo en entrenamiento**, no en validación/test). Su objetivo es mejorar la generalización introduciendo variaciones artificiales de las imágenes.

### `--n_epochs`

**Tipo:** int  
**Ejemplo:** `--n_epochs 200`

Número máximo de épocas de entrenamiento. Una época equivale a una pasada completa por el conjunto de entrenamiento.

### `--batch_size`

**Tipo:** int  
**Ejemplo:** `--batch_size 64`

Cantidad de muestras procesadas por iteración de backpropagation.

### `--lr`

**Tipo:** float  
**Ejemplo:** `--lr 5e-4`

Learning rate inicial del optimizador. Controla el tamaño del paso en la actualización de los pesos.

### `--dropout`

**Tipo:** float (0.0 a 1.0)  
**Ejemplo:** `--dropout 0.5`

Regularización por dropout (generalmente aplicada en el MLP/clasificador). Durante entrenamiento, desactiva aleatoriamente una fracción de neuronas para evitar co-adaptación.


### `--patience_early`

**Tipo:** int  
**Ejemplo:** `--patience_early 50`

Número de épocas sin mejora en la métrica monitorizada (por ejemplo `val_loss` o `val_auc`) antes de activar **Early Stopping**.

### `--min_lr`

**Tipo:** float  
**Ejemplo:** `--min_lr 1e-6`

Learning rate mínimo permitido (usualmente usado junto con un scheduler que reduce LR cuando no hay mejora).


### Aumentos de datos

Si usas `--augmentation`, el train loader aplica:

- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(degrees=15)`
- `RandomVerticalFlip(p=0.2)`
- `GaussianBlur` aleatorio

Validación y test usan solo `ToTensor()`.

---

## 7) Test / Evaluación

### 7.1. Test automático

Si ejecutas con `--train`, al finalizar se ejecuta automáticamente:

- `TrainModel.test_model()`

Ese método:

1) Carga `Saved_Models/Best_Model.pth`
2) Evalúa sobre `test_loader`
3) Guarda métricas y figuras en `Results/`

### 7.2. Métricas reportadas

Se calculan (promedio por batches en test, luego media/std):

- Accuracy
- Sensitivity (Recall)
- Specificity
- F1-score
- VPP (TP/(TP+FP))

Y además:

- Matriz de confusión (`confusion_matrix.png`)
- Curva ROC y AUC (`roc_curve.png`)

---

## 8) Salidas en disco

Con `--result_dir results --exp_name EXP01`:

```bash
results/EXP01/                          # Carpeta de resultados de un experimento (nombre = --exp_name o similar)
  config.txt                            # Configuración usada en el experimento (argumentos CLI/hiperparámetros/paths; sirve para reproducibilidad)
  batch_images.png                      # Ejemplo de batch (debug/inspección visual del DataLoader y transforms)
  Saved_Models/                         # Checkpoints del entrenamiento
    Best_Model.pth                      # Mejor checkpoint global (según métrica monitorizada: val_loss/val_auc/etc.)
    Best_Model_image.pth                # Pesos del backbone de imágenes (modelo de visión) del mejor checkpoint
    Best_Model_classifier.pth           # Pesos del clasificador (MLP/head) del mejor checkpoint
    Model_010.pth ...                   # Checkpoints por época (p.ej. Model_001.pth, Model_002.pth, ...), útiles para análisis/rollback
  Results/                              # Salidas del proceso de test/evaluación (y a veces validación final)
    test_summary.csv                    # Resumen tabular del desempeño en test (métricas por clase y/o globales)
    confusion_matrix.png                # Matriz de confusión renderizada como figura
    roc_curve.png                       # Curva ROC (y AUC) del modelo en test (binaria o multi-clase según implementación)
```

---
