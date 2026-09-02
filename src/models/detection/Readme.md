# Detección de lesiones en mamografía con YOLO (INC)

Este repositorio entrena y evalúa un modelo **YOLO (Ultralytics)** para **detección de lesiones** (cajas delimitadoras / bounding boxes) en mamografía, usando anotaciones en **formato YOLO** **.

---

## Estructura del repositorio

```bash
detection/
├─ main.py                 # Punto de entrada: parsea argumentos, guarda config/código y lanza entrenamiento
├─ options.py              # Argumentos CLI (define paths, hiperparámetros y flags)
├─ train.py                # Entrenamiento con Ultralytics YOLO + logging a W&B
├─ prediction.py           # Predicción/visualización (típicamente tiene rutas hardcodeadas: debes editar)
├─ run.sh                  # Script con comando(s) de ejemplo
├─ data_640.yaml           # YAML de dataset (ejemplo para imgsz=640)
└─ data_1024.yaml          # YAML de dataset (ejemplo para imgsz=1024)
```

## Preparación de los datos

YOLO requiere dataset en formato YOLO (imágenes + labels .txt).

### 1) Estructura esperada del dataset (típica)
```bash
<dataset_root>/
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/            # opcional (recomendado si harás evaluación final)
└─ labels/
   ├─ train/
   ├─ val/
   └─ test/            # opcional
```

**Reglas importantes**

Para cada imagen `<images/<split>/xxx.(png|jpg|jpeg)` debe existir su label `labels/<split>/xxx.txt` con el mismo nombre base.

Si una imagen no tiene objetos, el .txt puede existir vacío (según tu política) o no existir (según tu pipeline). Mantén consistencia.

Las coordenadas de YOLO deben estar normalizadas en [0, 1].

### 2) Formato de labels YOLO

Cada archivo `xxx.txt` contiene una o más líneas con:

```txt
<class_id> <x_center> <y_center> <width> <height>
```

Donde:

- class_id: entero 0..(nc-1)

- `x_center, y_center, width, height`: valores flotantes normalizados respecto al tamaño de la imagen.

### 3) Archivo YAML del dataset (Ultralytics)

El entrenamiento se controla con un archivo .yaml (ej.: data_1024.yaml).

Ejemplo (plantilla típica):

```yaml
path: /ruta/al/dataset_root
train: /ruta/al/dataset_root/images/train
val: /ruta/al/dataset_root/images/val
test: /ruta/al/dataset_root/images/test  # opcional

nc: 2
names:
  0: lesion
  1: lesion_maligna
```

## Cómo entrenar

`main.py` ejecuta el entrenamiento usando Ultralytics YOLO.

Ejemplo completo:

```bash
python3 main.py --path_data /ruta/a/data_1024.yaml \
                --img_size 1024 \
                --batch_size 16 \
                --epochs 300 \
                --device 0 \
                --pretrained \
                --path_project /ruta/a/resultados/detection_experiments \
                --experiment_name exp_yolo1024_bs16_ep300 \
                --optimizer auto \
                --seed 42 \
                --learning_rate 1e-3 \
                --freeze_layers 8 \
                --box_weight 4.0 \
                --classification_weight 0.5 \
                --dropout_rate 0.0 \
                --model_path yolo11_m_digitaleye.pt \
                --save_period 20
```

**Parámetros principales:**

- path_data: ruta al .yaml del dataset (formato Ultralytics).

- img_size: tamaño de entrada (imgsz) del modelo (ej. 640, 1024).

- batch_size: batch de entrenamiento.

- epochs: número de épocas.

- device: dispositivo (típicamente 0 para GPU0 o cpu).

- pretrained: activa entrenamiento con pesos preentrenados (si aplica al modelo base).

- path_project: carpeta raíz donde se guardan resultados del experimento.

- experiment_name: nombre del experimento (subcarpeta dentro de path_project).

- optimizer: optimizador (auto, SGD, AdamW, etc. según Ultralytics).

- learning_rate: LR inicial (lr0 en Ultralytics).

- freeze_layers: congela capas iniciales (útil en transfer learning).

- box_weight: peso de la pérdida de localización (box regression).

- classification_weight: peso de la pérdida de clasificación.

- dropout_rate: dropout (si el modelo lo soporta).

- model_path: pesos/modelo base (por ejemplo un .pt de Ultralytics).

- save_period: periodicidad de guardado de checkpoints.

### Qué se guarda como salida

Ultralytics crea una carpeta de experimento dentro de:

```bash
<path_project>/<experiment_name>/
```

Incluye:
```bash
<path_project>/<experiment_name>/
├─ weights/
│  ├─ best.pt           # Mejor checkpoint (según métrica de validación)
│  └─ last.pt           # Último checkpoint
├─ results.csv          # Métricas por época (según versión)
├─ confusion_matrix.png # Si aplica/si se genera
├─ PR_curve.png         # Curvas (según versión)
└─ ...                  # Otras figuras y logs de Ultralytics
```



