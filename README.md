# Proyecto de Clasificación y Detección de Lesiones Mamarias

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema completo de detección y clasificación de lesiones mamarias en imágenes DICOM utilizando técnicas de Deep Learning. El sistema combina:

1. **Detección de lesiones** mediante un modelo YOLO (You Only Look Once)
2. **Clasificación de malignidad** utilizando imágenes y datos clínicos mediante un modelo híbrido que combina ResNet50 y MLP (Perceptrón Multicapa)

El pipeline completo procesa imágenes DICOM de mamografías, detecta regiones de interés (ROIs), las clasifica como benignas o malignas, y genera visualizaciones con las predicciones.

## 🎯 Características Principales

- ✅ Procesamiento automático de imágenes DICOM
- ✅ Aplicación de ventaneo médico (windowing) para optimizar contraste
- ✅ Detección automática de lesiones con modelo tipo YOLO
- ✅ Clasificación de lesiones combinando datos de imagen y datos clínicos
- ✅ Generación de visualizaciones con bounding boxes y probabilidades
- ✅ Pipeline completo de entrenamiento y evaluación para múltiples modelos
- ✅ Preprocesamiento robusto de imágenes y datos

## 🚀 Ejecución del Pipeline de Inferencia

### Requisitos Previos

1. **Instalar el entorno conda:**
```bash
conda env create -f environment.yml
conda activate inc-combined
```

2. **Modelos pre-entrenados requeridos:**
   - Modelo YOLO para detección (archivo `.pt`)
   - Modelo de clasificación final (archivo `.pth`)

Los modelos se pueden encontrar en el siguiente repositorio: [Modelos Pre-entrenados](https://inccancer-my.sharepoint.com/personal/kosorno_cancer_gov_co/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fkosorno%5Fcancer%5Fgov%5Fco%2FDocuments%2FAnexosSegundoInforme&ga=1)

### Ejecutar la Inferencia

Para ejecutar el pipeline de inferencia sobre una imagen DICOM:

```bash
python scripts/pipeline-inferencia.py \
    --dcm-path /ruta/a/imagen.dcm \
    --output-dir /ruta/salida \
    --yolo-path-model /ruta/modelo_yolo.pt \
    --model-path /ruta/modelo_clasificacion.pth
```

**Parámetros:**
- `--dcm-path`: Ruta al archivo DICOM de entrada
- `--output-dir`: Directorio donde se guardarán los resultados
- `--yolo-path-model`: Ruta al modelo YOLO entrenado para detección de lesiones
- `--model-path`: Ruta al modelo de clasificación final entrenado

**Ejemplo:**
```bash
python scripts/pipeline-inferencia.py \
    --dcm-path data/mamografia_paciente_001.dcm \
    --output-dir resultados/inferencia \
    --yolo-path-model modelos/yolo_lesiones_best.pt \
    --model-path modelos/clasificador_final_best.pth
```

### Salidas del Pipeline

El pipeline generará:
- **Imágenes procesadas** en formato JPG (1024x1024 pixels)
- **Imágenes con bounding boxes** mostrando las lesiones detectadas
- **Predicciones por consola** con probabilidades de malignidad para cada lesión

**NOTA:** El pipeline incluye datos clínicos simulados en el código. Para usar datos reales del paciente, modifique el diccionario `datos_clinicos` en el archivo [pipeline-inferencia.py](scripts/pipeline-inferencia.py#L107-L132).

## 📁 Estructura del Proyecto

```
inc-project-models-classification-detection/
│
├── environment.yml                 # Configuración del entorno conda
├── README.md                       # Documentación del proyecto (este archivo)
│
├── models/                         # Definiciones de arquitecturas de modelos
│   ├── __init__.py
│   └── model.py                    # Modelo híbrido MLP_Final_Model (ResNet50 + MLP)
│
├── notebooks/                      # Notebooks de Jupyter para análisis
│   ├── dividir-conjunto-datos.ipynb        # División de datasets
│   └── procesar-datos-clinicos.ipynb       # Procesamiento de datos clínicos
│
├── scripts/                        # Scripts ejecutables principales
│   ├── __init__.py
│   ├── pipeline-inferencia.py      # Pipeline completo de inferencia
│   └── preprocesar-datos.py        # Preprocesamiento de datos
│
└── src/                            # Código fuente del proyecto
    ├── __init__.py
    │
    ├── data/                       # Módulos de procesamiento de datos
    │   ├── __init__.py
    │   ├── data_validator.py       # Validación de datos
    │   ├── dataset_generator.py    # Generación de datasets
    │   ├── dataset_splitter.py     # División de datasets en train/test/val
    │   └── run.sh                  # Script de ejecución
    │
    └── models/                     # Implementaciones de modelos
        │
        ├── classification_data_clinic/     # Clasificación usando datos clínicos
        │   ├── main.py                     # Script principal de entrenamiento
        │   ├── training.py                 # Lógica de entrenamiento
        │   ├── options.py                  # Configuración de hiperparámetros
        │   ├── metrics.py                  # Métricas de evaluación
        │   ├── losses.py                   # Funciones de pérdida
        │   ├── early_stopping.py           # Implementación de early stopping
        │   ├── README.md                   # Documentación específica
        │   ├── run.sh                      # Script de ejecución
        │   ├── dataloaders/
        │   │   └── data_loader_data.py     # Carga de datos clínicos
        │   └── models/
        │       ├── get_model.py            # Factory de modelos
        │       └── mlp_models.py           # Arquitecturas MLP
        │
        ├── classification_images/          # Clasificación usando imágenes
        │   ├── main.py                     # Script principal
        │   ├── training.py                 # Entrenamiento del modelo
        │   ├── test.py                     # Evaluación del modelo
        │   ├── options.py                  # Opciones de configuración
        │   ├── metrics.py                  # Métricas (Accuracy, F1, etc.)
        │   ├── losses.py                   # Funciones de pérdida
        │   ├── focal_loss.py               # Implementación de Focal Loss
        │   ├── early_stopping.py           # Early stopping
        │   ├── run.sh, run2.sh             # Scripts de ejecución
        │   ├── dataloaders/
        │   │   └── dataloader_images.py    # Cargador de imágenes
        │   └── models/
        │       ├── get_model.py            # Selector de modelo
        │       ├── image_models.py         # Modelos CNN (ResNet, DenseNet, etc.)
        │       ├── mlp_models.py           # MLPs auxiliares
        │       └── classifier_model.py     # Clasificadores personalizados
        │
        ├── classification_final/           # Clasificación híbrida (imágenes + datos clínicos)
        │   ├── main.py                     # Script principal
        │   ├── training.py                 # Entrenamiento
        │   ├── test.py                     # Evaluación
        │   ├── options.py                  # Configuración
        │   ├── metrics.py                  # Métricas
        │   ├── losses.py                   # Funciones de pérdida
        │   ├── focal_loss.py               # Focal Loss
        │   ├── early_stopping.py           # Early stopping
        │   ├── run.sh, run2.sh             # Scripts de ejecución
        │   ├── dataloaders/
        │   │   └── dataloader_images.py    # Carga de datos multimodales
        │   └── models/
        │       ├── get_model.py            # Factory de modelos
        │       ├── image_models.py         # Modelos de imagen
        │       └── mlp_models.py           # MLPs
        │
        └── detection/                      # Detección de lesiones con YOLO
            ├── main.py                     # Script principal
            ├── train.py                    # Entrenamiento YOLO
            ├── prediction.py               # Inferencia
            ├── options.py                  # Configuración
            ├── data_640.yaml               # Config para imágenes 640x640
            ├── data_1024.yaml              # Config para imágenes 1024x1024
            └── run.sh                      # Script de ejecución
```

## 🔧 Descripción de Módulos

### 📊 `src/data/` - Procesamiento de Datos

Este módulo contiene utilidades para preparar y validar los datos antes del entrenamiento:

- **`dataset_splitter.py`**: Divide el conjunto de datos en train/test/validation manteniendo la integridad de los pacientes (evita data leakage). Soporta imágenes en múltiples resoluciones (640x640 y 1024x1024) con sus correspondientes anotaciones YOLO.

- **`dataset_generator.py`**: Genera datasets en formatos específicos para cada modelo.

- **`data_validator.py`**: Valida la integridad de los datos, verifica que existan las anotaciones correspondientes y detecta inconsistencias.

### 🧠 `src/models/classification_data_clinic/` - Clasificación con Datos Clínicos

Entrena un modelo MLP (Perceptrón Multicapa) utilizando **únicamente datos clínicos** del paciente para predecir malignidad:

- **Características utilizadas:**
  - Edad de primera mamografía
  - IMC (Índice de Masa Corporal)
  - Estrato socioeconómico
  - Antecedentes médicos (tabaquismo, alcohol, TRH, etc.)
  - Historial familiar de cáncer de mama
  - Información reproductiva (menarquia, menopausia, primer parto)
  - Mutaciones BRCA
  
- **Preprocesamiento:**
  - Normalización Z-score para variables continuas
  - One-Hot Encoding para variables categóricas

### 🖼️ `src/models/classification_images/` - Clasificación con Imágenes

Entrena modelos de clasificación utilizando **únicamente imágenes** de mamografías:

- **Arquitecturas disponibles:**
  - ResNet50
  - DenseNet121
  - Inception v3
  
- **Características:**
  - Transfer learning con pesos pre-entrenados de ImageNet
  - Focal Loss para manejar desbalance de clases
  - Data augmentation durante entrenamiento
  - Early stopping basado en métricas de validación

### 🔀 `src/models/classification_final/` - Clasificación Híbrida

Modelo **multimodal** que combina información de imágenes y datos clínicos:

- **Arquitectura:**
  1. **Extractor de características de imagen**: ResNet50 pre-entrenado
  2. **Extractor de características clínicas**: MLP entrenado previamente (frozen)
  3. **Clasificador final**: MLP que fusiona ambas modalidades

- **Ventajas:**
  - Aprovecha información complementaria de ambas fuentes
  - Mejora la precisión y robustez de las predicciones
  - Permite interpretabilidad al separar contribuciones de cada modalidad

### 🎯 `src/models/detection/` - Detección de Lesiones

Implementación de **YOLO (You Only Look Once)** para detección de lesiones en mamografías:

- **Funcionalidades:**
  - Detección automática de regiones sospechosas
  - Entrenamiento con anotaciones en formato YOLO
  - Soporta múltiples resoluciones (640x640 y 1024x1024)
  - Genera bounding boxes con scores de confianza

- **Uso:**
  - Preprocessing para modelos de clasificación
  - Asistencia al radiólogo para localizar lesiones

### 📓 `notebooks/` - Análisis Exploratorio

- **`procesar-datos-clinicos.ipynb`**: Exploración, limpieza y análisis de datos clínicos. 

- **`dividir-conjunto-datos.ipynb`**: Visualización del proceso de división de datasets

### ⚙️ `scripts/` - Utilidades

- **`pipeline-inferencia.py`**: Pipeline completo de inferencia end-to-end que ejecuta todos los pasos desde el DICOM hasta la predicción final.

- **`preprocesar-datos.py`**: Scripts de preprocesamiento para convertir datos crudos a formatos utilizables por los modelos.

### 📦 `models/` - Arquitecturas

- **`model.py`**: Define la arquitectura del modelo final híbrido (`MLP_Final_Model`) que combina:
  - `ResNetModel`: Backbone CNN para extracción de características de imágenes
  - `MLP`: Redes neuronales densas para datos clínicos y clasificación final

## 🔄 Flujo de Trabajo Típico

### 1. Preparación de Datos
```bash
# Dividir dataset
python src/data/dataset_splitter.py

# Validar datos
python src/data/data_validator.py
```

### 2. Entrenamiento de Modelos

**a) Entrenar modelo de datos clínicos:**
```bash
cd src/models/classification_data_clinic
bash run.sh
```

**b) Entrenar modelo de detección (YOLO):**
```bash
cd src/models/detection
bash run.sh
```

**c) Entrenar modelo de clasificación de imágenes:**
```bash
cd src/models/classification_images
bash run.sh
```

**d) Entrenar modelo híbrido final:**
```bash
cd src/models/classification_final
bash run.sh
```

### 3. Inferencia en Nuevas Imágenes
```bash
python scripts/pipeline-inferencia.py \
    --dcm-path path/to/new/dicom.dcm \
    --output-dir results/ \
    --yolo-path-model models/yolo_best.pt \
    --model-path models/final_model_best.pth
```

## 📊 Métricas de Evaluación

Los modelos son evaluados utilizando:
- **Accuracy**: Precisión global
- **Sensitivity (Recall)**: Capacidad de detectar casos positivos
- **Specificity**: Capacidad de detectar casos negativos
- **F1-Score**: Media armónica de precisión y recall
- **AUC-ROC**: Área bajo la curva ROC
- **Matriz de Confusión**: Análisis detallado de predicciones

## 🛠️ Tecnologías Utilizadas

- **Python 3.10**
- **PyTorch**: Framework de Deep Learning
- **Ultralytics YOLO**: Detección de objetos
- **scikit-learn**: Preprocesamiento y métricas
- **scikit-image**: Procesamiento de imágenes
- **pydicom**: Lectura de archivos DICOM
- **pandas**: Manipulación de datos tabulares
- **NumPy**: Operaciones numéricas

---

## Advertencias
- El entrenamiento de los modelos se realizo utilizando el rastreador de experimentos Weights & Biases (wandb). Para reproducir los experimentos, es necesario configurar una cuenta en [Weights & Biases](https://wandb.ai/site) y ajustar las credenciales en los scripts correspondientes.


**Fecha de última actualización**: Diciembre 2025