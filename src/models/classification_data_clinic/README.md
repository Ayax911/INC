# Clasificación binaria con datos clínicos (INC)

Este repositorio entrena y evalúa un modelo MLP (Multi-Layer Perceptron) para clasificación binaria usando variables clínicas tabulares almacenadas en CSV.


## Estructura del repositorio

```bash
classification_data_clinic/
├─ main.py                       # Punto de entrada (train / test)
├─ options.py                    # Argumentos CLI
├─ training.py                   # Entrenamiento, validación y test
├─ losses.py                     # Funciones de pérdida (actualmente: BCE ponderada via CrossEntropy)
├─ metrics.py                    # Métricas de evaluación
├─ early_stopping.py             # Early stopping (guarda Best_Model.pth)
├─ dataloaders/
│  └─ data_loader_data.py        # Loader que espera 3 CSV con nombres fijos
└─ models/
   ├─ get_model.py               # Crea el MLP según options
   └─ mlp_models.py              # Definición de MLP
```


## Preparación de los datos
El loader no recibe un CSV individual, sino un directorio que debe contener exactamente estos archivos (nombres fijos):

```bash
<data_clinic_path>/
├─ train_clinical_data_processed.csv
├─ val_clinical_data_processed.csv
└─ test_clinical_data_processed.csv
```

Formato esperado de cada CSV
Cada CSV debe incluir al menos estas columnas:

- ID: identificador (se descarta en el modelo).
- Diagnostico: etiqueta binaria (0 = negativo, 1 = positivo).
- El resto de columnas deben ser features numéricas (float/int). El modelo usa:


**Reglas importantes**
1. Las columnas (features) deben ser idénticas en train/val/test, en el mismo orden.
2. No debe haber NaN. Imputa o elimina antes.
3. Si tienes variables categóricas, conviértelas a numéricas (one-hot, ordinal, etc.) antes.
4. El parámetro --input_size_clinic_model debe coincidir con el número de features:


## Cómo entrenar

1) Entrenamiento + evaluación al final
main.py hace esto cuando pasas --train:

Ejemplo completo:

```bash
python3 main.py --exp_name MLP_clinic_v1 \
                --data_clinic_path /ruta/a/tu/directorio_de_csv \
                --result_dir /ruta/a/resultados \
                --tag_exp INC clinic baseline \
                --input_size_clinic_model 106 \
                --hidden_layers_clinic_model 2048 512 128 \
                --output_size_clinic_model 2 \
                --activation_clinic_model LeakyReLU \
                --dropout_clinic_model 0.5 \
                --n_epochs 100 \
                --batch_size 32 \
                --lr 1e-4 \
                --min_lr 1e-6 \
                --patience_early 20 \
                --loss bce \
                --neg_weight 2.0 \
                --pos_weight 1.0 \
                --train
```

### Parámetros principales
- **exp_name:** nombre del experimento (carpeta dentro de result_dir).
- **data_clinic_path:** directorio que contiene los 3 CSV.
- **result_dir:** carpeta donde se almacena los resultados.
- **tag_exp:** Etiquetas asociadas al experimento (usadas típicamente en wandb) para facilitar la organización y filtrado de ejecuciones.
- **train:** Si se especifica, activa el modo entrenamiento del modelo.
- **test:** Si se especifica, activa el modo evaluación/prueba del modelo entrenado.
- **input_size_clinic_model:** Número de características de entrada del modelo clínico (dimensión del vector clínico).
- **hidden_layers_clinic_model:** lista de tamaños de capas ocultas.
- **output_size_clinic_model:** Número de neuronas de salida del modelo clínico.
- **activation_clinic_model:** Función de activación utilizada en las capas ocultas del modelo clínico.
- **dropout_clinic_model:** Tasa de dropout aplicada en el modelo clínico para reducir sobreajuste.
- **init_epoch:** Época inicial del entrenamiento. Útil para reanudar entrenamientos desde un checkpoint.
- **n_epochs:** Número total de épocas de entrenamiento.
- **batch_size:** Tamaño del lote utilizado durante el entrenamiento.
- **lr:** Tasa de aprendizaje inicial del optimizador.
- **b1:** Parámetro β₁ del optimizador Adam (primer momento).
- **b2:** Parámetro β₂ del optimizador Adam (primer momento).
- **patience_early:** Número de épocas sin mejora antes de detener el entrenamiento mediante EarlyStopping.
- **min_lr:** Tasa de aprendizaje mínima permitida por el scheduler.


