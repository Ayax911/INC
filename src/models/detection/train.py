from ultralytics import YOLO
import wandb

def train(options):
    
    # Inicializar el modelo YOLO
    model = YOLO(options.model_path)
    
    # Inicializar Weights & Biases
    wandb.init(project="mammo-yolov11", name=options.experiment_name, config=vars(options))
    
    # Iniciar el entrenamiento del modelo con los parámetros especificados
    model.train(
        data            = options.path_data,                # Ruta del archivo .yaml de datos
        imgsz           = options.img_size,                 # Tamaño de las imágenes
        batch           = options.batch_size,               # Tamaño del batch
        rect            = False,                     # Entrenamiento rectangular (automaticamente ajusta el tamaño con padding)
        mosaic          = 0.0,                              # Desactivar mosaic augmentation   
        device          = options.device,                   # Dispositivo de entrenamiento (CPU o GPU)
        epochs          = options.epochs,                   # Número de épocas 
        pretrained      = options.pretrained,               # Usar pesos preentrenados
        save            = True,                             # Guardar el modelo al final del entrenamiento
        save_period     = options.save_period,              # Periodo para guardar el modelo
        project         = options.path_project,             # Ruta del proyecto
        name            = options.experiment_name,          # Nombre del experimento
        exist_ok        = True,                             # Sobrescribir si el experimento ya existe
        optimizer       = options.optimizer,                # Optimizador a utilizar
        seed            = options.seed,                     # Semilla para reproducibilidad
        deterministic   = False,                            # Modo determinista
        single_cls      = options.single_cls,               # Considerar una sola clase, ignorando las etiquetas de clase (solo detección)
        cos_lr          = options.cosine_lr,                # Usar decaimiento de tasa de aprendizaje cosenoidal
        freeze          = options.freeze_layers,            # Capas a congelar durante el entrenamiento
        lr0             = options.learning_rate,            # Tasa de aprendizaje inicial
        lrf             = 0.001,                            # Tasa de aprendizaje final (como fracción de lr0)
        weight_decay    = 0.01,                           # Decaimiento de peso (L2 regularization)
        cls             = options.classification_weight,    # Peso para la clasificación
        box             = options.box_weight,               # Peso para la localización de cajas
        dropout         = options.dropout_rate,             # Tasa de dropout
        plots           = True,                             # Generar gráficos durante el entrenamiento
        mixup           = 0.0,
        copy_paste      = 0.0,
        flipud         = 0.0,
        fliplr         = 0.5,    # opcional
        hsv_h          = 0.0,
        hsv_s          = 0.0,
        hsv_v          = 0.1,
        scale          = 0.1,
        translate      = 0.05,
        perspective    = 0.0,
        shear          = 0.0,
        erasing        = 0.0,
        visualize       = True,
        patience        = 1000,
        multi_scale      = True,
        dfl             = 2.0

        
    )

if __name__ == "__main__":
    train(options=None)