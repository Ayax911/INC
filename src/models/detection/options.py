import argparse

def get_options():
    """
    Function to parse command line arguments and return them as a namespace.

    Returns:
        argparse.Namespace: parsed command line arguments
    """
    # Default directory paths
    data_dir_default            = "src/models/detection/data.yaml"
    project_dir_default         = "results/detection/"
    img_size_default            = 640
    batch_size_default          = 16
    epochs_default              = 100
    save_period_default         = 20
    experiment_name_default     = "exp"
    optimizer_default           = "auto"
    seed_default                = 42
    learning_rate_default       = 0.01
    classification_weight_default = 0.0
    box_weight_default          = 7.5
    dropouts_rate_default       = 0.0
    model_path_default          = "yolov8n.pt"
    
    # Available options
    img_size_choices = [640, 512, 1024]
    optimizer_choices = ["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"]

    # Create argument parser with description
    parser = argparse.ArgumentParser(description="Código para entrenar y evaluar modelos para el proyecto INC Mammography")

    # Add arguments for general configurations
    parser.add_argument("--path_data",          type=str, default=data_dir_default, help="Ruta del archivo .yaml de datos. Default = %(default)s")
    parser.add_argument("--img_size",           type=int, default=img_size_default, choices=img_size_choices, help="Tamaño de las imágenes. Default = %(default)s, Choices = %(choices)s")
    parser.add_argument("--batch_size",         type=int, default=batch_size_default, help="Tamaño del lote para el entrenamiento. Default = %(default)s")
    parser.add_argument("--rect",               help="Entrenamiento rectangular?", default=False, action="store_true")
    parser.add_argument("--device",             type=str, default="cuda:0", help="Dispositivo de entrenamiento. Default = %(default)s")
    parser.add_argument("--epochs",             type=int, default=epochs_default, help="Número de,epocas. Default = %(default)s")
    parser.add_argument("--pretrained",         help="Usar pesos preentrenados?", default=False, action="store_true")
    parser.add_argument("--save_period",        type=int, default=save_period_default, help="Periodo para guardar el modelo. Default = %(default)s")
    parser.add_argument("--path_project",       type=str, default=project_dir_default, help="Ruta del proyecto. Default = %(default)s")
    parser.add_argument("--experiment_name",    type=str, default=experiment_name_default, help="Nombre del experimento. Default = %(default)s")
    parser.add_argument("--optimizer",          type=str, default=optimizer_default, choices=optimizer_choices, help="Optimizador a utilizar. Default = %(default)s, Choices = %(choices)s")
    parser.add_argument("--seed",               type=int, default=seed_default, help="Semilla para la reproducibilidad. Default = %(default)s")
    parser.add_argument("--deterministic",      help="Modo determinista?", default=False, action="store_true")
    parser.add_argument("--single_cls",         help="Considerar una sola clase (solo detección)?", default=False, action="store_true")
    parser.add_argument("--cosine_lr",          help="Usar decaimiento de tasa de aprendizaje cosenoidal?", default=True, action="store_true")
    parser.add_argument("--freeze_layers",      type=int, help="Capas a congelar durante el entrenamiento. Default = %(default)s")
    parser.add_argument("--learning_rate",      type=float, default=learning_rate_default, help="Tasa de aprendizaje inicial. Default = %(default)s")
    parser.add_argument("--classification_weight", type=float, default=classification_weight_default, help="Peso para la clasificación. Default = %(default)s")
    parser.add_argument("--box_weight",         type=float, default=box_weight_default, help="Peso para la localización de cajas. Default = %(default)s")
    parser.add_argument("--dropout_rate",       type=float, default=dropouts_rate_default, help="Tasa de dropout. Default = %(default)s")
    parser.add_argument("--model_path",         type=str, default=model_path_default, help="Ruta del modelo YOLO preentrenado. Default = %(default)s")

    # Parse command line arguments and return as a namespace
    args = parser.parse_args()

    return args