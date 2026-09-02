import argparse

def parse_tuple(input_string):
    try:
        return tuple(map(int, input_string.split(',')))  # Convierte los valores a int
    except:
        raise argparse.ArgumentTypeError("Las entradas deben ser números enteros separados por comas.")

def get_options():
    """
    Function to parse command line arguments and return them as a namespace.

    Returns:
        argparse.Namespace: parsed command line arguments
    """
    
    # Default directory paths
    images_dir          = "/home/kevin-osorno-castillo/Documentos/parches_con_datos_clinicos/imagenes_npy"
    csv_data_path       = "/home/kevin-osorno-castillo/Documentos/parches_con_datos_clinicos"
    result_dir          = "results"
    path_model          = "models/pretrained_models/ResNet50.pt"  # Path to the weights file if needed #"/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/03-Challenges/01-MAMA-MIA/01-Code/RadImageNet_pytorch/01-Pytorch/ResNet50.pt"
    clinic_path_model   = "models/pretrained_models/data_clinic.pth"
    
    # Available options
    images_model_choices    = ["Inception", "ResNet", "DenseNet"]
    activation_choices      = ["Linear", "ReLU", "Sigmoid", "LeakyReLU", "Tanh", "Gelu"]
    model_clinic = [2048, 2048, 1024, 1024, 512, 256, 128, 64, 32]

    # Create argument parser with description
    parser = argparse.ArgumentParser(description="Código para entrenar y evaluar modelos para el proyecto cesm-synthesis")

    # Add arguments for general configurations
    parser.add_argument("--exp_name",           type=str, default="Test_Code", help="Nombre de experimento")
    parser.add_argument("--images_dir",         type=str, default=images_dir, help="Dirección del directorio de las imágenes")
    parser.add_argument("--csv_data_path",      type=str, default=csv_data_path, help="Ruta de los archivos csv con la estructuracion de la base de datos")
    parser.add_argument("--result_dir",         type=str, default=result_dir, help="Direccion en donde se guardaran los resultados. Default = %(default)s")
    parser.add_argument("--tag_exp",            type=str, nargs='+', default=["Test"], help="Etiquetas de wandb para el experimento")
    parser.add_argument("--train",              help="Entrenamiento?", default=True, action="store_true")
    parser.add_argument("--test",               help="Prueba?", default=False, action="store_true")

    # Configuration of images model
    parser.add_argument("--activation_image_model", type=str, default="ReLU", choices=activation_choices, help="Función de activación interna del modelo de imagen: %(choices)s")
    parser.add_argument("--image_model",            type=str, default="ResNet", choices=images_model_choices, help="Seleccione el modelo generador a utilizar: %(choices)s")
    parser.add_argument("--path_image_model",       type=str, default=path_model, help="Ruta al archivo de pesos del modelo preentrenado")
    parser.add_argument("--path_clinic_model",      type=str, default=clinic_path_model, help="Ruta al archivo de pesos del modelo clínico")
    
    # Configuration of final model
    parser.add_argument("--final_input_size",                 type=int, default=4096, help="Tamaño de entrada del modelo clínico")
    parser.add_argument("--final_hidden_layers",              type=int, nargs="+", default=[2048, 1024, 256, 128], help="Capas ocultas del modelo final")
    parser.add_argument("--final_output_size",                type=int, default=2, help="Número de clases de salida del modelo final")
    parser.add_argument("--final_activation",                 type=str, default="Gelu", choices=activation_choices, help="Función de activación del modelo final: %(choices)s")
    parser.add_argument("--final_dropout",                    type=float, default=0.5, help="Tasa de abandono del modelo final")
    
    # Configuration of clinical model
    parser.add_argument("--clinic_input_size",              type=int, default=51, help="Tamaño de entrada del modelo clínico")
    parser.add_argument("--clinic_hidden_layers",           type=int, nargs="+", default=model_clinic, help="Capas ocultas del modelo clínico")
    parser.add_argument("--clinic_activation",              type=str, default="ReLU", choices=activation_choices, help="Función de activación interna del modelo clínico: %(choices)s")
    parser.add_argument("--clinic_dropout",                 type=float, default=0.5, help="Tasa de abandono del modelo clínico")
    parser.add_argument("--clinic_idx_hidden_layer",        type=int, default= 2 , help="Índice de la capa oculta del modelo clínico para extraer las características")
    
    # Add arguments for data loader configuration
    parser.add_argument("--channels",       type=int, default=3, help="Número de canales de la imagen")
    parser.add_argument("--augmentation",   help="Utilizar aumento de datos en el entrenamiento?", default=False, action="store_true")
    parser.add_argument('--img_size',       type=parse_tuple, help='Dimension de las imagenes de entrada en formato (height, width)', default=(224, 224))

    # Add arguments for training parameters
    parser.add_argument("--init_epoch",     type=int, default=0, help="Epoca desde donde se inicia el entrenamiento")
    parser.add_argument("--n_epochs",       type=int, default=2, help="Número de epocas de entrenamiento")
    parser.add_argument("--batch_size",     type=int, default=32, help="Numero de lotes por cada paso de entrenamiento")
    parser.add_argument("--lr",             type=float, default=1e-4, help="Taza de aprendizaje del modelo generador (Unet-GAN)")
    parser.add_argument("--b1",             type=float, default=0.5, help="Adam: decaimiento del impulso de primer orden del gradiente")
    parser.add_argument("--b2",             type=float, default=0.999, help="Adam: decaimiento del impulso de primer orden del gradiente")
    
    # Configuración EarlyStopping
    parser.add_argument("--patience_early", type=int, default=20, help="Patience para EarlyStopping")

    # Configuración de Scheduler
    parser.add_argument("--min_lr",         type=float, default=1e-6, help="Taza de aprendizaje mínima para el Scheduler")
    
    # Configuracion perdida
    parser.add_argument("--loss",           type=str, default="BCE", help="Función de perdida: %(choices)s")
    
    # Configuracion de balanceo de clases BCE
    parser.add_argument("--class_balance",  help="Utilizar balanceo de clases?", default=True, action="store_true")
    parser.add_argument("--pos_weight",     type=float, default=1.0, help="Peso para la clase positiva en la función de pérdida BCE")
    parser.add_argument("--neg_weight",     type=float, default=2.0, help="Peso para la clase negativa en la función de pérdida BCE")
    parser.add_argument("--gamma",          type=float, default=2.0, help="Gamma para la función de pérdida focal")

    # Parse command line arguments and return as a namespace
    args = parser.parse_args()

    return args
