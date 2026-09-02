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
    data_clinic_path    = ""
    result_dir          = ""

    # Available options
    activation_choices      = ["Linear", "ReLU", "Sigmoid", "LeakyReLU", "Tanh"]
    loss_choices            = ["BCE", "Focal"]

    # Create argument parser with description
    parser = argparse.ArgumentParser(description="Código para entrenar y evaluar modelos para el proyecto cesm-synthesis")

    # Add arguments for general configurations
    parser.add_argument("--exp_name",           type=str, default="Test_Code", help="Nombre de experimento")
    parser.add_argument("--data_clinic_path",   type=str, default=data_clinic_path, help="Dirección del archivo de los datos clínicos")
    parser.add_argument("--result_dir",         type=str, default=result_dir, help="Direccion en donde se guardaran los resultados. Default = %(default)s")
    parser.add_argument("--tag_exp",            type=str, nargs='+', default=["Test"], help="Etiquetas de wandb para el experimento")
    parser.add_argument("--train",              help="Entrenamiento?", default=False, action="store_true")
    parser.add_argument("--test",               help="Prueba?", default=False, action="store_true")

    # Configuration of data clinic model
    parser.add_argument("--input_size_clinic_model",    type=int, default=106, help="Tamaño de datos de entrada del modelo clínico")
    parser.add_argument("--hidden_layers_clinic_model", type=int, nargs="+", default=[2048, ], help="Capas ocultas")
    parser.add_argument("--output_size_clinic_model",   type=int, default=1, help="Número de características de salida del modelo clínico")
    parser.add_argument("--activation_clinic_model",    type=str, default="LeakyReLU", choices=activation_choices, help="Función de activación del modelo clínico: %(choices)s")
    parser.add_argument("--dropout_clinic_model",       type=float, default=0.5, help="Tasa de abandono del modelo clínico")
    
    # Add arguments for training parameters
    parser.add_argument("--init_epoch",     type=int, default=0, help="Epoca desde donde se inicia el entrenamiento")
    parser.add_argument("--n_epochs",       type=int, default=1, help="Número de epocas de entrenamiento")
    parser.add_argument("--batch_size",     type=int, default=32, help="Numero de lotes por cada paso de entrenamiento")
    parser.add_argument("--lr",             type=float, default=1e-4, help="Taza de aprendizaje del modelo generador (Unet-GAN)")
    parser.add_argument("--b1",             type=float, default=0.5, help="Adam: decaimiento del impulso de primer orden del gradiente")
    parser.add_argument("--b2",             type=float, default=0.999, help="Adam: decaimiento del impulso de primer orden del gradiente")
    
    # Configuración EarlyStopping
    parser.add_argument("--patience_early", type=int, default=20, help="Patience para EarlyStopping")

    # Configuración de Scheduler
    parser.add_argument("--min_lr",         type=float, default=1e-6, help="Taza de aprendizaje mínima para el Scheduler")
    
    # Configuracion perdida
    parser.add_argument("--loss",           type=str, default="crossentropy", help="Función de perdida: %(choices)s")
    
    # Configuracion de balanceo de clases BCE
    parser.add_argument("--class_balance",  help="Utilizar balanceo de clases?", default=True, action="store_true")
    parser.add_argument("--pos_weight",     type=float, default=1.0, help="Peso para la clase positiva en la función de pérdida BCE")
    parser.add_argument("--neg_weight",     type=float, default=2.0, help="Peso para la clase negativa en la función de pérdida BCE")
    parser.add_argument("--gamma",          type=float, default=2.0, help="Gamma para la función de pérdida focal")

    # Parse command line arguments and return as a namespace
    args = parser.parse_args()

    return args
