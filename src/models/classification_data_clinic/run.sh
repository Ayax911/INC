
python3 main.py --exp_name <Nombre de experimento> \
                 --data_clinic_path <Ruta de los datos clinicos> \
                 --result_dir <Directorio de resultados> \
                 --tag_exp <Etiqueta de experimento para wandb> \
                 --input_size_clinic_model <Tamaño de entrada del modelo> \
                 --hidden_layers_clinic_model <Capas ocultas del modelo> \
                 --output_size_clinic_model <Cantidad de salidas del modelo> \
                 --activation_clinic_model <Función de activación del modelo> \
                 --dropout_clinic_model <Factor de dropout del modelo> \
                 --n_epochs <Número de épocas> \
                 --batch_size <Tamaño del lote> \
                 --lr <Tasa de aprendizaje> \
                 --patience_early <Patience para early stopping> \
                 --class_balance <Balanceo de clases> \
                 --neg_weight <Peso de la clase negativa> \
                 --pos_weight <Peso de la clase positiva> \
                 --train <Entrenamiento (True/False)> \