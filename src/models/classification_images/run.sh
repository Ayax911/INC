path_image_model_resnet="/src/models/classification_images/models/pretrained_models/ResNet50.pt"

result_dir="/ruta/de/resultados/experimento"

python3 main.py --exp_name "Classification Images" \
                --images_dir "/ruta/de/imagenes" \
                --csv_data_path "ruta/de/csv" \
                --result_dir $result_dir \
                --tag_exp "Classification Images" \
                --activation_image_model "Gelu" \
                --image_model "ResNet" \
                --path_image_model $path_image_model_resnet \
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

