python3 main.py     --path_data data_1024.yaml \
                    --img_size 1024 \
                    --batch_size 18 \
                    --pretrained \
                    --epochs 300 \
                    --device 0 \
                    --path_project /media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/05-INC/10-Preliminary-Version/results/detection_experiments \
                    --freeze_layers 8 \
                    --box_weight 4 \
                    --classification_weight 0.5 \
                    --experiment_name exp8_d1024-auto-bs-18_epochs-300_freeze-8_boxWeight-4_classificationWeight-0.5_DigitalEye_yoloL \
                    --optimizer auto \
                    --seed 42 \
                    --learning_rate 0.001 \
                    --model_path yolo11_m_digitaleye.pt \
                    --dropout 0.0 \




