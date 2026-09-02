from ultralytics import YOLO
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image


path_model = "yolo11_x_digitaleye.pt"
dir_test_images = "/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/05-INC/10-Preliminary-Version/data/processed/v2/imagenes_con_recorte/dataset_split/data_1024/images/test/"
dir_anotations = "/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/05-INC/10-Preliminary-Version/data/processed/v2/imagenes_anotadas_jpg"
dir_save  = "/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/05-INC/10-Preliminary-Version/data/processed/predictions_test_digitaleye_yolo11x/"

os.makedirs(dir_save, exist_ok=True)


def yolo_to_corners(x_center, y_center, width, height, img_width, img_height):
    """
    Convierte coordenadas YOLO normalizadas a coordenadas de esquinas en píxeles.
    
    Args:
        x_center, y_center, width, height: Coordenadas YOLO normalizadas (0-1)
        img_width, img_height: Dimensiones de la imagen en píxeles
    
    Returns:
        x1, y1, x2, y2: Coordenadas de esquinas en píxeles
    """
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    x1 = x_center_px - width_px / 2
    y1 = y_center_px - height_px / 2
    x2 = x_center_px + width_px / 2
    y2 = y_center_px + height_px / 2
    
    return x1, y1, x2, y2


def crear_visualizacion_comparativa(img_original, img_anotada, bbox_predicho, conf, 
                                    output_path, image_name):
    """
    Crea una visualización comparativa entre imagen anotada e imagen predicha.
    
    Args:
        img_original: Imagen original para predicción
        img_anotada: Imagen con anotaciones ground truth (JPG)
        bbox_predicho: Coordenadas YOLO del bbox predicho [x_center, y_center, width, height]
        conf: Confianza de la predicción
        output_path: Ruta donde guardar la imagen
        image_name: Nombre de la imagen
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Imagen izquierda: Ground Truth (anotación original)
    ax = axes[0]
    if len(img_anotada.shape) == 2:  # Escala de grises
        ax.imshow(img_anotada, cmap='gray')
    else:
        ax.imshow(img_anotada)
    ax.set_title(f'Ground Truth\n{image_name}', fontsize=14, fontweight='bold', color='blue')
    ax.axis('off')
    
    # Imagen derecha: Predicción
    ax = axes[1]
    if len(img_original.shape) == 2:  # Escala de grises
        ax.imshow(img_original, cmap='gray')
    else:
        ax.imshow(img_original)
    
    # Dibujar bbox predicho
    h, w = img_original.shape[:2]
    x_center, y_center, width, height = bbox_predicho
    x1, y1, x2, y2 = yolo_to_corners(x_center, y_center, width, height, w, h)
    
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    rect = patches.Rectangle(
        (x1, y1), bbox_width, bbox_height,
        linewidth=3, edgecolor='lime', facecolor='none'
    )
    ax.add_patch(rect)
        
    ax.set_title(f'Predicción', 
                fontsize=14, fontweight='bold', color='lime')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


os.makedirs(dir_save, exist_ok=True)

# Load a pretrained YOLO model
model = YOLO(path_model)

# Obtener la lista de imágenes de prueba
test_images = [f for f in os.listdir(dir_test_images) if f.endswith(('.jpg', '.png'))]

print(f"Procesando {len(test_images)} imágenes de test...")

# Procesar cada imagen de test
for idx, image_name in enumerate(test_images, 1):
    print(f"\n[{idx}/{len(test_images)}] Procesando: {image_name}")
    
    path_image_test = os.path.join(dir_test_images, image_name)
    path_annotation = os.path.join(dir_anotations, image_name)
    path_save = os.path.join(dir_save, f"{Path(image_name).stem}_comparison.jpg")
    
    # Verificar si existe la anotación
    if not os.path.exists(path_annotation):
        print(f"  ⚠ Anotación no encontrada: {path_annotation}")
        continue
    
    # Cargar imagen original (para predicción) usando PIL y convertir a RGB
    img_original = np.array(Image.open(path_image_test).convert('RGB'))
    
    # Cargar imagen anotada (ground truth) usando PIL y convertir a RGB
    img_anotada = np.array(Image.open(path_annotation).convert('RGB'))
    
    # Realizar predicción
    results = model.predict(
        source=path_image_test,
        conf=0.01,
        iou=0.5,
        imgsz=1024,
        device="cuda:0",
        save=False,
        verbose=False
    )
    
    # Extraer boxes
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        print(f"  ⚠ No se detectaron objetos en {image_name}")
        continue
    
    # Obtener la caja con mayor confianza
    confs = boxes.conf.cpu().numpy()
    idx_max = confs.argmax()
    best_box = boxes[idx_max]
    
    # Extraer coordenadas YOLO normalizadas
    x_center, y_center, width, height = best_box.xywhn[0].cpu().numpy()
    conf = float(confs[idx_max])
    
    print(f"  ✓ Predicción - Conf: {conf:.3f}, YOLO: [{x_center:.4f}, {y_center:.4f}, {width:.4f}, {height:.4f}]")
    
    # Crear visualización comparativa
    crear_visualizacion_comparativa(
        img_original=img_original,
        img_anotada=img_anotada,
        bbox_predicho=[x_center, y_center, width, height],
        conf=conf,
        output_path=path_save,
        image_name=image_name
    )
    
    print(f"  ✓ Visualización guardada: {path_save}")

print(f"\n✅ Procesamiento completado. {len(test_images)} imágenes procesadas.")
print(f"📁 Resultados guardados en: {dir_save}")
