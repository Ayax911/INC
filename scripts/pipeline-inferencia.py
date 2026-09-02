from skimage.morphology import remove_small_objects, binary_closing, disk
from pydicom.pixel_data_handlers.util import apply_windowing
from skimage.exposure import rescale_intensity
from scipy.ndimage import binary_fill_holes
from torchvision.models import resnet50
from skimage.transform import resize
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from pathlib import Path
from torch import nn
import pandas as pd
import numpy as np
import argparse
import pydicom
import torch
import sys

# Agregar el directorio raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.model import MLP_Final_Model


class PipelineInferencia:
    def __init__(self, dcm_path: str, output_dir: str, yolo_path_model: str, model_path: str):
        self.dcm_path = dcm_path
        self.output_dir = Path(output_dir)

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        # Cargar modelo de detección
        self.modelo_deteccion = YOLO(yolo_path_model)
        self.modelo_deteccion.to(self.device)                                                                                             # Enviar el modelo al dispositivo (CPU o GPU)
        self.modelo_deteccion.eval()

        # Cargar final
        self.model = MLP_Final_Model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)                                                                                                     # Enviar el modelo al dispositivo (CPU o GPU)
        self.model.eval()

    
    def ejecutar_pipeline(self):
        """
        Ejecuta el pipeline de inferencia
        """
        # 1. Leer imagen DICOM y aplicar ventaneo
        imagen = self._leer_imagen_dicom(self.dcm_path)

        # 2. Recortar la mama de la imagen
        imagen_recortada, info_recorte = self._recortar_mama(imagen)

        # 3. Agregar padding para hacer la imagen cuadrada
        tamaño_objetivo = max(imagen_recortada.shape)
        imagen_cuadrada, info_padding = self._agregar_padding_cuadrado(imagen_recortada, tamaño_objetivo)

        # 4. Obtener imagen redimensionada a 1024x1024
        imagen_1024 = self._obtener_imagen_1024(imagen_cuadrada)

        # 5. Guardar imágenes con ventaneo
        directorio_1024 = self.output_dir / "imagenes_1024"
        directorio_1024.mkdir(parents=True, exist_ok=True)
        nombre_archivo = Path(self.dcm_path).stem
        ruta_jpg = self._guardar_imagen_procesada_para_modelo(
            nombre_archivo, 
            imagen_1024, 
            directorio_1024
        )

        # 6. Ejecutar el modelo de inferencia (Deteccion)
        detection_result = self._modelo_inferencia_deteccion(ruta_jpg)
        
        # 6.1. Guardar imagen con bounding boxes visualizadas
        if detection_result and len(detection_result) > 0 and len(detection_result[0]) > 0:
            # detection_result es una lista de listas, tomamos la primera
            self._guardar_imagen_con_bounding_boxes(
                ruta_jpg, 
                detection_result[0],  # Tomar la primera lista (image_predictions)
                nombre_archivo
            )

        # 8. Recortar region detectada de la imagen ORIGINAL
        parches_imagen = []
        for idx, (_, conf, (x_center, y_center, width, height)) in enumerate(detection_result[0]):
            img_h_original, img_w_original = imagen_cuadrada.shape
            
            scale_w = img_w_original / 1024
            scale_h = img_h_original / 1024
            
            x1, y1, x2, y2 = self.yolo_to_corners(
                x_center, y_center, width, height, 
                img_w_original, img_h_original
            )
            
            # Recortar de la imagen original (con máxima resolución)
            parche = self._recortar_regiones_anotadas(x1, y1, x2, y2, imagen_cuadrada)
            parches_imagen.append(parche)

        # 9. Procesar cada parche detectado
        parches_procesados = []
        for parche in parches_imagen:
            parche_procesado = self._procesar_parche(parche)
            parches_procesados.append(parche_procesado)
        
        # 10. Procesar los datos clinicos del paciente
        # 10.1 Simular formulario de datos clinicos 
        datos_clinicos = {
            "Edad primera mamografia": 45,
            "IMC": 22.5,
            "Estrato socieconomico": 3, # 1,2,3,4
            "Residencia": "Urbana", # Opciones: 'Rural', 'Urbana'
            "Tipo de afiliacion": "Contributivo",
            "Escolaridad": "Universitaria",
            "Antecedente de Tabaquismo o tabaquismo activo": 0,
            "Antecedente consumo de alcohol": 1,
            "Antecedente de CDIS (carcinoma ductal in situ)": 0,
            "Antecedente de CDI (Carcinoma Ductal Infiltrante)": 0,
            "CLI (Carcinoma Lobulillar Infiltrante)": 0,
            "Antecedente de TRH": 1,
            "Antecedente de Bx mama benigna": 0,
            "Tipo de afiliacion": "Contributivo", # Opciones: 'Contributivo', 'Especial', 'Subsidiado'
            "Escolaridad": "Universitaria",    # Opciones: 'Ninguno', 'Primaria', 'Bachillerato', 'Tecnico', 'Universitario', 'Posgrado', 'Desconocido'
            "Edad Menarquia": "14 o más años", # Opciones: '12 a 13 años', '14 o más años', '7 a 11 años', 'Desconocido'
            "menopausia tardia (mayor a 55 años)": "No", # Opciones: 'Desconocido', 'No', 'Si'
            "Edad 1er parto": "25-29 años",    # Opciones: '20-24 años', '25-29 años', '< 20 años', '> 30 años', 'Desconocido', 'No embarazos'
            "Antecedente de mutacion BRCA": "No", # Opciones: 'Desconocido', 'No', 'Si'
            "# Bx realizadas": "1",             # Opciones: '1', '2 ó más', 'Desconocido'
            " Familiares en 1er grado con Ca de mama-": "Ninguna", # Opciones: '1', 'Desconocido', 'Más de 1', 'Ninguna'
            "Lateralidad del hallazgo": "Derecho" # Opciones: 'Bilateral', 'Derecho', 'Izquierdo'
        }
        # 10.2 Convertir a DataFrame
        datos_clinicos_df = pd.DataFrame([datos_clinicos])
        
        # 10.3 Aplicar One-Hot Encoding a columnas categóricas
        datos_clinicos_procesados = self._procesar_datos_clinicos(datos_clinicos_df)

        # 11. Realizar la inferencia para cada parche con los datos clinicos
        prediciones = []
        for i, parche_procesado in enumerate(parches_procesados):
            # Convertir parche a tensor
            parche_tensor = torch.from_numpy(parche_procesado).float()
            parche_tensor = parche_tensor.permute(2, 0, 1).unsqueeze(0) 
            parche_tensor = parche_tensor.to(self.device)

            # Convertir datos clinicos a tensor
            datos_clinicos_tensor = torch.from_numpy(datos_clinicos_procesados.values).float()
            datos_clinicos_tensor = datos_clinicos_tensor.to(self.device)

            # Realizar inferencia
            with torch.no_grad():
                predictions     = self.model(parche_tensor, datos_clinicos_tensor)
                probabilities   = nn.Softmax(dim=1)(predictions)
                prediciones.append((predictions.cpu().numpy(), probabilities.cpu().numpy()))
                print(f"Parche {i+1}/{len(parches_procesados)}.  La lesion tiene {probabilities[0,1]*100:.2f}% de probabilidad de ser maligna")


    def _procesar_datos_clinicos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Procesa los datos clínicos aplicando:
        1. Normalización Z-score a columnas numéricas continuas
        2. One-Hot Encoding a las columnas categóricas
        
        Se definen todas las categorías posibles para garantizar que siempre se generen
        las mismas columnas, independientemente del valor del paciente individual.
        
        Args:
            df: DataFrame con los datos clínicos sin procesar
            
        Returns:
            DataFrame con las columnas procesadas y codificadas
        """
        df_encoded = df.copy()
        
        # 1. Normalización Z-score para columnas numéricas continuas
        normalizacion_params = {
            'Edad primera mamografia': {'mean': 59.22, 'std': 11.93},  
            'IMC': {'mean': 28.27, 'std': 4.23} 
        }
        
        for columna, params in normalizacion_params.items():
            if columna in df_encoded.columns:
                # Z-score: (x - mean) / std
                df_encoded[columna] = (df_encoded[columna] - params['mean']) / params['std']
        
        # 2. Definir todas las categorías posibles para cada columna categórica
        categorias_definidas = {
            'Estrato socieconomico': [1, 2, 3, 4],
            'Residencia': ['Rural', 'Urbana'],
            'Tipo de afiliacion': ['Contributivo', 'Especial', 'Subsidiado'],
            'Escolaridad': ['Ninguno', 'Primaria', 'Bachillerato', 'Tecnico', 'Universitaria', 'Posgrado', 'Desconocido'],
            'Edad Menarquia': ['7 a 11 años', '12 a 13 años', '14 o más años', 'Desconocido'],
            'menopausia tardia (mayor a 55 años)': ['No', 'Si', 'Desconocido'],
            'Edad 1er parto': ['< 20 años', '20-24 años', '25-29 años', '> 30 años', 'No embarazos', 'Desconocido'],
            'Antecedente de mutacion BRCA': ['No', 'Si', 'Desconocido'],
            '# Bx realizadas': ['1', '2 ó más', 'Desconocido'],
            ' Familiares en 1er grado con Ca de mama-': ['Ninguna', '1', 'Más de 1', 'Desconocido'],
            'Lateralidad del hallazgo': ['Bilateral', 'Derecho', 'Izquierdo']
        }
        
        # 3. Aplicar One-Hot Encoding para cada columna categórica
        for columna, categorias in categorias_definidas.items():
            if columna in df_encoded.columns:
                # Crear columnas dummy para todas las categorías posibles
                for categoria in categorias:
                    nombre_columna = f"{columna}_{categoria}"
                    # Asignar 1 si coincide con el valor actual, 0 si no
                    df_encoded[nombre_columna] = (df_encoded[columna] == categoria).astype(int)
                
                # Eliminar la columna original
                df_encoded = df_encoded.drop(columns=[columna])
        
        return df_encoded

    def _procesar_parche(self, parche: np.ndarray):
        """
        Procesa el parche recortado (por ejemplo, redimensionar, normalizar, etc.)

        Args:
            parche: Array numpy con el parche recortado

        Returns:
            parche_procesado: Parche procesado listo para el siguiente modelo
        """
        # Redimensionar parche a 224x224
        parche_redimensionado = resize(parche, (224, 224), order=1, preserve_range=True)

        # Normalizar el parche individualmente para mejor contraste
        parche_normalizado = self._normalizar_imagen(parche_redimensionado)

        # Apilar la imagen 3 veces para simular 3 canales (mantiene valores originales)
        parche_normalizado = np.stack([parche_normalizado] * 3, axis=-1)

        return parche_normalizado
    
    def _normalizar_imagen(self, imagen: np.ndarray) -> np.ndarray:
        """
        Normaliza la imagen entre -1 y 1 usando rescale_intensity con range in [0, 4095].
        
        Args:
            imagen: Array numpy con la imagen
            
        Returns:
            Array numpy con la imagen normalizada
        """
        imagen_normalizada = rescale_intensity(
                imagen,
                in_range=(0, 4095),
                out_range=(-1,1)
            ).astype(np.float32)
        return imagen_normalizada

    def _recortar_regiones_anotadas(self, x1, y1, x2, y2, imagen):
        # Asegurar que x1 < x2 y y1 < y2
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Convertir a enteros para usar como índices
        x_min, x_max = int(x_min), int(x_max)
        y_min, y_max = int(y_min), int(y_max)

        # Recortar región
        parche = imagen[y_min:y_max, x_min:x_max]
        return parche


    def yolo_to_corners(self, x_center, y_center, width, height, img_width, img_height):
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

    def _modelo_inferencia_deteccion(self, image_path: Path):
        """
        Ejecuta el modelo de detección en la imagen dada.

        Args:
            image_path: Path de la imagen en formato JPG

        Returns:
            results: Resultados de la inferencia del modelo
        """
        predictions_all = []

        results = self.modelo_deteccion.predict(
            source=str(image_path),
            conf=0.25,
            iou=0.5,
            imgsz=1024,
            device="cpu",
            save=False,
            verbose=False
        )

        # Extraer boxes
        boxes = results[0].boxes
        image_predictions = []
        if len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            for i, box in enumerate(boxes):
                x_center, y_center, width, height = box.xywhn[0].cpu().numpy()
                w, h = 1024, 1024
                x1, y1, x2, y2 = self.yolo_to_corners(x_center, y_center, width, height, w, h)
                image_predictions.append(((x1, y1, x2, y2), float(confs[i]), (x_center, y_center, width, height)))

        predictions_all.append(image_predictions)
        return predictions_all
    
    def _guardar_imagen_con_bounding_boxes(self, imagen_path: Path, predictions: list, nombre_archivo: str):
        """
        Dibuja las bounding boxes predichas por YOLO sobre la imagen y la guarda.
        
        Args:
            imagen_path: Ruta de la imagen original
            predictions: Lista de predicciones [(x1, y1, x2, y2), confidence]
            nombre_archivo: Nombre base del archivo para guardar
        
        Returns:
            Path: Ruta donde se guardó la imagen con las bounding boxes
        """
        # Cargar la imagen
        imagen = Image.open(imagen_path)
        
        # Convertir a RGB si está en escala de grises
        if imagen.mode != 'RGB':
            imagen = imagen.convert('RGB')
        
        draw = ImageDraw.Draw(imagen)
        
        # Intentar cargar una fuente, si falla usar la predeterminada
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Dibujar cada bounding box
        for (x1, y1, x2, y2), confidence, _ in predictions:
            # Convertir coordenadas a enteros
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Dibujar rectángulo (bounding box)
            # Color verde (R, G, B) con grosor de 3 píxeles
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            
        # Guardar imagen con bounding boxes
        directorio_detecciones = self.output_dir / "detecciones_visualizadas"
        directorio_detecciones.mkdir(parents=True, exist_ok=True)
        ruta_salida = directorio_detecciones / f"{nombre_archivo}_detecciones.jpg"
        imagen.save(ruta_salida, format="JPEG", quality=95)
        
        print(f"Imagen con bounding boxes guardada en: {ruta_salida}")
        return ruta_salida
        
        
    def _guardar_imagen_procesada_para_modelo(
        self, 
        nombre: str, 
        data_1024: np.ndarray, 
        directorio_1024: Path,
        ):
        """
        Guarda las imágenes con ventaneo en formato JPG.
        
        Args:
            nombre: Nombre del archivo
            data_1024: Imagen de 1024 píxeles
            data_640: Imagen de 640 píxeles
        Return:
            Path de las imágenes guardadas
        """
        ruta_jpg_1024 = directorio_1024 / f"{nombre}.jpg"        
        Image.fromarray(data_1024).save(ruta_jpg_1024, format="JPEG", quality=95)
        return ruta_jpg_1024


    def _agregar_padding_cuadrado(self, imagen, tamaño_objetivo):
        """
        Agrega padding negro para convertir la imagen en cuadrada.
        
        Args:
            imagen: Array numpy con la imagen (alto x ancho)
            tamaño_objetivo: Tamaño del lado del cuadrado deseado
            
        Returns:
            imagen_con_padding: Imagen cuadrada con padding
            info_padding: Dict con información del padding aplicado
        """
        alto_actual, ancho_actual = imagen.shape[:2]
        
        # Calcular padding necesario
        pad_alto = tamaño_objetivo - alto_actual
        pad_ancho = tamaño_objetivo - ancho_actual
        
        # Distribuir padding equitativamente en ambos lados
        pad_top = pad_alto // 2
        pad_bottom = pad_alto - pad_top
        pad_left = pad_ancho // 2
        pad_right = pad_ancho - pad_left
        
        # Aplicar padding
        if len(imagen.shape) == 2: 
            imagen_con_padding = np.pad(
                imagen,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=0
            )
        else:  
            imagen_con_padding = np.pad(
                imagen,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=0
            )
        
        info_padding = {
            'pad_top': pad_top,
            'pad_bottom': pad_bottom,
            'pad_left': pad_left,
            'pad_right': pad_right,
            'tamaño_original': (alto_actual, ancho_actual),
            'tamaño_final': (tamaño_objetivo, tamaño_objetivo)
        }
        
        return imagen_con_padding, info_padding

    def _leer_imagen_dicom(self, archivo_dcm: Path):
        """
        Lee la imagen de un archivo DICOM y la retorna como un array numpy.

        Args:
            archivo_dcm: Path del archivo DICOM
            
        Returns:
            Array numpy con la imagen
        """
        dcm = pydicom.dcmread(archivo_dcm)
        imagen = dcm.pixel_array
        # Aplicar ventaneo (normalización básica)
        imagen_ventaneada = apply_windowing(imagen, dcm)

        # Normalizar para visualización
        if dcm.PhotometricInterpretation == "MONOCHROME1":
            imagen_ventaneada = np.amax(imagen_ventaneada) - imagen_ventaneada
        else:
            imagen_ventaneada = imagen_ventaneada - np.min(imagen_ventaneada)
        return imagen_ventaneada

    def _recortar_mama(self, imagen, umbral=10, margen=10):
        """
        Recorta la imagen para eliminar fondo negro, manteniendo solo la región de la mama.
        
        Args:
            imagen: Array numpy con la imagen
            umbral: Valor mínimo de píxel para considerar contenido (default: 10)
            margen: Píxeles adicionales alrededor de la región (default: 10)
            recorte: Booleano para indicar si se debe realizar el recorte (default: True)
        
        Returns:
            imagen_recortada: Imagen sin fondo
            info_recorte: Dict con información del recorte
        """        
        # 1. Binarizar imagen
        mascara = imagen > umbral
        
        # 2. Rellenar huecos dentro de la mama
        mascara = binary_fill_holes(mascara)
        
        # 3. Eliminar objetos pequeños (ruido)
        mascara = remove_small_objects(mascara, min_size=1000)
        
        # 4. Operación de cierre para suavizar contornos
        mascara = binary_closing(mascara, footprint=disk(15))
        
        # 5. Encontrar región de interés
        coords = np.argwhere(mascara)
        
        if len(coords) == 0:
            # Si no se encuentra nada, retornar imagen original
            print("⚠ No se detectó región de mama, retornando imagen original")
            return imagen, {
                'offset_x': 0,
                'offset_y': 0,
                'x_min': 0,
                'y_min': 0,
                'x_max': imagen.shape[1],
                'y_max': imagen.shape[0],
                'ancho_recortado': imagen.shape[1],
                'alto_recortado': imagen.shape[0]
            }
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # 6. Agregar margen
        y_min = max(0, y_min - margen)
        x_min = max(0, x_min - margen)
        y_max = min(imagen.shape[0], y_max + margen)
        x_max = min(imagen.shape[1], x_max + margen)
        
        # 7. Recortar imagen
        imagen_recortada = imagen[y_min:y_max, x_min:x_max]
        
        # 8. Información del recorte
        info_recorte = {
            'offset_x': x_min,
            'offset_y': y_min,
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'ancho_recortado': x_max - x_min,
            'alto_recortado': y_max - y_min,
            'ancho_original': imagen.shape[1],
            'alto_original': imagen.shape[0]
        }
        return imagen_recortada, info_recorte
    
    def _obtener_imagen_1024(self, image: np.ndarray):
        """
        Obtiene imagen con ventaneo en dos tamaños en tamano 1024x1024.

        Args:
            image: Array numpy con la imagen cuadrada
            
        Returns:
            Imagen de tamaño 1024x1024
        """

        imagen_redimensionada = resize(
                image, 
                (1024, 1024),  
                anti_aliasing=True, 
                preserve_range=True
        )
        imagen_rescalada = rescale_intensity(
            imagen_redimensionada, 
            out_range=(0, 255)
        ).astype(np.uint8)

        return imagen_rescalada

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Inferencia")

    parser.add_argument(
        "--dcm-path",
        type=str,
        required=True,
        help="Ruta al directorio que contiene los archivos DICOM.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directorio de salida para guardar los resultados.",
    )
    parser.add_argument(
        "--yolo-path-model",
        type=str,
        required=True,
        help="Ruta al modelo entrenado para detección.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Ruta al modelo entrenado para datos clínicos.",
    )
    args = parser.parse_args()

    pipeline = PipelineInferencia(
        dcm_path=args.dcm_path,
        output_dir=args.output_dir,
        yolo_path_model=args.yolo_path_model,
        model_path=args.model_path
    )
    pipeline.ejecutar_pipeline()