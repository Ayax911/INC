from pathlib import Path
import argparse
import json

from src.data.dataset_generator import DatasetGenerator
from src.data.data_validator import DataValidator
from src.data.dataset_splitter import DatasetSplitter

class PipelinePreprocessingData:
    """Pipeline completo de estructuración de datos."""

    def __init__(self, dir_dcm: str, dir_anotaciones:str, output_dir_data: str, clinical_csv: str, report_dir: str, 
                 train_ratio: float = 0.8, test_ratio: float = 0.15, val_ratio: float = 0.05, seed: int = 42) -> None:
        """
        Inicializa el pipeline.
        
        Args:
            dir_dcm: Directorio de estudios DICOM de entrada
            dir_anotaciones: Directorio de anotaciones JSON de entrada
            output_dir_data: Directorio de salida para datos procesados
            clinical_csv: Ruta al archivo CSV con datos clínicos de pacientes
            report_dir: Directorio para guardar reportes de validación
            train_ratio: Proporción para conjunto de entrenamiento
            test_ratio: Proporción para conjunto de prueba
            val_ratio: Proporción para conjunto de validación
            seed: Semilla para reproducibilidad
        """

        # Directorios
        self.csv_path = Path(clinical_csv)
        self.dir_dcm = Path(dir_dcm)
        self.dir_anotaciones = Path(dir_anotaciones)
        self.report_dir = Path(report_dir)
        self.dir_salida = Path(output_dir_data)

        # Parámetros de división
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.seed = seed

        # Componentes
        self.data_validator = DataValidator(self.csv_path, self.dir_dcm, self.dir_anotaciones)

        # Variable para almacenar IDs validados
        self.ids_validados = []

    
    def ejecutar(self) -> None:
        """Ejecuta el pipeline completo."""        
        resultado_validacion = self.data_validator._validar_datos_completos()
        
        # Guardar resultado en JSON
        reports_dir = self.report_dir
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        with open(reports_dir / "resultado_validacion.json", 'w', encoding='utf-8') as f:
            json.dump(resultado_validacion, f, indent=4, ensure_ascii=False)

        self.ids_validados = resultado_validacion['ids_estudios_y_anotaciones']
        self._generar_dataset()
        
        
    def _generar_dataset(self) -> None:
        """Generar dataset estructurado a partir de datos validados"""
        # Crear generador de dataset
        generador = DatasetGenerator(
            dir_dcm=self.dir_dcm,                     # Directorio con archivos DICOM
            dir_anotaciones=self.dir_anotaciones,     # Directorio con anotaciones JSON
            dir_salida=self.dir_salida,               # Directorio de salida
            ids_validos=self.ids_validados            # Ids con toda la data       
        )
        # Procesar
        generador.procesar()
        self._dividir_dataset(self.dir_salida / "imagenes_sin_recortar")
        self._dividir_dataset(self.dir_salida / "imagenes_con_recorte")
    
    def _dividir_dataset(self, dir_salida: Path) -> None:
        """Divide el dataset en train/test/val."""
        # Directorios de entrada con imágenes procesadas
        dir_img_640 = dir_salida / "imagenes_con_ventaneo_640_jpg"
        dir_img_1024 = dir_salida / "imagenes_con_ventaneo_1024_jpg"
        dir_anotaciones = dir_salida / "anotaciones_txt"
        
        # Directorio de salida para dataset dividido
        dir_salida_split = dir_salida / "dataset_split"
        
        # Crear divisor de dataset
        splitter = DatasetSplitter(
            dir_img_640=dir_img_640,
            dir_img_1024=dir_img_1024,
            dir_anotaciones=dir_anotaciones,
            dir_salida=dir_salida_split,
            train_ratio=self.train_ratio,
            test_ratio=self.test_ratio,
            val_ratio=self.val_ratio,
            seed=self.seed
        )
        
        # Ejecutar división
        splitter.ejecutar()

# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesamiento de estudios DICOM con anotaciones y datos clinicos para generar dataset estructurado."
    )
    parser.add_argument(
        '--dir-dcm',
        type=str,
        required=True,
        help='Directorio de estudios DICOM de entrada'
    )
    parser.add_argument(
        '--dir-anotaciones',
        type=str,
        required=True,
        help='Directorio de anotaciones JSON de entrada'
    )
    parser.add_argument(
        '--output-dir-data',
        type=str,
        required=True,
        help='Directorio de salida para datos procesados'
    )
    parser.add_argument(
        '--clinical-csv',
        type=str,
        required=True,
        help='Ruta al archivo CSV con datos clínicos de pacientes'
    )
    parser.add_argument(
        '--report-dir',
        type=str,
        required=True,
        help='Directorio para guardar reportes de validación'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Proporción de datos para el conjunto de entrenamiento (default: 0.8)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Proporción de datos para el conjunto de prueba (default: 0.15)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.05,
        help='Proporción de datos para el conjunto de validación (default: 0.05)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla para reproducibilidad en la división del dataset (default: 42)'
    )

    args = parser.parse_args()

    pipeline = PipelinePreprocessingData(
        dir_dcm=args.dir_dcm,
        dir_anotaciones=args.dir_anotaciones,
        output_dir_data=args.output_dir_data,
        clinical_csv=args.clinical_csv,
        report_dir=args.report_dir,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    pipeline.ejecutar()
