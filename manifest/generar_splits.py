"""
Genera los CSV train/val/test que espera classification_images/dataloaders/dataloader_images.py
(Loader.__init__ busca train_clinical_data.csv, test_clinical_data.csv, val_clinical_data.csv
en el directorio pasado como --csv_data_path).

ImageDataset.__getitem__ usa las columnas [0] y [1] por POSICION (.iloc[sample, 0] para la
ruta de la imagen, .iloc[sample, 1] para la etiqueta), asi que se anteponen dos columnas
derivadas con esos nombres, y se conservan todas las demas columnas originales del manifest:
    - image_path (columna 0): copia de 'preprocessed_image_path'
      (ej. "norm_neg1_1/cmmd/cmmd_0.tiff"), se concatena con --images_dir en el dataloader.
    - label (columna 1): 0 = Benign, 1 = Malignant, codificada desde 'classification'.
    - el resto de columnas del manifest se mantienen tal cual (incluye 'preprocessed_image_path'
      y 'classification' originales), excepto 'split', que ya queda implicita por archivo.

Particion segun la columna 'split' del manifest (train/val/test).

Usa solo la libreria estandar (csv) para no depender de pandas/el env conda.
"""

import csv
import os

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "fedmammobench_norm_neg1_1.csv")
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "splits")

LABEL_MAP = {"Benign": 0, "Malignant": 1}

SPLIT_TO_FILENAME = {
    "train": "train_clinical_data.csv",
    "val":   "val_clinical_data.csv",
    "test":  "test_clinical_data.csv",
}


def main():
    rows_by_split = {split: [] for split in SPLIT_TO_FILENAME}
    original_fieldnames = None

    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_fieldnames = reader.fieldnames
        for row in reader:
            split = row["split"]
            if split not in rows_by_split:
                raise ValueError(f"Valor de 'split' no esperado: {split!r}")

            label_str = row["classification"]
            if label_str not in LABEL_MAP:
                raise ValueError(f"Valor de 'classification' sin mapeo en LABEL_MAP: {label_str!r}")

            out_row = {
                "image_path": row["preprocessed_image_path"],
                "label":      LABEL_MAP[label_str],
            }
            out_row.update(row)
            del out_row["split"]

            rows_by_split[split].append(out_row)

    fieldnames = ["image_path", "label"] + [c for c in original_fieldnames if c != "split"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split_name, filename in SPLIT_TO_FILENAME.items():
        out_path = os.path.join(OUTPUT_DIR, filename)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_by_split[split_name])
        print(f"{filename}: {len(rows_by_split[split_name])} filas -> {out_path}")


if __name__ == "__main__":
    main()
