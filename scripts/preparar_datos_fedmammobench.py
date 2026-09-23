"""Convierte el dataset de FedMammoBench al formato que consume classification_images.

Puente de datos para la comparación INC vs FedMammoBench con hiperparámetros
idénticos (ver ../../run.sh). Produce dos cosas a partir de UN
manifest de FedMammoBench:

  1. Los tres CSV de splits, siempre.
  2. Opcionalmente (`--formato npy`) un directorio de parches `.npy`
     (224, 224, 3) float32 para `--images_dir`.

Con el default `--formato tiff` NO se copia ninguna imagen: los CSV apuntan a los
`.tiff` originales de FedMammoBench y `--images_dir` es su mismo `image_root`, así
que las dos rutas de código leen literalmente el mismo archivo.

Los CSV son `train_clinical_data.csv`, `val_clinical_data.csv` y
     `test_clinical_data.csv` -- lo que espera `--csv_data_path`, con los
     nombres exactos que arma `dataloaders/dataloader_images.py:Loader`.
     Columna 0 = ruta del .npy relativa a `--images_dir`, columna 1 = etiqueta
     entera (benigno=0, maligno=1), que es el orden posicional que lee
     `ImageDataset.__getitem__` (`self.data.iloc[sample, 0]` / `[sample, 1]`).

CUÁNDO HACE FALTA --formato npy
-------------------------------
Ya casi nunca. `ImageDataset` detecta el modo PIL `"F"` (TIFF float de un canal,
lo que escribe el preprocesamiento de FedMammoBench) y lo lee sin `.convert()`,
replicando a 3 canales sobre el array -- igual que `MammoBenchDataset`.

Antes de ese soporte, la única rama para `.tiff` era
`Image.open(path).convert("RGB")`, que sobre modo `"F"` trunca y recorta a
enteros 0..255 en vez de reescalar: las imágenes salían COMPLETAMENTE NEGRAS
(medido: min=max=mean=0, tanto en [0,1] como en [-1,1]). El espejo `.npy` existía
para esquivar eso por la rama `np.load()`, que no toca los valores.

Sigue disponible por si hay que correr contra una copia del dataloader anterior a
ese arreglo. Cuesta ~5 GB: se replica a 3 canales en disco porque el backbone
espera 3 y `T.ToTensor()` sobre un array 2-D entregaría (1, H, W).

Uso:
    python3 scripts/preparar_datos_fedmammobench.py \\
        --manifest ../FedMammoBench/manifests/fedmammobench_norm_0_1.csv \\
        --image-root ../FedMammoBench/data/preproccesed_julian \\
        --out-images data/fedmammobench_npy/norm_0_1 \\
        --out-splits data/fedmammobench_splits/norm_0_1

Es idempotente: un .npy que ya existe no se vuelve a escribir, así que
relanzarlo tras una interrupción solo completa lo que falta. Los tres CSV sí
se reescriben siempre (son baratos).
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

# Mismo criterio que `Manifest._normalize_labels()` de FedMammoBench: la clase
# positiva (1) es la maligna. Si se invirtiera acá, los pesos [neg, pos] de la
# loss quedarían al revés respecto del config centralizado.
LABEL_MAP = {"benign": 0, "malignant": 1}

# Nombres de archivo exigidos por dataloader_images.py:Loader -- no son
# configurables desde la CLI del INC.
SPLIT_FILENAMES = {
    "train": "train_clinical_data.csv",
    "val": "val_clinical_data.csv",
    "test": "test_clinical_data.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convierte el manifest de FedMammoBench a parches .npy + CSV de splits del INC"
    )
    parser.add_argument("--manifest", required=True, help="CSV manifest de FedMammoBench")
    parser.add_argument(
        "--image-root",
        required=True,
        help="Raíz contra la que resolver `preprocessed_image_path` del manifest",
    )
    parser.add_argument(
        "--formato",
        default="tiff",
        choices=["tiff", "npy"],
        help=(
            "tiff (default): NO copia imágenes; los CSV apuntan a los .tiff originales, que el "
            "dataloader ya lee bien desde que soporta el modo 'F'. npy: escribe un espejo .npy "
            "(~5 GB) -- solo hace falta con una copia del dataloader anterior a ese soporte."
        ),
    )
    parser.add_argument("--out-images", help="Directorio destino de los .npy (solo con --formato npy)")
    parser.add_argument("--out-splits", required=True, help="Directorio destino de los 3 CSV")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16"],
        help="Precisión de los .npy (float16 pesa la mitad; ToTensor los sube a float32)",
    )
    return parser.parse_args()


def convertir_imagen(src: str, dst: str, dtype: str) -> None:
    """Lee un TIFF float de 1 canal y lo escribe como .npy (H, W, 3) sin reescalar."""
    arr = np.array(Image.open(src))  # modo "F" -> float32 (H, W), valores intactos
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    np.save(dst, arr.astype(dtype))


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.manifest)
    faltan = {"preprocessed_image_path", "classification", "split"} - set(df.columns)
    if faltan:
        raise SystemExit(f"El manifest no tiene las columnas {sorted(faltan)}")

    df["label"] = df["classification"].str.lower().map(LABEL_MAP)
    if df["label"].isna().any():
        desconocidas = sorted(df.loc[df["label"].isna(), "classification"].unique())
        raise SystemExit(f"Etiquetas no reconocidas en `classification`: {desconocidas}")

    os.makedirs(args.out_splits, exist_ok=True)

    if args.formato == "tiff":
        # Nada que convertir: los CSV apuntan a los .tiff tal cual, y --images_dir del
        # INC es el mismo image_root de FedMammoBench. Las dos rutas de código leen
        # entonces exactamente el mismo archivo, sin copia intermedia.
        df["ruta_rel"] = df["preprocessed_image_path"]
        print(f"Formato tiff: sin copia de imágenes; --images_dir debe ser {args.image_root}")
    else:
        if not args.out_images:
            raise SystemExit("--formato npy requiere --out-images")
        # `preprocessed_image_path` viene como norm_0_1/<base>/<archivo>.tiff; se
        # conserva esa jerarquía en el destino para no colisionar nombres entre bases.
        df["ruta_rel"] = df["preprocessed_image_path"].str.replace(r"\.tiff?$", ".npy", regex=True)
        os.makedirs(args.out_images, exist_ok=True)

        convertidas, saltadas = 0, 0
        for rel_tiff, rel_npy in zip(df["preprocessed_image_path"], df["ruta_rel"]):
            dst = os.path.join(args.out_images, rel_npy)
            if os.path.exists(dst):
                saltadas += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            convertir_imagen(os.path.join(args.image_root, rel_tiff), dst, args.dtype)
            convertidas += 1
            if convertidas % 500 == 0:
                print(f"  {convertidas} imágenes convertidas...", flush=True)

        print(f"Imágenes: {convertidas} convertidas, {saltadas} ya existían -> {args.out_images}")

    for split, filename in SPLIT_FILENAMES.items():
        sub = df[df["split"] == split]
        if sub.empty:
            raise SystemExit(f"El manifest no tiene filas con split={split!r}")
        destino = os.path.join(args.out_splits, filename)
        sub[["ruta_rel", "label"]].to_csv(destino, index=False, header=["image", "label"])
        n_pos = int(sub["label"].sum())
        print(f"{filename}: {len(sub)} filas (benigno={len(sub) - n_pos}, maligno={n_pos}) -> {destino}")

    # Los pesos de la loss del config centralizado salen de aquí; se imprimen para
    # poder verificarlos contra el YAML sin recalcularlos a mano.
    tr = df[df["split"] == "train"]
    n_neg, n_pos, total = int((tr["label"] == 0).sum()), int((tr["label"] == 1).sum()), len(tr)
    if n_neg and n_pos:
        print(
            "\nPesos de clase sobre train (fórmula total/(2*n_clase), la de exp05):\n"
            f"  --neg_weight {total / (2 * n_neg):.4f}   --pos_weight {total / (2 * n_pos):.4f}"
        )
    else:
        print(
            f"\nSin pesos de clase: el split de train tiene una sola clase "
            f"(benigno={n_neg}, maligno={n_pos})."
        )


if __name__ == "__main__":
    main()
