"""Artefactos de entrenamiento/evaluación con el MISMO formato que FedMammoBench.

Port de `FedMammoBench/src/reporting.py` (repo hermano en ../FedMammoBench): mismas
funciones, mismos nombres de archivo, mismos ejes y mismo layout de gráficas, para que
una corrida de este proyecto y una de FedMammoBench se lean y comparen igual, tanto en
disco como en W&B. Si cambia allá, hay que portarlo a mano -- los dos repos no comparten
código.

Qué escribe (todo bajo `<result_dir>/<exp_name>/`):
    metrics.csv                     una fila por época, claves train_*/val_*
    plots/loss_curve.png            train vs val loss, eje Y desde 0 y techo >= 1
    plots/<métrica>_curve.png       train vs val por métrica, eje Y fijo [0, 1]
    val/ y test/                    metrics.json, confusion_matrix_metrics.json,
                                    predictions.csv (y_true,y_pred,y_prob),
                                    confusion_matrix.png, roc_curve.png
    test/                           + desglose por base de datos (ver
                                    plot_confusion_matrix_by_database/plot_metrics_by_database)

Usa torchmetrics para matriz de confusión y ROC, igual que FedMammoBench, para que los
números salgan de la misma implementación en ambos proyectos.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin display -- sin esto, corriendo sin X11 Qt aborta con core dump
import matplotlib.pyplot as plt
import torch
from torchmetrics.classification import BinaryConfusionMatrix, BinaryROC

# Nombres a mostrar -- idénticos a METRIC_DISPLAY_NAMES de FedMammoBench (eval_pipeline.py).
METRIC_DISPLAY_NAMES = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "sensitivity": "Sensibilidad",
    "specificity": "Especificidad",
    "f1": "F1",
    "f1_macro": "F1-macro",
    "precision": "Precisión",
}


def save_metrics_json(metrics, path):
    """Vuelca un dict de métricas a JSON, creando el directorio padre si hace falta."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True))


def save_predictions_csv(y_true, y_pred, y_prob, path):
    """Guarda `y_true,y_pred,y_prob` alineados por posición -- mismo esquema que FedMammoBench.

    Args:
        y_true: etiquetas reales (0=benigno, 1=maligno).
        y_pred: clase predicha.
        y_prob: probabilidad de la clase positiva (softmax[:, 1]).
        path: ruta del `.csv`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred", "y_prob"])
        writer.writerows(zip(y_true, y_pred, y_prob))


class MetricsCsvWriter:
    """Escribe `metrics.csv` una fila por época (columna `epoch` + las claves recibidas).

    Mismo formato que `MetricsLogger.log()` de FedMammoBench. Se abre en el primer
    `write()`, no en el constructor, para no truncar un `metrics.csv` existente si nadie
    llega a escribir (ej. una corrida con `--test` sin `--train`).
    """

    def __init__(self, path):
        self.path = Path(path)
        self._file = None
        self._writer = None

    def write(self, epoch, metrics):
        row = {"epoch": epoch, **metrics}
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()


def plot_confusion_matrix(y_true, y_pred, path, class_names=("Benign", "Malignant")):
    """Matriz de confusión binaria como PNG (filas = real, columnas = predicho)."""
    cm = BinaryConfusionMatrix()(torch.tensor(y_pred), torch.tensor(y_true)).numpy()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    ax.set_xticks([0, 1], labels=class_names)
    ax.set_yticks([0, 1], labels=class_names)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _safe_div(numerator, denominator):
    """División con 0.0 si el denominador es 0 (zero_division=0, como scikit-learn)."""
    return numerator / denominator if denominator != 0 else 0.0


def compute_confusion_matrix_metrics(y_true, y_pred):
    """Todas las métricas binarias derivables de TP/TN/FP/FN, con prefijo `cm_`.

    Idéntica a `compute_confusion_matrix_metrics()` de FedMammoBench -- ver su docstring
    para el detalle de cada métrica y la advertencia sobre los likelihood ratios que valen
    0.0 cuando el valor real es +infinito.
    """
    cm = BinaryConfusionMatrix()(torch.tensor(y_pred), torch.tensor(y_true)).numpy()
    tn, fp, fn, tp = (float(cm[0, 0]), float(cm[0, 1]), float(cm[1, 0]), float(cm[1, 1]))
    total = tp + tn + fp + fn

    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)

    accuracy = _safe_div(tp + tn, total)
    f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity)
    f1_negative = _safe_div(2 * npv * specificity, npv + specificity)

    mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    expected_agreement = _safe_div((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp), total * total)

    return {
        "cm_tp": tp,
        "cm_tn": tn,
        "cm_fp": fp,
        "cm_fn": fn,
        "cm_accuracy": accuracy,
        "cm_balanced_accuracy": (sensitivity + specificity) / 2,
        "cm_sensitivity": sensitivity,
        "cm_specificity": specificity,
        "cm_precision": precision,
        "cm_npv": npv,
        "cm_fpr": _safe_div(fp, fp + tn),
        "cm_fnr": _safe_div(fn, fn + tp),
        "cm_fdr": _safe_div(fp, fp + tp),
        "cm_for": _safe_div(fn, fn + tn),
        "cm_f1": f1,
        "cm_f1_macro": (f1 + f1_negative) / 2,
        "cm_mcc": _safe_div(tp * tn - fp * fn, mcc_denominator),
        "cm_kappa": _safe_div(accuracy - expected_agreement, 1 - expected_agreement),
        "cm_youden_j": sensitivity + specificity - 1,
        "cm_markedness": precision + npv - 1,
        "cm_prevalence": _safe_div(tp + fn, total),
        "cm_positive_likelihood_ratio": _safe_div(sensitivity, 1 - specificity),
        "cm_negative_likelihood_ratio": _safe_div(1 - sensitivity, specificity),
        "cm_diagnostic_odds_ratio": _safe_div(tp * tn, fp * fn),
        "cm_g_mean": (sensitivity * specificity) ** 0.5,
        "cm_fowlkes_mallows": (precision * sensitivity) ** 0.5,
        "cm_threat_score": _safe_div(tp, tp + fn + fp),
    }


def _check_curves(train_values, val_values):
    if len(train_values) != len(val_values):
        raise ValueError(
            f"train y val deben tener el mismo largo — recibidos {len(train_values)} y {len(val_values)}"
        )
    if not train_values:
        raise ValueError("No hay ninguna época que graficar (historial vacío).")


def plot_loss_curve(train_loss, val_loss, path, best_epoch=None):
    """Pérdida train vs val por época.

    Eje Y desde 0 y con techo mínimo 1 (no un [0, 1] fijo: la CrossEntropy ponderada
    puede arrancar por encima de 1 y no se recorta). `best_epoch` en base 0, se dibuja
    en base 1 igual que el eje.
    """
    _check_curves(train_loss, val_loss)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Training loss", color="#1f77b4")
    ax.plot(epochs, val_loss, label="Validation loss", color="#d62728")
    if best_epoch is not None:
        ax.axvline(best_epoch + 1, linestyle="--", color="gray", lw=1, label=f"Mejor época ({best_epoch + 1})")
    ax.set_ylim(0, max(1.0, max(train_loss), max(val_loss)))
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title("Pérdida de entrenamiento vs. validación")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_metric_curve(train_values, val_values, path, metric_name, best_epoch=None):
    """Una métrica clínica train vs val por época, eje Y fijo [0, 1]."""
    _check_curves(train_values, val_values)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_values) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_values, label=f"Train {metric_name}", color="#1f77b4")
    ax.plot(epochs, val_values, label=f"Val {metric_name}", color="#d62728")
    if best_epoch is not None:
        ax.axvline(best_epoch + 1, linestyle="--", color="gray", lw=1, label=f"Mejor época ({best_epoch + 1})")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Época")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} — entrenamiento vs. validación")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc_curve(y_true, y_prob, path):
    """Curva ROC + AUC como PNG."""
    fpr, tpr, _ = BinaryROC()(torch.tensor(y_prob), torch.tensor(y_true))
    roc_auc = torch.trapz(tpr, fpr).item()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr.numpy(), tpr.numpy(), label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Aleatorio")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_metrics_by_database_json(metrics_by_db, path):
    """Vuelca `base_de_datos -> {métrica: valor}` a JSON."""
    save_metrics_json(metrics_by_db, path)


def plot_confusion_matrix_by_database(cm_by_db, path, class_names=("Benign", "Malignant")):
    """Panel 1xN con una matriz de confusión por base de datos (`db -> (y_true, y_pred)`)."""
    if not cm_by_db:
        raise ValueError("cm_by_db está vacío -- no hay ninguna base de datos que graficar.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    db_names = list(cm_by_db.keys())
    fig, axes_grid = plt.subplots(1, len(db_names), figsize=(5 * len(db_names), 5), squeeze=False)
    for ax, db_name in zip(axes_grid[0], db_names):
        y_true, y_pred = cm_by_db[db_name]
        cm = BinaryConfusionMatrix()(torch.tensor(y_pred), torch.tensor(y_true)).numpy()
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
        ax.set_xticks([0, 1], labels=class_names)
        ax.set_yticks([0, 1], labels=class_names)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title(db_name)

    fig.suptitle("Matriz de confusión por base de datos")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# Misma paleta fija que FedMammoBench: un color por métrica, nunca reciclado.
_METRIC_BAR_COLORS = {
    "accuracy": "#2a78d6",
    "auc": "#eb6834",
    "sensitivity": "#1baf7a",
    "specificity": "#eda100",
    "precision": "#e87ba4",
    "f1_macro": "#008300",
}


def plot_metrics_by_database(
    metrics_by_db,
    path,
    metric_keys=("accuracy", "auc", "sensitivity", "specificity", "precision", "f1_macro"),
    metric_display_names=None,
):
    """Barras agrupadas por base de datos y por métrica, eje Y fijo [0, 1], valor encima de cada barra."""
    if not metrics_by_db:
        raise ValueError("metrics_by_db está vacío -- no hay ninguna base de datos que graficar.")

    display_names = metric_display_names or {}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    db_names = list(metrics_by_db.keys())
    n_metrics = len(metric_keys)
    bar_width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(6.0, 2.2 * len(db_names)), 5))
    for metric_idx, metric_key in enumerate(metric_keys):
        offsets = [g + (metric_idx - (n_metrics - 1) / 2) * bar_width for g in range(len(db_names))]
        values = [metrics_by_db[db].get(metric_key, 0.0) for db in db_names]
        bars = ax.bar(
            offsets,
            values,
            width=bar_width * 0.9,
            label=display_names.get(metric_key, metric_key),
            color=_METRIC_BAR_COLORS.get(metric_key, "#52514e"),
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    ax.set_xticks(range(len(db_names)), labels=db_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valor de la métrica")
    ax.set_title("Métricas de test por base de datos")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    # Leyenda fuera del área de dibujo para no tapar barras cercanas a 0.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(n_metrics, 6), fontsize=8, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
