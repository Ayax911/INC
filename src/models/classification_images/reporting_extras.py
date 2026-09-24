"""Artefactos de métricas/evaluación adicionales a los que ya produce `training.py`.

Inspirado en cómo FedMammoBench separa esto en `src/tracking.py`/`src/reporting.py`
(ver CLAUDE.md, sección "Reference: metrics/evaluation/W&B in the sibling
FedMammoBench project"), pero sin tocar nada de lo que `metrics.py`/
`early_stopping.py`/`training.py` ya calculan o deciden -- esto solo agrega
artefactos nuevos encima:

    - EpochLogger: escribe `metrics.csv` (una fila por época) + eventos de
      TensorBoard, además de lo que `training.py` ya loguea a wandb.
    - compute_confusion_matrix_metrics(): métricas derivadas de la matriz de
      confusión que `metrics.Metrics` no trackea (NPV, MCC, kappa,
      balanced accuracy, likelihood ratios), calculadas una sola vez sobre
      las predicciones completas de un split, no por batch.
    - BinaryMacroF1Score: promedio del F1 de ambas clases (positiva y
      negativa) -- complementa a `BinaryF1Score` de `metrics.py`, que solo
      mide la clase positiva.
    - save_metrics_json() / save_predictions_csv(): persisten a disco lo
      mismo que ya se calcula en `training.py`, en formato reusable
      (`predictions.csv` con y_true/y_pred/y_prob, `metrics.json`).

Ejemplo de uso (ver `training.py:_evaluate_and_report`):
    >>> logger = EpochLogger(run_dir="results/exp01")
    >>> logger.log(epoch=1, metrics={"Train_Accuracy": 0.8, "Val_Accuracy": 0.75})
    >>> logger.close()
"""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, ModuleNotFoundError):
    SummaryWriter = None
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef
from torchmetrics import Metric
from torchmetrics.classification import BinaryAUROC


class EpochLogger:
    """Escribe `metrics.csv` (una fila por época) y eventos de TensorBoard.

    Abre ambos de forma perezosa -- recién en la primera llamada a `log()`,
    nunca en `__init__` -- para no truncar el `metrics.csv` de una corrida
    de entrenamiento previa si nadie vuelve a llamar a `log()` (ej. una
    corrida lanzada solo con `--test`, que reevalúa un checkpoint ya
    entrenado sin reentrenar).
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_file: Any = None
        self._csv_writer: Any = None
        self._tb_writer: SummaryWriter | None = None

    def log(self, epoch: int, metrics: dict[str, float]) -> None:
        row: dict[str, Any] = {"epoch": epoch, **metrics}

        if self._csv_file is None:
            self._csv_file = open(self._csv_path, "w", newline="")
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        if self._tb_writer is None and SummaryWriter is not None:
            self._tb_writer = SummaryWriter(log_dir=str(self.run_dir))
        if self._tb_writer is not None:
            for name, value in metrics.items():
                try:
                    self._tb_writer.add_scalar(name, value, epoch)
                except (TypeError, ValueError):
                    pass  # valores no escalares (si los hubiera) se ignoran en TensorBoard

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
        if self._tb_writer is not None:
            self._tb_writer.close()


class BinaryMacroF1Score(Metric):
    """F1-macro binario: promedio simple del F1 de la clase negativa y la positiva.

    Copiado de FedMammoBench (`src/metrics.py:BinaryMacroF1Score`) -- complementa
    a `BinaryF1Score` (`metrics.py` de este subproyecto), que solo mide la clase
    positiva (maligno) y por eso puede quedar en 0.0 varias épocas mientras el
    backbone está congelado. Pensada para uso "de una sola vez" sobre un split
    completo (ver `compute_extra_metrics()`), no para acumular por batch dentro
    del loop de entrenamiento.
    """

    higher_is_better: bool = True
    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(self, threshold: float = 0.5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("tp", default=torch.zeros(2), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(2), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(2), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        pred_labels = (preds > self.threshold).long() if preds.is_floating_point() else preds.long()
        target = target.long()
        for class_idx in (0, 1):
            predicted = pred_labels == class_idx
            actual = target == class_idx
            self.tp[class_idx] += (predicted & actual).sum()
            self.fp[class_idx] += (predicted & ~actual).sum()
            self.fn[class_idx] += (~predicted & actual).sum()

    def compute(self) -> torch.Tensor:
        denominator = 2 * self.tp + self.fp + self.fn
        f1_per_class = torch.where(
            denominator > 0, 2 * self.tp / denominator, torch.zeros_like(denominator)
        )
        return f1_per_class.mean()


def compute_extra_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob_positive: np.ndarray) -> dict[str, float]:
    """Métricas adicionales a las 6 que ya trackea `metrics.Metrics`, sobre un split completo.

    No reemplaza nada -- `_evaluate_and_report()` (`training.py`) sigue calculando
    Accuracy/Sensitivity/Specificity/F1/BCE/VPP como siempre; esto se guarda aparte
    en `confusion_matrix_metrics.json` y como resumen extra en W&B.

    Args:
        y_true: etiquetas reales (0=benigno, 1=maligno).
        y_pred: clase predicha (argmax de la probabilidad softmax).
        y_prob_positive: probabilidad de la clase maligna (columna 1 del softmax).

    Returns:
        dict con AUC, F1-macro, y las métricas derivadas de la matriz de
        confusión (NPV, PPV, MCC, kappa, balanced accuracy, likelihood ratios),
        todas con prefijo `cm_` salvo `auc` y `f1_macro`.
    """
    y_true_t = torch.as_tensor(y_true, dtype=torch.long)
    y_prob_t = torch.as_tensor(y_prob_positive, dtype=torch.float32)
    y_pred_t = torch.as_tensor(y_pred, dtype=torch.long)

    auc_metric = BinaryAUROC()
    f1_macro_metric = BinaryMacroF1Score()

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "auc": auc_metric(y_prob_t, y_true_t).item(),
        "f1_macro": f1_macro_metric(y_pred_t, y_true_t).item(),
        "cm_tp": tp,
        "cm_tn": tn,
        "cm_fp": fp,
        "cm_fn": fn,
        "cm_npv": npv,
        "cm_ppv": ppv,
        "cm_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cm_mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cm_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "cm_positive_likelihood_ratio": (sensitivity / (1 - specificity)) if specificity < 1 else float("inf"),
        "cm_negative_likelihood_ratio": ((1 - sensitivity) / specificity) if specificity > 0 else float("inf"),
    }


def save_metrics_json(metrics: dict[str, Any], path: str | Path) -> None:
    """Vuelca un dict de métricas a JSON, creando el directorio padre si hace falta."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: (v.item() if hasattr(v, "item") else v) for k, v in metrics.items()}
    path.write_text(json.dumps(clean, indent=2, sort_keys=True))


def save_predictions_csv(y_true: Any, y_pred: Any, y_prob: Any, path: str | Path) -> None:
    """Guarda etiquetas/predicciones/probabilidades alineadas por posición en un CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred", "y_prob"])
        writer.writerows(zip(y_true, y_pred, y_prob))
