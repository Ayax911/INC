from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryAUROC, BinaryPrecision, BinarySpecificity, BinaryRecall
import torch


class Metrics():

    """ Clase que calcula las metricas de evaluacion de similitud entre dos imagenes """

    def __init__(self, device) -> None:
        
        self.accuracy       = BinaryAccuracy(threshold=0.5).to(device)
        self.sensitivity    = BinaryRecall().to(device)
        self.specificity    = BinarySpecificity().to(device)
        self.f1_score       = BinaryF1Score().to(device)
        self.bce_loss       = torch.nn.CrossEntropyLoss().to(device)
        

    def get_metrics(self, prediction: torch.tensor, target: torch.tensor, probs: torch.tensor, stage: str)->dict:

        """
        Calculate metrics for a given set of predictions and targets.

        Args:
            prediction (torch.tensor): Predictions tensor.
            target (torch.tensor): Target tensor.
            probs (torch.tensor): Probability tensor.

        Returns:
            dict: Dictionary of metrics.
        """
        
        # Calcular las metricas de clasificacion
        accuracy_       = self.accuracy(prediction, target)
        sensitivity_    = self.sensitivity(prediction, target)
        specificity_    = self.specificity(prediction, target)
        f1_score_       = self.f1_score(prediction, target)
        bce_error_      = self.bce_loss(probs, target)
        vpp_            = self.vpp(prediction, target)

        dict_metrics = {
            f"{stage}_Accuracy"       : accuracy_,
            f"{stage}_Sensitivity"    : sensitivity_,
            f"{stage}_Specificity"    : specificity_,
            f"{stage}_F1-Score"       : f1_score_,
            f"{stage}_BCE-Loss"       : bce_error_,
            f"{stage}_VPP"            : vpp_
        }

        return dict_metrics
    
    import torch

    def vpp(self, preds, targets):
        """
        logits:  [N, 2]
        targets: [N] con 0 o 1
        """


        # TP: pred=1 y target=1
        TP = ((preds == 1) & (targets == 1)).sum().item()

        # FP: pred=1 pero target=0
        FP = ((preds == 1) & (targets == 0)).sum().item()

        # Evitar división por cero
        if TP + FP == 0:
            return 0.0

        return TP / (TP + FP)


# --------------------------------------------------------------------------------------
# Métricas por época con el MISMO cálculo que FedMammoBench (src/metrics.py): torchmetrics
# acumulado sobre TODA la época (update por batch, compute al final), no el promedio de
# métricas por batch que usa la clase Metrics de arriba. Son las que se loguean a W&B y a
# metrics.csv con las claves train_*/val_*, para que las curvas de ambos proyectos sean
# comparables panel a panel. Metrics (arriba) se mantiene tal cual para el early stopping.
# --------------------------------------------------------------------------------------
from torchmetrics import Metric, MetricCollection


class BinaryMacroF1Score(Metric):
    """F1-macro binario: promedio del F1 de la clase negativa y de la positiva.

    Copia de BinaryMacroF1Score de FedMammoBench -- equivale a
    sklearn f1_score(average="macro"). Recibe la probabilidad de la clase positiva [B].
    """

    higher_is_better = True
    is_differentiable = False
    full_state_update = False

    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("tp", default=torch.zeros(2), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(2), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(2), dist_reduce_fx="sum")

    def update(self, preds, target):
        pred_labels = (preds > self.threshold).long() if preds.is_floating_point() else preds.long()
        target = target.long()
        for class_idx in (0, 1):
            predicted = pred_labels == class_idx
            actual = target == class_idx
            self.tp[class_idx] += (predicted & actual).sum()
            self.fp[class_idx] += (predicted & ~actual).sum()
            self.fn[class_idx] += (~predicted & actual).sum()

    def compute(self):
        denominator = 2 * self.tp + self.fp + self.fn
        f1_per_class = torch.where(denominator > 0, 2 * self.tp / denominator, torch.zeros_like(denominator))
        return f1_per_class.mean()


def build_metric_collection(device="cpu"):
    """Las siete métricas de FedMammoBench, con los mismos nombres de clave."""
    return MetricCollection({
        "accuracy": BinaryAccuracy(),
        "auc": BinaryAUROC(),
        "sensitivity": BinaryRecall(),
        "specificity": BinarySpecificity(),
        "f1": BinaryF1Score(),              # solo la clase positiva
        "f1_macro": BinaryMacroF1Score(),   # promedio de ambas clases
        "precision": BinaryPrecision(),     # = VPP
    }).to(device)
