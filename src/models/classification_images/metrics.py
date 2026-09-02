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
