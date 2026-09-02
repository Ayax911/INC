from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryAUROC, BinaryPrecision, BinarySpecificity, BinaryRecall
import torch


class Metrics():

    """ Clase que calcula las metricas de evaluacion de similitud entre dos imagenes """

    def __init__(self, device) -> None:
        
        """
        Constructor de la clase Metrics 
        
        Args:
            - device: (torch.device), device to run the metrics on.
        
        """
        
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
        
        # Asegurar tipo long y binario
        pred = prediction.view(-1).long()
        tgt  = target.view(-1).long()

        # Calcular TP, FP, TN, FN (binario: 0 = negativo, 1 = positivo)
        tp = torch.sum((pred == 1) & (tgt == 1)).item()
        fp = torch.sum((pred == 1) & (tgt == 0)).item()
        tn = torch.sum((pred == 0) & (tgt == 0)).item()
        fn = torch.sum((pred == 0) & (tgt == 1)).item()

        # Valor predictivo positivo (VPP = TP / (TP + FP))
        vpp = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        dict_metrics = {
            f"{stage}_Accuracy"    : accuracy_,
            f"{stage}_Sensitivity" : sensitivity_,
            f"{stage}_Specificity" : specificity_,
            f"{stage}_F1-Score"    : f1_score_,
            f"{stage}_BCE-Loss"    : bce_error_,
            f"{stage}_VPP"         : vpp,
        }

        return dict_metrics
    