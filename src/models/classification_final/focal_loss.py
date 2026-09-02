import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss para clasificación multiclase con logits [N, C].
    Ideal para C=2 (binario con 2 logits).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        
        self.alpha = alpha  # tensor con pesos por clase o None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits:  [N, 2]
        targets: [N] con 0 o 1
        """

        # Probabilidades
        probs = F.softmax(logits, dim=1)          # [N,2]

        # Probabilidad asignada a la clase correcta
        pt = probs[range(len(targets)), targets]  # [N]

        # Focal term (1 - pt)^gamma
        focal_factor = (1 - pt) ** self.gamma

        # log_softmax para CrossEntropy estable
        log_probs = F.log_softmax(logits, dim=1)
        ce_loss = -log_probs[range(len(targets)), targets]  # CrossEntropy base

        # Aplica alpha si se especifica
        if self.alpha is not None:
            alpha_factor = self.alpha[targets]  # α para cada muestra
            focal_loss = alpha_factor * focal_factor * ce_loss
        else:
            focal_loss = focal_factor * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
