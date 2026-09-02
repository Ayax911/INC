import torch
import torch.nn as nn


def get_loss(name:str = "", positive_weight:float = None, negative_weight: float = None):
    
    """
    Get loss function
    
    Args:
        name (str): Name of the loss function
        positive_weight (float): Weight for positive class
        negative_weight (float): Weight for negative class
        
    Returns:
        loss function
    """
    
    name = name.lower()
    
    if name == "bce":
        weigths = torch.tensor([negative_weight, positive_weight]).to(torch.device('cuda'))
        return nn.CrossEntropyLoss( weight = weigths)
    
    else:
        raise ValueError(f"Función de perdida no soportada: {name}")