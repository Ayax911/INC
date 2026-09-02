from torch import nn
import torch
from types import SimpleNamespace
from models.mlp_models import MLP

def get_model(options:dict):
    
    """
    Function to get the model based on the given options.
    
    Args:
        options (dict): Dictionary containing model configuration options.
        
    Returns:
        model (nn.Module): Configured model instance.
    """
    
    final_model = MLP(
        input_size      = options.input_size_clinic_model,
        hidden_layers   = options.hidden_layers_clinic_model,
        output_size     = options.output_size_clinic_model,
        activation      = options.activation_clinic_model,
        dropout         = options.dropout_clinic_model,
    )

    return final_model


if __name__ == "__main__":
    
    options = {
        "strategy"          : "images",
        "image_model"       : "ResNet",
        "path_image_model"  : "/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/03-Challenges/01-MAMA-MIA/01-Code/RadImageNet_pytorch/01-Pytorch/ResNet50.pt",
        "freeze_backbone"   : True,
    }
    
    options = SimpleNamespace(**options)
    
    model   = get_model(options)
    image   = torch.randn(1, 3, 224, 224)
    output  = model(image)
    print(model)