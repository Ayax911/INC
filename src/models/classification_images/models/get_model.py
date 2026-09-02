from models.image_models import get_image_model
from torch import nn
import torch
from types import SimpleNamespace
from models.mlp_models import MLP

def get_model(options:dict):
    
    image_model = get_image_model(
        model_name      = options.image_model,
        weigths_file    = options.path_image_model,
        num_freeze      = options.num_freeze
    )
    
    classifier  = MLP(input_size=image_model.features, hidden_layers=options.hidden_layers, output_size=options.output_size, activation=options.activation, dropout=options.dropout)
    final_model = nn.Sequential(image_model, classifier)

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