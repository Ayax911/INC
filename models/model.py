from torch import nn
import torch
from torchvision.models import resnet50, densenet121, inception_v3


def get_hidden_layer_features(model, x, layer_idx):
    # Recorremos las capas del modelo
    for i, layer in enumerate(model.model.children()):
        x = layer(x)
        if i == layer_idx:
            return x  # Devolver la capa en el índice especificado
    return x

def get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "leakyrelu":
        return nn.LeakyReLU(0.2)
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Función de activación no soportada: {name}")

class MLP_Final_Model(nn.Module):
    def __init__(self):
        super(MLP_Final_Model, self).__init__()
                
        self.image_model = ResNetModel()
        
        self.clinic_model = MLP(
            input_size      = 51, 
            hidden_layers   = [2048, 2048, 1024, 1024, 512, 256, 128, 64, 32], 
            output_size     = 2, 
            activation      = "LeakyReLU", 
            dropout         = 0.2
        )
                
        for param in self.clinic_model.parameters():
            param.requires_grad = False
                
        self.classifier = MLP(
            input_size      = 4096,
            hidden_layers   = [2048, 1024, 256, 128],
            output_size     = 2,
            activation      = "Gelu",
            dropout         = 0.2
        )
    
    def forward(self, image, clinic_data):
        image_features      = self.image_model(image)
        clinic_features     = get_hidden_layer_features(self.clinic_model, clinic_data, 2)
        combined_features   = torch.cat((image_features, clinic_features), dim=1)
        output              = self.classifier(combined_features)
        return output
    
    



class ResNetModel(nn.Module):
    """
    Model class for ResNet50 architecture.
    This class initializes the ResNet50 model without the final fully connected layer.
    It uses the torchvision implementation of ResNet50.
    The model is designed to be used as a backbone for further classification tasks.
    """

    def __init__(self):
        super(ResNetModel, self).__init__()
        
        base_model      = resnet50(pretrained=False)
        self.features   = base_model.fc.in_features
        encoder_layers  = list(base_model.children())
        self.backbone   = nn.Sequential(*encoder_layers[:9])

    
    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        return out


class MLP(nn.Module):
    def __init__(
            self, 
            input_size:int      = 20, 
            hidden_layers: list = [64, 32],
            output_size: int    = 2, 
            activation:str      = "ReLU", 
            dropout: float      = 0.5
        ):
        super(MLP, self).__init__()
        
        """
        This is a simple Multi-layer perceptron (MLP) model.
        
        Args:
            input_size (int): The size of the input layer.
            hidden_layers (list): A list of integers representing the sizes of the hidden layers.
            output_size (int): The size of the output layer.
            activation (str): The activation function to use in the hidden layers.
            dropout (float): The dropout rate to use in the hidden layers.
            
        Returns:
            nn.Module: The MLP model.
        """
        
        layers      = []
        prev_size   = input_size

        for h in hidden_layers:
            layers.append(nn.Linear(prev_size, h))
            layers.append(get_activation(activation))

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_size = h

        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
