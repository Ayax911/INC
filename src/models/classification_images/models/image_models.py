from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    inception_v3, Inception_V3_Weights,
)
from torch import nn
import torch

def get_image_model(
        model_name:str          = "ResNet",
        weigths_file:str        = None,
        num_freeze:int          = 0,
        pretrained:bool         = False) -> nn.Module:
    """
    Function to get the model based on the given name.

    Args:
        model_name (str)        : Name of the model to be retrieved.
        weigths_file (str)      : Path to the weights file to load into the model.
        freeze_backbone (bool)  : Whether to freeze the backbone of the model or not.
        pretrained (bool)       : Si es True, inicializa el backbone con los pesos ImageNet de torchvision.
            Si además se entrega `weigths_file`, ese checkpoint se carga después y sobreescribe los pesos ImageNet.
    Returns:
        torch.nn.Module: The model class corresponding to the given name.
    """

    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if(model_name == "ResNet"):
        base_model      = ResNetModel(pretrained=pretrained)
    elif(model_name == "ResNet18"):
        base_model      = ResNet18Model(pretrained=pretrained)
    elif(model_name == "DenseNet"):
        base_model      = DenseNetModel(pretrained=pretrained)
    elif(model_name == "Inception"):
        base_model      = InceptionModel(pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not supported. Please choose from 'ResNet', 'ResNet18', 'DenseNet', or 'Inception'.")
    
    # Si los pesos son proporcionados, cargarlos en el modelo base
    if weigths_file is not None:
        if not isinstance(weigths_file, str):
            raise ValueError("Weights file must be a string path to the weights file.")
        else:
            base_model.load_state_dict(torch.load(weigths_file, map_location=device))
    
    # Congelar las capas del backbone si freeze_backbone es True
    if num_freeze > 0:
        base_model = freeze_layers(base_model, num_freeze=num_freeze)

    return base_model

def freeze_layers(model, num_freeze):
    """
    Congela las primeras `num_freeze` capas (parámetros) del modelo.
    Parámetros posteriores siguen entrenables.
    """

    # Lista ordenada de todos los parámetros entrenables
    params = list(model.parameters())

    # Asegurar límites
    num_freeze = min(num_freeze, len(params))

    print("----------------------------------------------")
    print(f"Congelando las primeras {num_freeze} capas")
    print("----------------------------------------------")
    # Congelar primeras N capas
    for i in range(num_freeze):
        print(f"Congelando capa {params[i]}")
        params[i].requires_grad = False

    print("----------------------------------------------")
    print(f"Reuiere Grad las capas {num_freeze}")
    print("----------------------------------------------")
    # Descongelar el resto
    for i in range(num_freeze, len(params)):
        print(f"Descongelando capa {params[i]}")
        params[i].requires_grad = True

    return model


class ResNetModel(nn.Module):
    """
    Model class for ResNet50 architecture.
    This class initializes the ResNet50 model without the final fully connected layer.
    It uses the torchvision implementation of ResNet50.
    The model is designed to be used as a backbone for further classification tasks.
    """

    def __init__(self, pretrained: bool = False):
        super(ResNetModel, self).__init__()

        weights         = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base_model      = resnet50(weights=weights)
        self.features   = base_model.fc.in_features
        encoder_layers  = list(base_model.children())
        self.backbone   = nn.Sequential(*encoder_layers[:9])

    
    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        return out

class ResNet18Model(nn.Module):
    """
    Model class for ResNet18 architecture.
    This class initializes the ResNet18 model without the final fully connected layer.
    It uses the torchvision implementation of ResNet18 (512 features tras el global pooling).
    The model is designed to be used as a backbone for further classification tasks.
    """

    def __init__(self, pretrained: bool = False):
        super(ResNet18Model, self).__init__()

        weights         = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base_model      = resnet18(weights=weights)
        self.features   = base_model.fc.in_features
        encoder_layers  = list(base_model.children())
        self.backbone   = nn.Sequential(*encoder_layers[:9])

    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        return out

class DenseNetModel(nn.Module):
    """
    This class initializes the DenseNet121 model without the final fully connected layer.
    It uses the torchvision implementation of ResNet50.
    The model is designed to be used as a backbone for further classification tasks.
    """

    def __init__(self, pretrained: bool = False):
        super(DenseNetModel, self).__init__()

        weights             = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        base_model          = densenet121(weights=weights)
        encoder_layers      = list(base_model.children())
        self.backbone       = nn.Sequential(*encoder_layers[:-1])
        self.global_pool    = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        out = self.backbone(x)
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        return out


class InceptionModel(nn.Module):
    """
    This class initializes the InceptionV3 model without the final fully connected layer.
    It uses the torchvision implementation of Inception.
    The model is designed to be used as a backbone for further classification tasks.
    """

    def __init__(self, pretrained: bool = False):
        super(InceptionModel, self).__init__()

        # Los pesos ImageNet de torchvision solo se pueden cargar con aux_logits=True;
        # se reemplaza esa rama por Identity para no romper el ensamblado secuencial del backbone.
        weights             = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        base_model          = inception_v3(weights=weights, aux_logits=pretrained)
        if pretrained:
            base_model.AuxLogits = nn.Identity()
        encoder_layers      = list(base_model.children())
        self.backbone       = nn.Sequential(*encoder_layers[:-1])
    
    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        return out


if __name__ == "__main__":
    model_name = "ResNet"
    path_weigths = "/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/13-PregradoJulian/Federal Learning/infraestructura federada/FedMammoBench/weights/ResNet50.pt"  # Path to the weights file if needed
    model = get_image_model(model_name, weigths_file=path_weigths, freeze_backbone=True)
    print(model)
    
    # Test the model with a random input
    x = torch.randn(1, 3, 224, 224)  # Example input tensor
    output = model(x)
    print(output.shape)  # Should match the number of classes