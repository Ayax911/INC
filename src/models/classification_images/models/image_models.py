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
    elif(model_name == "CustomCNN"):
        base_model      = CustomCNNModel(pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not supported. Please choose from 'ResNet', 'ResNet18', 'DenseNet', 'Inception', or 'CustomCNN'.")

    # CustomCNN no tiene pesos preentrenados que cargar -- a diferencia de las demás
    # arquitecturas, weigths_file nunca debería llegar no-None acá: si el caller no pasó
    # --pretrained ni --from_scratch, get_model.py le pasaría el checkpoint default
    # (ResNet50.pt), que no calza con esta arquitectura y fallaría con un error de shape
    # confuso dentro de load_state_dict(). Se corta antes, con un mensaje claro.
    if model_name == "CustomCNN" and weigths_file is not None:
        raise ValueError("CustomCNN no admite --path_image_model (no tiene pesos preentrenados): usa --from_scratch.")

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


class CustomCNNModel(nn.Module):
    """
    Backbone CNN custom (diseño propio, sin pesos preentrenados): 4 bloques de
    Conv2d(3x3, padding="same") -> BatchNorm2d -> ReLU repetido 2 veces, con
    MaxPool2d(2) al final de los bloques 1-3 (canales 32/64/128/256; el bloque
    4 no reduce resolución) + AdaptiveAvgPool2d(1) (GlobalAveragePooling2D).
    Entrega 256 features tras el pooling global. Réplica en este repo de
    FedMammoBench/src/models/custom_cnn.py::CustomCNNBackbone (mismo diseño,
    código paralelo -- cada repo es autocontenido, no hay import cruzado).
    """

    def __init__(self, pretrained: bool = False):
        super(CustomCNNModel, self).__init__()

        if pretrained:
            raise ValueError("CustomCNN no tiene pesos preentrenados: --pretrained no aplica.")

        def conv_bn_relu(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        block1 = nn.Sequential(conv_bn_relu(3, 32), conv_bn_relu(32, 32), nn.MaxPool2d(2))
        block2 = nn.Sequential(conv_bn_relu(32, 64), conv_bn_relu(64, 64), nn.MaxPool2d(2))
        block3 = nn.Sequential(conv_bn_relu(64, 128), conv_bn_relu(128, 128), nn.MaxPool2d(2))
        # Bloque 4 sin MaxPool2d -- por diseño, termina en 32x32x256 (para input 256x256).
        block4 = nn.Sequential(conv_bn_relu(128, 256), conv_bn_relu(256, 256))
        gap     = nn.AdaptiveAvgPool2d(1)

        self.features   = 256
        self.backbone   = nn.Sequential(block1, block2, block3, block4, gap)

    def forward(self, x):
        out = self.backbone(x)
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