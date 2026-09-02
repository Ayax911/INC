from models.image_models import get_image_model
from torch import nn
import torch
from models.mlp_models import MLP

def get_hidden_layer_features(model, x, layer_idx):
    # Recorremos las capas del modelo
    for i, layer in enumerate(model.model.children()):
        x = layer(x)
        if i == layer_idx:
            return x  # Devolver la capa en el índice especificado
    return x


class MLP_Final_Model(nn.Module):
    def __init__(self, options):
        super(MLP_Final_Model, self).__init__()
        
        self.options = options
        
        self.image_model = get_image_model(
            model_name      = options.image_model,
            weigths_file    = options.path_image_model,
        )
        
        self.clinic_model = MLP(
            input_size      = options.clinic_input_size, 
            hidden_layers   = options.clinic_hidden_layers, 
            output_size     = 2, 
            activation      = options.clinic_activation, 
            dropout         = options.clinic_dropout
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clinic_model.load_state_dict(torch.load(options.path_clinic_model, map_location=device))
        
        for param in self.clinic_model.parameters():
            param.requires_grad = False
                
        self.classifier = MLP(
            input_size      = options.final_input_size,
            hidden_layers   = options.final_hidden_layers,
            output_size     = options.final_output_size,
            activation      = options.final_activation,
            dropout         = options.final_dropout
        )
    
    def forward(self, image, clinic_data):
        image_features      = self.image_model(image)
        clinic_features     = get_hidden_layer_features(self.clinic_model, clinic_data, self.options.clinic_idx_hidden_layer)
        combined_features   = torch.cat((image_features, clinic_features), dim=1)
        output = self.classifier(combined_features)
        return output
