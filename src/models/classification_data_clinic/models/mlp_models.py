import torch
import torch.nn as nn

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
            layers.append(nn.BatchNorm1d(h))
            layers.append(get_activation(activation))

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_size = h

        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


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


if __name__ == "__main__":
    # Example usage
    input_size = 20
    hidden_layers = [64, 32]
    output_size = 2
    activation = "ReLU"
    normalization = None  # Not used in this example
    dropout = 0.5

    model = MLP(input_size, hidden_layers, output_size, activation, normalization, dropout)
    print(model)

    # Test the model with a random input
    x = torch.randn(1, input_size)  # Example input tensor
    output = model(x)
    print(output)