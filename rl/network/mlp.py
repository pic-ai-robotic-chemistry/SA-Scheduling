# https://github.com/zcaicaros/L2D for reference

import torch
import torch.nn as nn


class TanhMLP(nn.Module):
    """
    A multilayer perceptron with tanh activation and no BatchNorm.

    Actor and Critic share the same architecture; constructor arguments determine their dimensions.
    When num_layers=1, the model reduces to a linear layer.

    Args:
        num_layers: Total number of linear layers (>=1); 1 indicates a linear model
        input_dim: Input feature dimension
        hidden_dim: Hidden layer width (shared by all hidden layers)
        output_dim: Output dimension
    """

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_layers = num_layers

        if num_layers == 1:
            self.is_linear = True
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.is_linear = False
            self.linears = nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.is_linear:
            return self.linear(x)
        h = x
        for layer in self.linears[:-1]:
            h = torch.tanh(layer(h))
        return self.linears[-1](h)



MLPActor = TanhMLP
MLPCritic = TanhMLP
