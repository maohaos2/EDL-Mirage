import numpy as np
import torch.nn as nn
from edl_mirage.architectures.spectral_linear import SpectralLinear


def linear_sequential(
    input_dims, hidden_dims, output_dim, k_lipschitz=None, p_drop=None
):
    dims = [np.prod(input_dims)] + hidden_dims + [output_dim]
    num_layers = len(dims) - 1
    layers = []
    for i in range(num_layers):
        if k_lipschitz is not None:
            layer = SpectralLinear(dims[i], dims[i + 1], k_lipschitz ** (1.0 / num_layers))
            layers.append(layer)
        else:
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < num_layers - 1:
            layers.append(nn.ReLU())
            if p_drop is not None:
                layers.append(nn.Dropout(p=p_drop))
    return nn.Sequential(*layers)
