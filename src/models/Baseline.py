import torch
from torch import nn
from .Common import activation_func


# Baseline model,
# config:
#     n_input: input size
#     n_inner: hidden layer size
#     normalization: noralization layer
#     activation: activation layer
class Baseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config['n_input'], config['n_inner']),
            activation_func(config['activation']),
            nn.Dropout(0.1),
            nn.Linear(config['n_inner'], config['n_inner'] // 2),
            activation_func(config['activation']),
            nn.Dropout(0.1),
            nn.Linear(config['n_inner'] // 2, 1),
        )

    def forward(self, mol_feat, pro_feat, mol_embed, pro_embed):
        input_embed = torch.cat([mol_embed, pro_embed], axis=-1)
        return self.net(input_embed)
