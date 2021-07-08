import torch
from torch import nn

from embeddings import D_MOLECULE_EMBEDDING, D_PROTEIN_EMBEDDING


class Baseline(nn.Module):
    def __init__(self, d_molecule_embedding: int, d_protein_embedding: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_molecule_embedding + d_protein_embedding, hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, molecule_embedding, protein_embedding):
        return self.net(torch.cat([molecule_embedding, protein_embedding], dim=-1))


def create_default_baseline_model():
    return Baseline(D_MOLECULE_EMBEDDING, D_PROTEIN_EMBEDDING, 256)
