from torch import nn
import torch
from torch.utils.data import DataLoader
from .Common import LayerNorm, ScaleNorm


class Embed_mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        esm_dim = config['esm_dim']
        mol_dim = config['mol_dim']
        n_inner = config['n_inner']
        if config['norm'] == "LayerNorm":
            norm = LayerNorm
        elif config['norm'] == "ScaleNorm":
            norm = ScaleNorm
        N = config['N']
        classifier = []
        classifier.append(nn.Linear(esm_dim+mol_dim, n_inner))
        classifier.append(nn.LeakyReLU())
        classifier.append(norm(n_inner))
        classifier.append(nn.Dropout(0.1))

        for i in range(N-1):
            classifier.append(nn.Linear(n_inner, n_inner))
            classifier.append(nn.LeakyReLU())
            classifier.append(norm(n_inner))
            classifier.append(nn.Dropout(0.1))

        classifier.append(nn.Linear(n_inner, 1))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, mol_feat, pro_feat, mol_embed, pro_embed):
        if mol_embed.dim() == 3:
            mol_embed = mol_embed.reshape((mol_embed.shape[0], -1))
            pro_embed = pro_embed.reshape((pro_embed.shape[0], -1))
        input_embed = torch.cat([mol_embed, pro_embed], axis=-1)
        out = self.classifier(input_embed)
        return out


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import DataLoaderCPI
    import tqdm
    import Config

    dset = DataLoaderCPI.CPI_data(
        config=Config.DataConfig, partition='train')
    loader = DataLoader(dset, batch_size=64, num_workers=10)
    device = torch.device('cuda:0')
    model = Embed_mlp(config=Config.ModelConfig['Embed_mlp']).to(device)
    pbar = tqdm.tqdm(total=len(loader))
    for molecule, molecule_feat, molecule_embed, protein, protein_feat, \
            protein_embed, label in loader:
        pbar.update()
        pro_embed, mol_embed = protein_embed.to(device), \
            molecule_embed.to(device)
        out = model(mol_embed=mol_embed, pro_embed=pro_embed,
                    pro_feat=None, mol_feat=None)
