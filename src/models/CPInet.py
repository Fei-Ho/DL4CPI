from torch import nn
import torch

class CPInet(nn.Module):
    def __init__(self, config):
        super().__init__()
        esm_dim = config['esm_dim']
        mol_dim = config['mol_dim']
        mol_feat_dim = config['mol_feat_dim']
        self.kernel_size = config['kernel_size']
        ''' 
        self.ProteinCNN = nn.Sequential(
            nn.Conv1d(in_channels = esm_dim, out_channels=512, kernel_size = self.kernel_size,
                      padding = int((self.kernel_size - 1) / 2)),
            nn.ReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels = 512, out_channels= 1024, kernel_size = self.kernel_size,
                      padding = int((self.kernel_size - 1) / 2)),
            nn.ReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels = 1024, out_channels= 1024, kernel_size = self.kernel_size,
                      padding = int((self.kernel_size - 1) / 2)),
            nn.ReLU(inplace = True),
            nn.Dropout(0.2)
        )

        self.MoleculeCNN = nn.Sequential(
            nn.Conv1d(in_channels = mol_feat_dim, out_channels= 64, kernel_size = self.kernel_size,
                      padding = int((self.kernel_size - 1) / 2)),
            nn.ReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels = 64, out_channels= 256, kernel_size = self.kernel_size,
                      padding = int((self.kernel_size - 1) / 2)),
            nn.ReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels = 256, out_channels= 512, kernel_size = self.kernel_size,
                      padding = int((self.kernel_size - 1) / 2)),
            nn.ReLU(inplace = True),
            nn.Dropout(0.2)
        )
        '''

        self.ProteinLSTM = nn.LSTM(input_size = esm_dim, hidden_size = 512, 
                                    num_layers = 2, batch_first = True, dropout = 0.3, bidirectional = True)
        self.MoleculeLSTM = nn.LSTM(input_size = mol_feat_dim, hidden_size = 512,
                                    num_layers = 2, batch_first = True, dropout = 0.3, bidirectional = True)

        self.classifier = nn.Sequential(
            nn.Linear(esm_dim + mol_dim + 2048 + 2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 1)
        )

    def forward(self, mol_feat, pro_feat, mol_embed, pro_embed):
        mol_h = self.MoleculeLSTM(mol_feat)[1][0]
        pro_h = self.ProteinLSTM(pro_feat)[1][0]
        mol_h = mol_h.view(mol_h.size(1), mol_h.size(0) * mol_h.size(2))
        pro_h = mol_h.view(pro_h.size(1), pro_h.size(0) * pro_h.size(2))
        

        input_embed = torch.cat([mol_embed, pro_embed, mol_h, pro_h], axis=-1)
        out = self.classifier(input_embed)
        return out

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import tqdm
    import numpy as np
    import sys
    sys.path.append('../')
    import Config
    import DataLoaderCPI
    
    device = torch.device('cuda:0')
    dset = DataLoaderCPI.CPI_data(config = Config.DataConfig['Morgan+ESM'],partition = 'train')
    model = CPInet(config = Config.ModelConfig['CPInet']).to(device)
    loader = DataLoader(dset, batch_size = 1, num_workers = 10)
    mol_len = []
    pbar = tqdm.tqdm(total=len(loader))
    for molecule, molecule_feat, molecule_embed, protein, protein_feat, protein_embed, label in loader:
        pbar.update()
        out = model(molecule_feat.float().to(device), protein_feat.float().to(device), molecule_embed.float().to(device), 
                    protein_embed.float().to(device))

