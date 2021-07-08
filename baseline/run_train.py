import os

import tqdm
import torch
from torch import optim
from torch.nn.functional import binary_cross_entropy
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from embeddings import get_molecule_vec, get_protein_vec
from model import create_default_baseline_model


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class CPIDataset(Dataset):
    def __init__(self, file, limit=None):
        self.samples = []
        with open(file, 'r') as fd:  # Row Format: SMILES Protein Label
            for line in tqdm.tqdm(fd.readlines(), total=limit,
                                  desc=f'Loading dataset from "{os.path.basename(file)}"'):
                smiles, protein, label = line.strip().split(',')
                self.samples.append((
                    get_molecule_vec(smiles),
                    get_protein_vec(protein),
                    torch.tensor([float(label)])
                ))
                if limit is not None and len(self.samples) >= limit:
                    break

    @staticmethod
    def collate(samples):
        return map(torch.stack, zip(*samples))

    def create_data_loader(self, **kwargs):
        return DataLoader(self, collate_fn=CPIDataset.collate, **kwargs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def main():
    model = create_default_baseline_model()
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.05)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    train_set = CPIDataset('../data/small_data.csv', limit=5000)  # we only use first 5000 samples here for test.
    train_loader = train_set.create_data_loader(batch_size=512, shuffle=True, drop_last=True)

    for i in range(100):
        with tqdm.tqdm(train_loader, desc=f'Epoch {i + 1}') as epoch_loader:
            for molecule, protein, true_y in epoch_loader:
                pred_y = model(molecule.to(DEVICE), protein.to(DEVICE))
                loss = binary_cross_entropy(pred_y, true_y.to(DEVICE).float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loader.set_postfix(loss=f'{loss.item():.4f}')
        scheduler.step()

    torch.save(model.state_dict(), 'baseline_model.pkl')
    print('Baseline model saved to "./baseline_model.pkl"')


if __name__ == '__main__':
    main()
