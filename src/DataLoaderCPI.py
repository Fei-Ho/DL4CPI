import torch
from torch.utils import data as D
from models.Common import generate_molecule_features
from Config import DataConfig


class CPI_data(D.Dataset):
    def __init__(self, config, partition):
        self.config = config
        assert partition in ['train', 'val'], \
            "please specify partition(train, val)" # noqa
        data_path = self.config['data_path'] + '_' + partition + '.pt'
        self.dataset = torch.load(data_path)
        self.embed_dir = self.config['embed_path']
        self.avg_embd = self.config['embed_avg']

        self.UseMolFeat = self.config['UseMolFeat']
        self.id_list = list(self.dataset.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        data_id = self.id_list[index]
        molecule = self.dataset[data_id]['smiles']
        protein = self.dataset[data_id]['protein']
        embeds = torch.load(self.embed_dir + '/' + str(data_id) + '_embeds')
        molecule_embed = embeds['mol_embed']
        protein_embed = embeds['pro_embed']
        label = torch.tensor(self.dataset[data_id]['label']).float()

        # Embed need to calcualte the mean
        if self.avg_embd:
            protein_embed = torch.mean(protein_embed, axis=0)
        if self.config['moleculeEmbedding'] == 'MAT':
            molecule_embed = torch.mean(molecule_embed[0], axis=0)

        # molecule_feature
        if self.UseMolFeat:
            protein_feat = embeds['pro_embed'].T
            pro_padder = torch.nn.ConstantPad1d(
                (0, self.pro_pad - protein_feat.size()[1]), 0)
            protein_feat = pro_padder(protein_feat)
            if 'mol_feat' in embeds:
                molecule_feature = embeds['mol_feat']
            else:
                molecule_feature = generate_molecule_features(molecule)

            return molecule, molecule_feature, molecule_embed, \
                protein, protein_feat, protein_embed, label
        else:
            return molecule, 0, molecule_embed, \
                protein, 0, protein_embed, label


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import tqdm
    dset = CPI_data(config=DataConfig['Morgan+ESM'], partition='train')
    loader = DataLoader(dset, batch_size=1, num_workers=10)
    mol_len = []
    pbar = tqdm.tqdm(total=len(loader))
    for molecule, molecule_feat, molecule_embed, \
            protein, protein_feat, protein_embed, label in loader:
        pbar.update()
        mol_len.append(molecule_feat.size()[0])
