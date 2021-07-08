import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import esm
from gensim.models import Word2Vec
from .Common import atom_features, featurize_mol, generate_molecule_features
from .MAT import make_model
from transformers import AutoModelForMaskedLM, AutoTokenizer


class DataModule(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        DataType = config['DataType']
        self.config = config[DataType]
        self.device = device

        # Loading the molecule embedding
        if self.config['moleculeEmbedding'] == 'RDKFingerprint':
            print('Use RDKFingerprint for moleculeEmbedding')
        elif self.config['moleculeEmbedding'] == 'MorganFingerprint':
            print('Use MorganFingerprint for moleculeEmbedding')
        elif self.config['moleculeEmbedding'] == 'MAT':
            print('Use MAT for moleculeEmbedding')
            self.MAT = make_model(28, N=4, h=4)
            pretrained_state_dict = torch.load(
                '../external/pretrained_weights.pt',
                map_location=torch.device('cpu'))
            model_state_dict = self.MAT.state_dict()
            for name, param in pretrained_state_dict.items():
                if 'generator' in name:
                    continue
                if isinstance(param, torch.nn.Parameter):
                    param = param.data
                    model_state_dict[name].copy_(param)
            self.MAT.to(device)
            self.MAT.eval()
        elif self.config['moleculeEmbedding'] == "ChemBERT" or \
                self.config['moleculeEmbedding'] == "AllChemBERT" or \
                self.config['moleculeEmbedding'] == "ChemBERTout":
            print('Use %s for moleculeEmbedding' %
                  (self.config['moleculeEmbedding']))
            self.model = AutoModelForMaskedLM.from_pretrained(
                # "/lustre/S/wufandi/Project/CPI/CPI_Baseline/"
                # "external/PubChem10M_SMILES_BPE_120k")
                './external/PubChem10M_SMILES_BPE_120k')
            self.tokenizer = AutoTokenizer.from_pretrained(
                # "/lustre/S/wufandi/Project/CPI/CPI_Baseline/"
                # "external/PubChem10M_SMILES_BPE_120k")
                './external/PubChem10M_SMILES_BPE_120k')
            self.model.to(device)
            self.model.eval()

        self.UseMolFeat = self.config['UseMolFeat']

        # Loading the protein embedding model
        if self.config['proteinEmbedding'] == 'transformerCPI':
            print('Use transformerCPI for proteinEmbedding')
            self.W2V_MODEL = Word2Vec.load('./baseline/word2vec_30.model')
        elif self.config['proteinEmbedding'] == 'ESM' or \
                self.config['proteinEmbedding'] == 'AllESM':
            print('Use %s for proteinEmbedding' %
                  (self.config['proteinEmbedding']))
            # this path also needs to be configurable
            self.ESM_MODEL, self.alphabet = \
                esm.pretrained.load_model_and_alphabet_local(
                    './external/checkpoints/esm1_t12_85M_UR50S.pt')
            # "/lustre/S/wufandi/Project/CPI/CPI_Baseline"
            # "/external/checkpoints/esm1_t12_85M_UR50S.pt")
            self.batch_converter = self.alphabet.get_batch_converter()
            self.ESM_MODEL.to(device)

    '''func for gneerating molecular fix-length embeddings'''
    def get_molecule_vec(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if self.config['moleculeEmbedding'] == 'RDKFingerprint':
            fingerprint = Chem.RDKFingerprint(mol, fpSize=2048)
            output = fingerprint
            return torch.tensor(output, dtype=torch.float,
                                device=self.device)
        elif self.config['moleculeEmbedding'] == 'atom_feature':
            atom_feat = np.zeros((mol.GetNumAtoms(), 34))
            for atom in mol.GetAtoms():
                atom_feat[atom.GetIdx(), :] = atom_features(atom)
            return torch.tensor(atom_feat, dtype=torch.float,
                                device=self.device)
        elif self.config['moleculeEmbedding'] == 'MAT':
            try:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, maxAttempts=5000, randomseed=233)
                AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
            except: # noqa
                AllChem.Compute2DCoords(mol)
            node_features, adj_matrix, dist_matrix = \
                featurize_mol(mol, True, True)
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            return self.MAT(node_features.to(self.device),
                            batch_mask.to(self.device),
                            adj_matrix.to(self.device),
                            dist_matrix.to(self.device), None)[0]
        elif self.config['moleculeEmbedding'] == 'MorganFingerprint':
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=3, nBits=1024)
            output = fingerprint
            return torch.tensor(output, dtype=torch.float, device=self.device)
        elif self.config['moleculeEmbedding'] == 'RDK+Morgan':
            fingerprint1 = torch.tensor(
                AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=3, nBits=1024), dtype=torch.float,
                device=self.device)
            fingerprint2 = torch.tensor(
                Chem.RDKFingerprint(mol, fpSize=2048), dtype=torch.float,
                device=self.device)
            return torch.cat([fingerprint1, fingerprint2])
        elif self.config['moleculeEmbedding'] == 'ChemBERT':
            input = self.tokenizer.encode(smile, return_tensors='pt').to(
                self.device)
            token_logits = self.model(input, output_hidden_states=True)
            molecule_embed = torch.mean(
                token_logits.hidden_states[-1][0], axis=0).detach()
            return molecule_embed
        elif self.config['moleculeEmbedding'] == 'AllChemBERT':
            encoded_input = self.tokenizer.encode(
                smile, return_tensors='pt').to(self.device)
            model_out = self.model(encoded_input, output_hidden_states=True)
            molecule_embed = torch.stack(model_out.hidden_states, dim=1)[0]
            molecule_embed = torch.mean(
                molecule_embed.permute(1, 0, 2), axis=0).detach()
            return molecule_embed
        elif self.config['moleculeEmbedding'] == 'ChemBERTout':
            encoded_input = self.tokenizer.encode(
                smile, return_tensors='pt').to(self.device)
            model_out = self.model(encoded_input)[0]
            molecule_embed = torch.mean(model_out[0], axis=0).detach()
            return molecule_embed

    '''func for generating molecular atom embeddings'''
    def get_molecule_feat(self, smile):
        return torch.tensor(generate_molecule_features(smile),
                            dtype=torch.float, device=self.device)

    '''func for generating protein AA embeddings'''
    def get_protein_vec(self, protein):
        if self.config['proteinEmbedding'] == 'transformerCPI':
            k = 3
            vectors = torch.tensor([list(self.W2V_MODEL.wv[protein[i:i+k]])
                                    for i in range(len(protein) - k + 1)],
                                   dtype=torch.float)
            output = torch.cat([vectors.mean(0), vectors.max(0)[0]])
            return output

        if self.config['proteinEmbedding'] == 'ESM':
            seq_len = len(protein)
            if seq_len > 1000:
                protein = protein[:1000]
                seq_len = 1000
            data = [('', protein)]
            _, _, token = self.batch_converter(data)
            rep = self.ESM_MODEL(
                token.to(self.device), repr_layers=[12],
                return_contacts=False)['representations'][12][0, 1:seq_len + 1]
            rep = torch.mean(rep, axis=0).detach()
            return rep

        elif self.config['proteinEmbedding'] == 'AllESM':
            seq_len = len(protein)
            if seq_len > 1000:
                protein = protein[:1000]
                seq_len = 1000
            data = [('', protein)]
            _, _, token = self.batch_converter(data)
            rep = self.ESM_MODEL(
                token.to(self.device), repr_layers=list(range(1, 13)),
                return_contacts=False)['representations']
            rep_output = []
            for i in range(1, 13):
                rep_output.append(rep[i][0, 1:seq_len+1])
            rep_output = torch.stack(rep_output)
            rep_output = rep_output.permute(1, 0, 2)
            rep_output = torch.mean(rep_output, axis=0).detach()
            return rep_output

    def forward(self, smiles, protein):
        moleculeEmbedding = self.get_molecule_vec(smiles).detach().cpu()
        proteinEmbedding = self.get_protein_vec(protein).detach().cpu()
        if self.UseMolFeat:
            moleculeFeat = self.get_molecule_feat(smiles).detach().cpu()
            return moleculeEmbedding, moleculeFeat, proteinEmbedding
        else:
            return moleculeEmbedding, proteinEmbedding
