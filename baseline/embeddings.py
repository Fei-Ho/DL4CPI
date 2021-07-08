import torch
from rdkit import Chem
from gensim.models import Word2Vec


"""
This word2vec model is downloaded from https://github.com/lifanchen-simm/transformerCPI.
"""
W2V_MODEL = Word2Vec.load("word2vec_30.model")
D_MOLECULE_EMBEDDING = 2048
D_PROTEIN_EMBEDDING = 200


def get_molecule_vec(smiles):  # length: 2048
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = Chem.RDKFingerprint(mol, fpSize=D_MOLECULE_EMBEDDING)
    return torch.tensor(fingerprint, dtype=torch.float)


def get_protein_vec(protein, k=3):  # length: 200
    vectors = torch.tensor([list(W2V_MODEL.wv[protein[i:i+k]]) for i in range(len(protein) - k + 1)], dtype=torch.float)
    return torch.cat([vectors.mean(0), vectors.max(0)[0]])  # mean & max
