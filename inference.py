import torch
import src.models.DataModule as DataModule
from src.Config import DataConfig, ModelConfig   # noqa
from src.models import Embed_mlp, Baseline, CPInet  # noqa


DEVICE = 'cpu'
FEAT_GEN = DataModule.DataModule(config=DataConfig, device=DEVICE)
DataType = DataConfig['DataType']

if ModelConfig['model'] == 'Baseline':
    MODEL_PATH = 'trained_models/BaseLine/Baseline.pt'
    MODEL = Baseline.Baseline(ModelConfig['Baseline']).to(DEVICE)

elif ModelConfig['model'] == 'Embed_mlp':
    if DataConfig[DataType]['moleculeEmbedding'] == 'RDKFingerprint' and \
            DataConfig[DataType]['proteinEmbedding'] == 'ESM':
        MODEL_PATH = 'trained_models/Embed_mlp/RDK_ESM.pt'
    elif DataConfig[DataType]['moleculeEmbedding'] == 'MAT' and \
            DataConfig[DataType]['proteinEmbedding'] == 'ESM':
        MODEL_PATH = 'trained_models/Embed_mlp/MAT_ESM.pt'
    elif DataConfig[DataType]['moleculeEmbedding'] == 'ChemBERT' and \
            DataConfig[DataType]['proteinEmbedding'] == 'ESM':
        MODEL_PATH = 'trained_models/Embed_mlp/ChemBERT.pt.best'
    if DataConfig[DataType]['moleculeEmbedding'] == 'AllChemBERT' and \
            DataConfig[DataType]['proteinEmbedding'] == 'AllESM':
        MODEL_PATH = 'trained_models/Embed_mlp/ChemBERT_All.pt.best'
    MODEL = Embed_mlp.Embed_mlp(ModelConfig['Embed_mlp']).to(DEVICE)

elif ModelConfig['model'] == 'CPInet':
    MODEL_PATH = 'trained_models/CPInet/CPInet_esm_morgan_LSTM.pt'
    MODEL = CPInet.CPInet(ModelConfig['CPInet']).to(DEVICE)

ProteinDict = dict()
MolDict = dict()
STATE_DICT = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
MODEL.load_state_dict(STATE_DICT['state_dict'])
MODEL.eval()


def inference(molecule_smiles: str, protein_fasta: str) -> float:
    if protein_fasta not in ProteinDict:
        proteinFeat = FEAT_GEN.get_protein_vec(protein_fasta)
        ProteinDict[protein_fasta] = proteinFeat
    else:
        proteinFeat = ProteinDict[protein_fasta]

    if molecule_smiles not in MolDict:
        moleculeEmbedding = FEAT_GEN.get_molecule_vec(molecule_smiles).half()
        moleculeFeat = FEAT_GEN.get_molecule_feat(molecule_smiles).half()
        MolDict[molecule_smiles] = (moleculeEmbedding, moleculeFeat)
    else:
        moleculeEmbedding, moleculeFeat = MolDict[molecule_smiles]

    if DataConfig[DataConfig['DataType']]['embed_avg']:
        proteinEmbedding = torch.mean(proteinFeat.float(), axis=0)
        if DataConfig[DataConfig['DataType']]['moleculeEmbedding'] == 'MAT':
            moleculeEmbedding = torch.mean(moleculeEmbedding.float(), axis=0)
    else:
        proteinEmbedding = proteinFeat.float()

    out = torch.sigmoid(
        MODEL(mol_embed=moleculeEmbedding.unsqueeze(0).float().to(DEVICE),
              pro_embed=proteinEmbedding.unsqueeze(0).float().to(DEVICE),
              mol_feat=moleculeFeat.unsqueeze(0).float().to(DEVICE),
              pro_feat=proteinFeat.unsqueeze(0).float().to(DEVICE)))
    return out.item()
