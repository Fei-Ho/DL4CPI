import logging

import torch

from embeddings import get_molecule_vec, get_protein_vec
from model import create_default_baseline_model


LOGGER = logging.getLogger(__file__)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASELINE_MODEL = create_default_baseline_model()
BASELINE_MODEL.load_state_dict(torch.load('baseline_model.pkl', map_location='cpu'))
BASELINE_MODEL.eval()
BASELINE_MODEL.to(DEVICE)


@torch.no_grad()
def inference(molecule_smiles: str, protein_fasta: str) -> float:
    LOGGER.debug(f'inference: smiles={molecule_smiles}, fasta={protein_fasta}')
    prob = float('nan')
    try:
        prob = BASELINE_MODEL(
            molecule_embedding=get_molecule_vec(molecule_smiles).view(1, -1).to(DEVICE),
            protein_embedding=get_protein_vec(protein_fasta).view(1, -1).to(DEVICE)
        ).item()
    except Exception as e:
        LOGGER.error(f'inference - failed: smiles={molecule_smiles}, fasta={protein_fasta}, error={e}')
    else:
        LOGGER.info(f'inference - success: smiles={molecule_smiles}, fasta={protein_fasta}, prob={prob}')
    return prob
