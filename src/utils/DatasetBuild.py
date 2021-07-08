import random
import sys
import torch
import tqdm
sys.path.append('../')
from models import DataModule   # noqa
from Config import DataConfig   # noqa

DirtyProtein = ['APPIQSRIIGGRECEKNSHPWQVAIYHYSSFQCGGVLVNPKWVLTAAHCKNDNYEVWLGRHNLFENENTAQFFGVTADFPHPGFNLSLLKXHTKADGKDYSHDLMLLRLQSPAKITDAVKVLELPTQEPELGSTCEASGWGSIEPGPDBFEFPDEIQCVQLTLLQNTFCABAHPBKVTESMLCAGYLPGGKDTCMGDSGGPLICNGMWQGITSWGHTPCGSANKPSIYTKLIFYLDWINDTITENP', 'HLLDFRKMIRYTTGKEATTSYGAYGCHCGVGGRGAPKXAKFLSYKFSMKKAAAACFKYQFYPNNRCXG'] # noqa

TrainValRate = 0.9
random.seed(1)
device = torch.device('cuda:0')
DataSetSmall = '../../../CPI_TrainSet/small_data.csv'
DataSetLarge = '../../../CPI_TrainSet/large_data.csv'


def read(name):
    with open(name, 'r') as fh:
        content = [line.strip() for line in list(fh)]
    return content


def parser_dataset(name):
    print('DataSet saving in %s' %
          DataConfig[DataConfig['DataType']]['embed_path'])
    raw_data = read(name)
    dataset = dict()
    pbar = tqdm.tqdm(total=len(raw_data))
    feat_generator = DataModule.DataModule(config=DataConfig, device=device)
    DataType = DataConfig['DataType']

    n = 1
    for data in raw_data:
        smiles, protein, label = data.strip().split(',')
        if not DataConfig[DataType]['UseMolFeat']:
            moleculeEmbedding, proteinEmbedding = feat_generator(
                smiles, protein)
        else:
            moleculeEmbedding, proteinEmbedding, moleculeFeat = feat_generator(
                smiles, protein)

        if protein not in DirtyProtein:
            dataset[n] = {
                 'smiles': smiles,
                 'protein': protein,
                 'label': int(label)
            }
            embeds = {
                'mol_embed': moleculeEmbedding.half().to('cpu'),
                'pro_embed': proteinEmbedding.half().to('cpu'),
            }
            if DataConfig[DataType]['UseMolFeat']:
                embeds['mol_feat'] = moleculeFeat

        torch.save(
            embeds, '%s/%d_embeds' % (DataConfig[DataType]['embed_path'], n))
        n += 1
        pbar.update()

    return dataset


dataset = parser_dataset(DataSetSmall)

DataType = DataConfig['DataType']
all_file = DataConfig[DataType]['data_path']

torch.save(dataset, all_file)
