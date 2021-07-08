import numpy as np
import torch
np.random.seed(100)
data_path = '/lustre/S/wufandi/Project/CPI/CPI_Baseline/dataset/CPI2021JUN30'
train_path = '/lustre/S/wufandi/Project/CPI/CPI_Baseline/dataset/CPI2021JUL07_train.pt'
val_path = '/lustre/S/wufandi/Project/CPI/CPI_Baseline/dataset/CPI2021JUL07_val.pt'


def train_val_split():
    val_ratio = 0.1
    dataset = torch.load(data_path)
    smile_dict=dict()
    for data_id in dataset:
        smiles = dataset[data_id]['smiles']
        if smiles in smile_dict:
            smile_dict[smiles].append(data_id)
        else:
            smile_dict[smiles] = [data_id]

    smile_list = list(smile_dict.keys())
    np.random.shuffle(smile_list)
    val_index = int(val_ratio * len(smile_list))
    val_seeds = smile_list[:val_index]
    train_seeds = smile_list[val_index:]

    val_ids = []
    train_ids = []
    for smile in val_seeds:
        val_ids.extend(smile_dict[smile])
    for smile in train_seeds:
        train_ids.extend(smile_dict[smile])

    train_dict = dict()
    val_dict = dict()
    for data_id in train_ids:
        train_dict[data_id] = dataset[data_id]
    for data_id in val_ids:
        val_dict[data_id] = dataset[data_id]

    torch.save(train_dict, train_path)
    torch.save(val_dict, val_path)




if __name__ == "__main__":
    train_val_split()
