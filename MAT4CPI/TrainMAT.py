import pickle
import argparse
import numpy as np
import torch
from tqdm.auto import tqdm
import torch.optim as optim
from featurization.data_utils import construct_loader
from transformer import make_model
from torch.nn import DataParallel


DATA_PATH = '/lustre/S/wufandi/Project/CPI/CPI_Baseline/dataset/CPI2021JUN30'
EMB_PATH = '/lustre/S/wufandi/Project/CPI/CPI_Baseline/dataset/embeddings_MAT'


def train(param_dict, epoch, model_path):
    model = param_dict['model'].train()
    running_loss = []
    pbar = tqdm(total=len(param_dict['train_loader']))

    loss_fn = param_dict['loss_fn']
    train_loader = param_dict['train_loader']
    val_loader = param_dict['val_loader']
    optimizer = param_dict['optim']
    device = param_dict['device']

    scaler = torch.cuda.amp.GradScaler()

    for adjacency_matrix, node_features, distance_matrix, \
            protein_embedding, label in train_loader:
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        label = label.unsqueeze(1).to(device)
        with torch.cuda.amp.autocast():
            output = model(
                node_features, batch_mask, adjacency_matrix, distance_matrix,
                None, protein_embedding)
            loss = loss_fn(output, label)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss.append(loss.item())
        pbar.update()
        pbar.set_description(
            str(epoch) + '/' + str(param_dict['train_epoch']) + '  ' +
            str(np.mean(running_loss))[:6])

    val_loss = []
    model.eval()

    for adjacency_matrix, node_features, distance_matrix, \
            protein_embedding, label in val_loader:
        label = label.unsqueeze(1).to(device)
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        with torch.cuda.amp.autocast():
            output = model(
                node_features, batch_mask, adjacency_matrix, distance_matrix,
                None, protein_embedding)
            loss = loss_fn(output, label)
            val_loss.append(loss.item())

    pbar.set_description(str(epoch) + '/' + str(np.mean(val_loss))[:6])
    pbar.close()
    lr_scheduler.step()

    model_states = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": running_loss}
    torch.save(model_states, model_path,
               pickle_protocol=pickle.HIGHEST_PROTOCOL,
               _use_new_zipfile_serialization=False)
    return np.mean(val_loss)


if __name__ == "__main__":
    print('------------Starting------------' + '\n')

    parser = argparse.ArgumentParser(
        description='Training script for MetaChrom')
    parser.add_argument(
        '--Device', type=str, help='CUDA device for training', default='0')
    args = parser.parse_args()
    device = torch.device('cuda:' + args.Device)

    args = parser.parse_args()

    train_loader = construct_loader(
        data_path=DATA_PATH, embed_dir=EMB_PATH, partition='train',
        batch_size=128, add_dummy_node=True, one_hot_formal_charge=True,
        shuffle=True)
    val_loader = construct_loader(
        data_path=DATA_PATH, embed_dir=EMB_PATH, partition='val',
        batch_size=1, add_dummy_node=True, one_hot_formal_charge=True,
        shuffle=True)

    model_params = {
        'd_atom': 28,
        'd_model': 1024,
        'N': 8,
        'h': 16,
        'N_dense': 1,
        'lambda_attention': 0.33,
        'lambda_distance': 0.33,
        'leaky_relu_slope': 0.1,
        'dense_output_nonlinearity': 'relu',
        'distance_matrix_kernel': 'exp',
        'dropout': 0.0,
        'aggregation_type': 'mean'
    }

    model = make_model(**model_params)
    # load_pre_training
    # model_state_dict = model.state_dict()
    # pretrained_name = '../pretrained_weights.pt'
    # pretrained_state_dict = torch.load(pretrained_name)
    # for name, param in pretrained_state_dict.items():
    #     if 'generator' in name:
    #         continue
    #     if isinstance(param, torch.nn.Parameter):
    #         param = param.data
    #     model_state_dict[name].copy_(param)
    model_dict = torch.load(
            '../MATmodel.pt', map_location=device)
    model.load_state_dict(model_dict['state_dict'])

    model = DataParallel(model, device_ids=[0, 1]).to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.5)

    param_dict = {
        'train_epoch': 100,
        'model': model,
        'optim': optimizer,
        'lr_scheduler': lr_scheduler,
        'loss_fn': loss_fn,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'device': device,
    }

    early_stop_cout = 0
    pre_val_loss = 100
    min_loss = 100
    for epoch in range(1, 100):
        model_path = '../MATmodel.pt'
        val_loss = train(
            param_dict=param_dict, epoch=epoch, model_path=model_path)
        if val_loss < min_loss:
            early_stop_cout = 0
            min_loss = val_loss
        else:
            early_stop_cout += 1
        if early_stop_cout >= 15:
            break
