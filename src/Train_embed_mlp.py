import argparse
import pickle
import sys
import shutil
import DataLoaderCPI
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from models import Embed_mlp, Baseline, CPInet    # noqa
from tqdm import tqdm
import numpy as np
from Config import DataConfig, ModelConfig, TrainConfig


torch.manual_seed(100)


def train(param_dict, epoch, model_path):
    model = param_dict['model'].train()
    running_loss = []
    pbar = tqdm(total=len(param_dict['train_loader']))

    device = param_dict['device']
    loss_fn = param_dict['loss_fn']
    train_loader = param_dict['train_loader']
    val_loader = param_dict['val_loader']
    optimizer = param_dict['optim']

    scaler = torch.cuda.amp.GradScaler()

    c = 0
    for molecule, molecule_feat, molecule_embed, protein, protein_feat, \
            protein_embed, label in train_loader:
        with torch.cuda.amp.autocast(TrainConfig['MixPrec']):
            molecule_embed, protein_embed, label = \
                molecule_embed.to(device), protein_embed.to(device), \
                label.unsqueeze(1).to(device)
            molecule_feat, protein_feat = \
                molecule_feat.to(device), protein_feat.to(device)
            out = model(
                mol_embed=molecule_embed, pro_embed=protein_embed,
                mol_feat=molecule_feat, pro_feat=protein_feat)
            loss = loss_fn(out, label)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss.append(loss.item())
        pbar.update()
        c += 1
        if c == 100:
            pbar.set_description(
                str(epoch) + '/' + str(param_dict['train_epoch']) + '  ' +
                str(np.mean(running_loss))[:6])
            c = 0

    val_loss = []
    model.eval()

    for molecule, molecule_feat, molecule_embed, \
            protein, protein_feat, protein_embed, label in val_loader:
        with torch.cuda.amp.autocast(TrainConfig['MixPrec']):
            molecule_embed, protein_embed, label = \
                molecule_embed.to(device), protein_embed.to(device), \
                label.unsqueeze(1).to(device)
            molecule_feat, protein_feat = \
                molecule_feat.to(device), protein_feat.to(device)
            out = model(
                mol_embed=molecule_embed, pro_embed=protein_embed,
                mol_feat=molecule_feat, pro_feat=protein_feat)
            loss = loss_fn(out, label)
            val_loss.append(loss.item())
    pbar.set_description(str(epoch) + '/' + str(np.mean(val_loss))[:6])
    lr_scheduler.step()

    model_states = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": running_loss}
    torch.save(model_states, model_path,
               pickle_protocol=pickle.HIGHEST_PROTOCOL,
               _use_new_zipfile_serialization=False)
    pbar.close()

    return np.mean(val_loss)


if __name__ == "__main__":
    print('------------Starting------------' + '\n')

    parser = argparse.ArgumentParser(
        description='Training script for MetaChrom')
    parser.add_argument(
        '--Device', type=str, help='CUDA device for training', default='0')

    args = parser.parse_args()
    DataType = DataConfig['DataType']
    # /mnt/storage_3/blai/CPI_data/dataset/
    train_set = DataLoaderCPI.CPI_data(DataConfig[DataType], partition='train')
    val_set = DataLoaderCPI.CPI_data(DataConfig[DataType], partition='val')

    tarin_loader = DataLoader(
        train_set, batch_size=TrainConfig['BatchSize'],
        num_workers=40, shuffle=True)
    val_loader = DataLoader(
        val_set, batch_size=1,
        num_workers=40, shuffle=False)

    device = torch.device('cuda:' + args.Device)

    if ModelConfig['model'] == 'Embed_mlp':
        model = Embed_mlp.Embed_mlp(ModelConfig['Embed_mlp']).to(device)
    elif ModelConfig['model'] == 'Baseline':
        model = Baseline.Baseline(ModelConfig['Baseline']).to(device)
    elif ModelConfig['model'] == 'CPInet':
        model = CPInet.CPInet(ModelConfig['CPInet']).to(device)
    else:
        sys.exit(-1)

    optimizer = {'SGD': optim.SGD(model.parameters(),
                                  lr=TrainConfig['optim']['lr']),
                 'Adam': optim.Adam(model.parameters(),
                                    lr=TrainConfig['optim']['lr']),
                 'AdamW': optim.AdamW(model.parameters(),
                                      lr=TrainConfig['optim']['lr']),
                 'RMSprop': optim.RMSprop(model.parameters(),
                                          lr=TrainConfig['optim']['lr'])
                 }[TrainConfig['optim']['optimizer']]
    loss_fn = torch.nn.BCEWithLogitsLoss()
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=TrainConfig['optim']['lr_scheduler_step_size'],
        gamma=TrainConfig['optim']['lr_scheduler_gamma'])

    # re_training
    if 'InitModel' in TrainConfig:
        model_dict = torch.load(
            TrainConfig['InitModel'], map_location=device)
        model.load_state_dict(model_dict['state_dict'])
        optimizer.load_state_dict(model_dict['optimizer'])

    param_dict = {
        'train_epoch': TrainConfig['Epoch'],
        'model': model,
        'optim': optimizer,
        'lr_scheduler': lr_scheduler,
        'loss_fn': loss_fn,
        'train_loader': tarin_loader,
        'val_loader': val_loader,
        'device': device,
    }

    print('Number of Training Sequence: ' + str(train_set.__len__()))
    print('Batch Size: ' + str(TrainConfig['BatchSize']))
    print('Learning Rate: ' + str(TrainConfig['optim']['lr']))
    print('Max training Epochs: ' + str(TrainConfig['Epoch']))
    print('Early stopping patience: ' + str(TrainConfig['Patience']))
    print('Saving trained model at: ' + TrainConfig['ModelOut'])

    early_stop_cout = 0
    pre_val_loss = 100
    min_loss = 100
    for epoch in range(1, TrainConfig['Epoch']):
        model_path = TrainConfig['ModelOut']
        val_loss = train(
            param_dict=param_dict, epoch=epoch, model_path=model_path)
        if val_loss < min_loss:
            early_stop_cout = 0
            min_loss = val_loss
            shutil.copy(TrainConfig['ModelOut'],
                        TrainConfig['ModelOut'] + '.best')
        else:
            early_stop_cout += 1
        if early_stop_cout >= TrainConfig['Patience']:
            break
