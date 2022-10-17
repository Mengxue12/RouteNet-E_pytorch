import os
import sys
sys.path.insert(1, '../../')
sys.path.insert(1, '../')

import psutil
from read_dataset import GenDataset
from model_torch import GNN_Model
import configparser
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

EPOCHS = 35
epoch_number = 0
best_vloss = .01
best_MAPE = 5
len_train = 20_000
len_val = 10_000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 'cpu') # 
params = configparser.ConfigParser()
params._interpolation = configparser.ExtendedInterpolation()
params.read('config.ini')
LR = float(params['HYPERPARAMETERS']['learning_rate'])


def transformation(x, y):
    traffic_mean = 660.5723876953125
    traffic_std = 420.22003173828125
    packets_mean = 0.6605737209320068
    packets_std = 0.42021000385284424
    capacity_mean = 25442.669921875
    capacity_std = 16217.9072265625

    x["traffic"] = (x["traffic"] - traffic_mean) / traffic_std

    x["packets"] = (x["packets"] - packets_mean) / packets_std

    x["capacity"] = (x["capacity"] - capacity_mean) / capacity_std

    return x, torch.log(y)

def denorm_MAPE(y_true, y_pred):
    denorm_y_true = torch.exp(y_true)
    denorm_y_pred = torch.exp(y_pred)
    return torch.abs((denorm_y_pred - denorm_y_true) / denorm_y_true) * 100

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

model = GNN_Model(params)
model.to(device)
print(model)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=LR)

def train_one_epoch(epoch_index, tb_writer):
    print("begin training...=====================")
    # Iterate in batches over the training dataset.
    running_loss_1k = 0.
    running_MAPE_1k = 0.
    running_loss = 0.
    running_MAPE = 0.
    last_loss = 0.
    last_MAPE = 0.
    avg_loss = 0.
    avg_MAPE = 0.

    ds_train = GenDataset(
    '../../../data/traffic_models/autocorrelated/train', 
    label='delay', 
    shuffle=True, 
    sample=len_train,
    transform=lambda x, y: transformation(x, y))

    loader_train = torch.utils.data.DataLoader(ds_train)
    #for i, (x, y) in enumerate(ds_train):       # target [len(y)]
    for i, (x, y) in enumerate(loader_train):    # target [batch_size, len(y)]
        y = y.to(device)
        optimizer.zero_grad()     # Clear gradients.
        out = model(x)
        loss = criterion(out, y)  # Compute the loss.
        loss.backward()           # Derive gradients.
        optimizer.step()          # Update parameters based on gradients.    
        # Gather data and report
        val_denorm_MAPE = torch.mean(denorm_MAPE(y,out)).detach()
        running_loss += loss.item()
        running_MAPE += val_denorm_MAPE
        running_loss_1k += loss.item()
        running_MAPE_1k += val_denorm_MAPE
        avg_loss = running_loss / (i + 1)
        avg_MAPE = running_MAPE / (i + 1)
        # print("loss",loss,"loss.item",loss.item(),"MAPE",val_denorm_MAPE)
        print('train epoch {} batch {} loss: {:.8} MAPE {:.8}'.format(epoch_index+1, i + 1, loss.item(),val_denorm_MAPE),'cpu mem', round(psutil.virtual_memory().used/1024**2), 'MB cuda mem', torch.cuda.memory_allocated(device),end='\r')
        if i % 1000 == 999:
            print("train batch {} avg_vloss: {} avg_vMAPE {}".format(i + 1, avg_loss, avg_MAPE))
            last_loss = running_loss_1k / 1000 # loss per batch
            last_MAPE = running_MAPE_1k / 1000
            print('train batch {} last 1k loss: {} MAPE {}'.format(i + 1, last_loss, last_MAPE))
            tb_x = epoch_index * len_train + i + 1  # epoch=tb_x/len_train
            tb_writer.add_scalar('Training LOSS', last_loss, tb_x)
            tb_writer.add_scalar('Training MAPE', last_MAPE, tb_x)
            running_loss_1k = 0.
            running_MAPE_1k = 0.
    
    return last_loss, last_MAPE

def val_one_epoch(epoch_index,tb_writer):
    print("begin validating...=====================")
    running_vloss_1k = 0.0
    running_vMAPE_1k = 0.
    running_vloss = 0.0
    running_vMAPE = 0.
    last_vloss = 0.
    last_vMAPE = 0.
    avg_vloss = .0
    avg_vMAPE = 0.

    ds_test = GenDataset(
    '../../../data/traffic_models/autocorrelated/test', 
    label='delay', 
    shuffle=True, 
    sample=len_val,
    transform=lambda x, y: transformation(x, y))
    
    loader_test = torch.utils.data.DataLoader(ds_test)
    # for i, vdata in enumerate(ds_test):
    for i, vdata in enumerate(loader_test):
        x, y = vdata
        y = y.to(device)
        voutputs = model(x)
        vloss = criterion(voutputs, y)
        val_denorm_MAPE = torch.mean(denorm_MAPE(y,voutputs)).detach()
        running_vloss += vloss.item()
        running_vMAPE += val_denorm_MAPE
        running_vloss_1k += vloss.item()
        running_vMAPE_1k += val_denorm_MAPE
        avg_vloss = running_vloss / (i + 1)
        avg_vMAPE = running_vMAPE / (i + 1)
        
        print('val epoch {} batch {} vloss: {:.8} MAPE {:.8}'.format(epoch_index+1, i + 1, vloss, val_denorm_MAPE),' mem used '+ str(round(psutil.virtual_memory().used/1024**2)) +' MIB')
        
        if i % 1000 == 999:
            print('val batch {} avg_vloss: {} avg_vMAPE {}'.format(i + 1, avg_vloss,avg_vMAPE))
            last_vloss = running_vloss_1k / 1000 # loss per batch
            last_vMAPE = running_vMAPE_1k / 1000
            print('val batch {} last 1k avg_vloss: {} avg_vMAPE {}'.format(i + 1, last_vloss, last_vMAPE))
            tb_x = epoch_index * len_val + i + 1  # epoch=tb_x/len_train
            tb_writer.add_scalar('Val LOSS', last_vloss, tb_x)
            tb_writer.add_scalar('Val MAPE', last_vMAPE, tb_x)
            running_vloss_1k = 0.
            running_vMAPE_1k = 0.
    return avg_vloss, avg_vMAPE


timestamp = (datetime.datetime.now()+datetime.timedelta(hours=2)).strftime('%Y%m%d_%H%M')
suffix = '_train_'+human_format(len_train)+'_val_'+human_format(len_val)+'_EPOCHS_'+str(EPOCHS)+'_lr_'+str(LR)

# tensorboard
writer = torch.utils.tensorboard.SummaryWriter('runs/{}{}'.format(timestamp,suffix))

# checkpoint TODO if found best model, load
ckpt_folderpath = os.path.join('checkpoint',str(timestamp) + suffix)
if not os.path.exists(ckpt_folderpath):
    print("create checkpoint to save model...")
    os.makedirs(ckpt_folderpath)
else:
    print("checkpoint folder exists")


for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))
    # begin training
    model.train(True)
    avg_loss, avg_MAPE = train_one_epoch(epoch_number, writer)
    model.train(False) # eval
    avg_vloss, avg_vMAPE = val_one_epoch(epoch_number, writer)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    print('MAPE train {} valid {}'.format(avg_MAPE, avg_vMAPE))
    
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.add_scalars('Training vs. Validation MAPE',
                    { 'Training' : avg_MAPE, 'Validation' : avg_vMAPE },
                    epoch_number + 1)
    writer.flush()
    print('current time {}'.format((datetime.datetime.now()+datetime.timedelta(hours=2)).strftime('%Y%m%d_%H:%M:%S')))
    
    model_path = '{}/epoch_{}_vloss_{:.8}_vMAPE_{:.8}'.format(ckpt_folderpath, epoch_number+1, avg_vloss,avg_vMAPE)
    torch.save({
            'epoch': epoch_number+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            'vloss': avg_vloss}, model_path)
    if avg_vMAPE < best_MAPE:
        # best_vloss = avg_vloss
        best_MAPE = avg_vMAPE
        model_path = 'checkpoint/best_models/{}_epoch_{}_vloss_{:.8}_vMAPE_{:.8}{}'.format(timestamp, epoch_number+1, avg_vloss, avg_vMAPE, suffix)
        torch.save({
            'epoch': epoch_number+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            'vloss': avg_vloss}, model_path)
    
    epoch_number += 1

