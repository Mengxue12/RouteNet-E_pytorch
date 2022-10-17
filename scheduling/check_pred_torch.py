import os
import sys
sys.path.insert(1, '../../')
sys.path.insert(1, '../')

from read_dataset import GenDataset
from model_torch import GNN_Model

import configparser
import torch
import csv
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

len_test = 80_000
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

def denorm_MAPE_abs(y_true, y_pred):
    denorm_y_true = torch.exp(y_true)
    denorm_y_pred = torch.exp(y_pred)
    return torch.abs((denorm_y_pred - denorm_y_true) / denorm_y_true) * 100

def denorm_MAPE(y_true, y_pred):
    denorm_y_true = torch.exp(y_true)
    denorm_y_pred = torch.exp(y_pred)
    return ((denorm_y_pred - denorm_y_true) / denorm_y_true) * 100

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

model = GNN_Model(params)
model.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=LR)

PATH = input()
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
print('epoch',epoch,'loss',loss,'opti')
model.train(False)


ds_test = GenDataset(
    '../../data/scheduling/test', 
    label='delay', 
    shuffle=False, 
    sample=len_test,
    transform=lambda x, y: transformation(x, y))

loader_test = torch.utils.data.DataLoader(ds_test)


mape  = []
mape_abs = []
pred = []
ydata=[]
loss =[]

for i, vdata in enumerate(loader_test):
    x, y = vdata
    y = y.to(device)
    voutputs = model(x)
    
    vloss = criterion(voutputs, y)
    val_denorm_MAPE = denorm_MAPE(y,voutputs)[0].cpu().detach().numpy()
    val_denorm_MAPE_abs = denorm_MAPE_abs(y,voutputs)[0].cpu().detach().numpy()

    mape.append(val_denorm_MAPE)
    mape_abs.append(val_denorm_MAPE_abs)
    pred.append(voutputs.cpu().detach().numpy()[0])
    ydata.append(y.cpu().detach().numpy()[0])
    loss.append(vloss.item())


with open('pred.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for j in pred:
        writer.writerow(j)

with open('y.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for j in ydata:
        writer.writerow(j)
        
with open('re.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for j in mape:
        writer.writerow(j)
        
with open('are.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for j in mape_abs:
        writer.writerow(j)
    
with open('vloss.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for j in loss:
        writer.writerow(j)
    