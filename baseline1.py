import os
import time
import pickle
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
from scipy import interpolate
from datetime import datetime
import pandas as pd
from baseline.EESbaseline.tool import EarlyStopping
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from baseline.EESbaseline.common import *
from baseline.EESbaseline.net import CRNN

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from torch.utils.data.sampler import RandomSampler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings

warnings.filterwarnings('ignore')

n_cyc = 100
in_stride = 10
fea_num = 100

v_low = 3.36
v_upp = 3.60
q_low = 610
q_upp = 1190
rul_factor = 3000
cap_factor = 1190

ign_bat = ['b25','b33','a0','a1','a2','a3','a4','b22','b24','b26','b38','b44','c25','c43','c44']

train_fea = load_obj('./data/ne_data/fea_train')
train_lbl = load_obj('./data/ne_data/label_train')
ne_train_name = list(train_fea.keys())

valid_fea = load_obj('./data/ne_data/fea_test')
valid_lbl = load_obj('./data/ne_data/label_test')
ne_valid_name = list(valid_fea.keys())

test_fea = load_obj('./data/ne_data/fea_sec')
test_lbl = load_obj('./data/ne_data/label_sec')
ne_test_name = list(test_fea.keys())

all_loader = dict()
for name in ne_train_name:
    tmp_fea, tmp_lbl = train_fea.get(name), train_lbl.get(name)
    all_loader.update({name: {'fea': tmp_fea, 'lbl': tmp_lbl}})

for name in ne_valid_name:
    tmp_fea, tmp_lbl = valid_fea.get(name), valid_lbl.get(name)
    all_loader.update({name: {'fea': tmp_fea, 'lbl': tmp_lbl}})

for name in ne_test_name:
    tmp_fea, tmp_lbl = test_fea.get(name), test_lbl.get(name)
    all_loader.update({name: {'fea': tmp_fea, 'lbl': tmp_lbl}})

for nm in list(all_loader.keys()):
    del_rows = []
    tmp_fea = all_loader[nm]['fea']
    tmp_lbl = all_loader[nm]['lbl']
    for i in range(1, 11):
        del_rows += list(np.where(np.abs(np.diff(tmp_lbl[:, i])) > 0.05)[0])
    tmp_fea = np.delete(tmp_fea, del_rows, axis=0)
    tmp_lbl = np.delete(tmp_lbl, del_rows, axis=0)
    all_loader.update({nm: {'fea': tmp_fea, 'lbl': tmp_lbl}})

pkl_list = os.listdir('./data/our_data/')
pkl_list = sorted(pkl_list, key=lambda x: int(x.split('-')[0]) * 10 + int(x[-5]))

train_name = []
for name in pkl_list:
    train_name.append(name[:-4])

print('----init_train----')
for name in train_name:
    tmp_fea, tmp_lbl = get_xy(name, n_cyc, in_stride, fea_num, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor)
    all_loader.update({name: {'fea': tmp_fea, 'lbl': tmp_lbl}})


new_valid = ['4-3', '5-7', '3-3', '2-3', '9-3', '10-5', '3-2', '3-7']
new_train = ['9-1', '2-2', '4-7','9-7', '1-8','4-6','2-7','8-4', '7-2','10-3', '2-4', '7-4', '3-4',
            '5-4', '8-7','7-7', '4-4','1-3', '7-1','5-2', '6-4', '9-8','9-5','6-3','10-8','1-6','3-5',
             '2-6', '3-8', '3-6', '4-8', '7-8','5-1', '2-8', '8-2','1-5','7-3', '10-2','5-5', '9-2','5-6', '1-7',
             '8-3', '4-1','4-2','1-4','6-5', ]
new_test  = ['9-6','4-5','1-2', '10-7','1-1', '6-1','6-6', '9-4','10-4','8-5', '5-3','10-6',
            '2-5','6-2','3-1','8-8', '8-1','8-6','7-6','6-8','7-5','10-1']

stride = 10
train_fea, train_lbl = [], []
for name in ne_train_name + ne_test_name:
    if name in ign_bat:continue
    tmp_fea, tmp_lbl = all_loader[name]['fea'],all_loader[name]['lbl']
    train_fea.append(tmp_fea[::stride])
    train_lbl.append(tmp_lbl[::stride])
train_fea = np.vstack(train_fea)
train_lbl = np.vstack(train_lbl).squeeze()

stride = 10
valid_fea, valid_lbl = [], []
for name in ne_valid_name:
    if name in ign_bat:continue
    tmp_fea, tmp_lbl = all_loader[name]['fea'],all_loader[name]['lbl']
    valid_fea.append(tmp_fea[::stride])
    valid_lbl.append(tmp_lbl[::stride])
valid_fea = np.vstack(valid_fea)
valid_lbl = np.vstack(valid_lbl).squeeze()

print(train_fea.shape, train_lbl.shape, valid_fea.shape, valid_lbl.shape)

seed_torch(0)
batch_size = 256

train_fea_ = train_fea[:].copy()
train_lbl_ = train_lbl[:].copy()

train_fea_ = train_fea_.transpose(0,3,2,1)
valid_fea_ = valid_fea.transpose(0,3,2,1)

trainset = TensorDataset(torch.Tensor(train_fea_), torch.Tensor(train_lbl_))
validset = TensorDataset(torch.Tensor(valid_fea_), torch.Tensor(valid_lbl))

train_loader = DataLoader(trainset, batch_size=batch_size,shuffle=True)
valid_loader = DataLoader(validset, batch_size=batch_size,)

train_loader_t = DataLoader(trainset, batch_size=batch_size,shuffle=False)


'''
lamda (float): The weight of RUL loss
alpha (List: [float]): The weights of Capacity loss
'''
lamda = 1e-2
alpha = torch.Tensor([0.1] * 10 )

tic = time.time()
seed_torch(0)
device = 'cuda'
model = CRNN(100,4,64,64,sigmoid=True)
model = model.to(device)

num_epochs = 12000


model_load = False
trainer = Trainer(lr = 8e-4, n_epochs = num_epochs,device = device, patience = 1600,
                  lamda = lamda, alpha = alpha, model_name='./model/ne2wx/ne_pretrain')
model ,train_loss, valid_loss, total_loss = trainer.train(train_loader, valid_loader, model, model_load)

print(time.time()-tic)


lamda = 0.0
train_weight9 = [0.1] * 9
valid_weight9 = [0. if (i!=0) else 0.1 for i in train_weight9]
train_alpha = torch.Tensor(train_weight9 + [0.] )
valid_alpha = torch.Tensor(valid_weight9 + [0.])

pretrain_model_path = './model/ne2wx/ne_pretrain_best.pt'
finetune_model_path = './model/ne2wx/ne_finetune'

# online transfer

lamda = 0.0
train_weight9 = [0.1] * 9
valid_weight9 = [0. if (i!=0) else 0.1 for i in train_weight9]
train_alpha = torch.Tensor(train_weight9 + [0.] )
valid_alpha = torch.Tensor(valid_weight9 + [0.])

pretrain_model_path = './model/ne2wx/ne_pretrain_best.pt'
finetune_model_path = './model/ne2wx/ne_finetune'

res_dict = {}

for name in new_test[:]:

    stride = 1
    test_fea, test_lbl = [], []
    if name in ign_bat: print('wrong bat')
    tmp_fea, tmp_lbl = all_loader[name]['fea'], all_loader[name]['lbl']
    test_fea.append(tmp_fea[::stride])
    test_lbl.append(tmp_lbl[::stride])
    test_fea = np.vstack(test_fea)
    test_lbl = np.vstack(test_lbl).squeeze()

    batch_size = 20 if len(test_fea) % 20 != 1 else 21
    rul_true, rul_pred, rul_base, SOH_TRUE, SOH_PRED, SOH_BASE = [], [], [], [], [], []
    for i in range(test_fea.shape[0] // batch_size + 1):

        test_fea_ = test_fea[i * batch_size: i * batch_size + batch_size].transpose(0, 3, 2, 1)
        test_lbl_ = test_lbl[i * batch_size: i * batch_size + batch_size]
        testset = TensorDataset(torch.Tensor(test_fea_), torch.Tensor(test_lbl_))
        test_loader = DataLoader(testset, batch_size=batch_size, )
        if test_fea_.shape[0] == 0: continue

        model = CRNN(100, 4, 64, 64, sigmoid=True)
        model = model.to(device)
        model.load_state_dict(torch.load(pretrain_model_path))

        _, y_pred, _, _, soh_pred = trainer.test(test_loader, model)
        rul_base.append(y_pred.cpu().detach().numpy())
        SOH_BASE.append(soh_pred.cpu().detach().numpy())

        for p in model.soh.parameters():
            p.requires_grad = False
        for p in model.rul.parameters():
            p.requires_grad = False
        for p in model.cnn.parameters():
            p.requires_grad = False

        tic = time.time()
        seed_torch(2021)

        num_epochs = 200
        model_load = False
        trainer = FineTrainer(lr=1e-4, n_epochs=num_epochs, device=device, patience=1000,
                              lamda=lamda, train_alpha=train_alpha, valid_alpha=valid_alpha,
                              model_name=finetune_model_path)
        model, train_loss, valid_loss, total_loss, added_loss = trainer.train(test_loader, test_loader, model,
                                                                              model_load)

        y_true, y_pred, mse_loss, soh_true, soh_pred = trainer.test(test_loader, model)
        rul_true.append(y_true.cpu().detach().numpy().reshape(-1, 1))
        rul_pred.append(y_pred.cpu().detach().numpy())
        SOH_TRUE.append(soh_true.cpu().detach().numpy())
        SOH_PRED.append(soh_pred.cpu().detach().numpy())

    rul_true = np.vstack(rul_true).squeeze()
    rul_pred = np.vstack(rul_pred).squeeze()
    rul_base = np.vstack(rul_base).squeeze()
    SOH_TRUE = np.vstack(SOH_TRUE)
    SOH_PRED = np.vstack(SOH_PRED)
    SOH_BASE = np.vstack(SOH_BASE)

    fig = plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.plot(rul_true[:] * rul_factor, '.', label='true')
    plt.plot(rul_pred[:] * rul_factor, '.', label='transfer')
    plt.legend(fontsize=20)
    plt.title(f'{name} cycle life ({len(test_fea)}): RUL', fontsize=20)
    plt.xlabel('Cycle', fontsize=20)
    plt.ylabel('RUL', fontsize=20)
    plt.subplot(122)
    for seq_num in range(9, 10):
        plt.plot(SOH_TRUE[:, seq_num] * cap_factor, '.', label='true')
        plt.plot(SOH_PRED[:, seq_num] * cap_factor, '.', label='transfer')
    plt.legend(fontsize=20)
    plt.title(f'{name}: Capacity', fontsize=20)
    plt.xlabel('Cycle', fontsize=20)
    plt.ylabel('Capacity', fontsize=20)
    plt.show()

    res_dict.update({name: {
        'rul': {
            'true': rul_true[:] * rul_factor,
            'base': rul_base[:] * rul_factor,
            'transfer': rul_pred[:] * rul_factor,
        },
        'soh': {
            'true': SOH_TRUE[:, 9] * cap_factor,
            'base': SOH_BASE[:, 9] * cap_factor,
            'transfer': SOH_PRED[:, 9] * cap_factor,
        },
    }
    })
    save_obj(res_dict, './result/ne_dict')