import os
import time
import pickle
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
# import seaborn as sns
# from scipy import interpolate
from datetime import datetime
import pandas as pd
# from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
# from common import *
# from net import CRNN
# from traditional_methods.net import CRNN
import utils
import dataloader
# from attmodels import make_model
from vitmodels import ViT, RelationNetwork, weights_init, lstm_encoder
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from torch.utils.data.sampler import RandomSampler

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings

warnings.filterwarnings('ignore')
FROM_SCRATCH = True
n_cyc = 30
in_stride = 3
fea_num = 100

v_low = 3.36
v_upp = 3.60
q_low = 610
q_upp = 1190
rul_factor = 3000
cap_factor = 1190
i_low = -2199
i_upp = 5498
pkl_dir = './our_data/'
pkl_list = os.listdir(pkl_dir)
# pkl_list = sorted(pkl_list, key=lambda x: int(x.split('-')[0]) * 10 + int(x[-5]))
'''below: useful'''
# train_name = []
# for name in pkl_list:
#     train_name.append(name[:-4])
# all_series = dict()
# print('----init_train----')
# for name in train_name:
#     # print(name)
#     # tmp_fea, tmp_lbl = dataloader.get_xy(name, n_cyc, in_stride, fea_num, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor)
#     tmp_fea = dataloader.get_xyv2(name, n_cyc, i_low, i_upp, v_low, v_upp, q_low, q_upp, rul_factor,
#                                          cap_factor, pkl_dir)
#     all_series.update({name: {'fea': tmp_fea}})
'''end for useful'''
# import pdb;pdb.set_trace()

new_valid = ['4-3', '5-7', '3-3', '2-3', '9-3', '10-5', '3-2', '3-7']
new_train = ['9-1', '2-2', '4-7', '9-7', '1-8', '4-6', '2-7', '8-4', '7-2', '10-3', '2-4', '7-4', '3-4',
             '5-4', '8-7', '7-7', '4-4', '1-3', '7-1', '5-2', '6-4', '9-8', '9-5', '6-3', '10-8', '1-6', '3-5',
             '2-6', '3-8', '3-6', '4-8', '7-8', '5-1', '2-8', '8-2', '1-5', '7-3', '10-2', '5-5', '9-2', '5-6', '1-7',
             '8-3', '4-1', '4-2', '1-4', '6-5', ]
new_test = ['9-6', '4-5', '1-2', '10-7', '1-1', '6-1', '6-6', '9-4', '10-4', '8-5', '5-3', '10-6',
            '2-5', '6-2', '3-1', '8-8', '8-1', '8-6', '7-6', '6-8', '7-5', '10-1']

# new_valid = ['4-3']
# new_train = ['9-1', '2-2', '4-7']
# new_test = ['9-6']

stride = 10
train_fea, train_ruls, train_batteryids = [], [], []
# series_lens = [80 + 5 * i for i in range(1)]
series_lens = [100]#[80 + 5 * i for i in range(85)]
retrieval_set = {}
battery_id = 0
for name in new_train + new_valid:
    # fea, lbl = dataloader.get_xyv2(name, series_lens, i_low, i_upp, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor, pkl_dir, raw_features=False)
    # tmp_fea, tmp_lbl = all_loader[name]['fea'], all_loader[name]['lbl']
    fea, lbl = dataloader.get_xy_from_start(name, series_lens, i_low, i_upp, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor, pkl_dir, raw_features=False)
    train_fea.append(fea)#[::stride])
    train_ruls.append(lbl)#[::stride])
    train_batteryids += [battery_id for i in range(len(fea))]#[::stride]))]
    retrieval_set[battery_id] = dataloader.get_xyv2(name, series_lens, i_low, i_upp, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor, pkl_dir, raw_features=True)
    battery_id += 1
    # import pdb;pdb.set_trace()

train_fea = np.vstack(train_fea)
train_ruls = np.hstack(train_ruls)
# import pdb;pdb.set_trace()
train_batteryids = np.array(train_batteryids)
train_lbl = np.vstack((train_ruls, train_batteryids)).T

valid_fea, valid_rul, valid_batteryids = [], [], []
valid_battery_id = 0
for name in new_test:
    fea, lbl = dataloader.get_xy_from_start(name, series_lens, i_low, i_upp, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor,
                                   pkl_dir, raw_features=False)
    valid_fea.append(fea)#[::stride])
    valid_rul.append(lbl)#[::stride])
    valid_batteryids += [valid_battery_id for i in range(len(fea))]#[::stride]))]
    valid_battery_id += 1
valid_fea = np.vstack(valid_fea)
valid_rul = np.hstack(valid_rul)#.squeeze()
valid_batteryids = np.array(valid_batteryids)
valid_lbl = np.vstack((valid_rul, valid_batteryids)).T

print(train_fea.shape, train_lbl.shape, valid_fea.shape, valid_lbl.shape)

# seed_torch(0)
batch_size = 1
valida_batch_size = 1
# train_fea_ = train_fea[:].copy()
# train_lbl_ = train_lbl[:].copy()
#
# train_fea_ = train_fea_.transpose(0, 3, 2, 1)
# valid_fea_ = valid_fea.transpose(0, 3, 2, 1)
'''do not forget to add data augmentation!'''
trainset = TensorDataset(torch.Tensor(train_fea), torch.Tensor(train_lbl))
validset = TensorDataset(torch.Tensor(valid_fea), torch.Tensor(valid_lbl))
# trainset = dataloader.Seriesset(torch.Tensor(train_fea), torch.Tensor(train_lbl), torch.Tensor(train_batteryids))
# validset = dataloader.Seriesset(valid_fea, valid_lbl, valid_batteryids)
# import pdb;pdb.set_trace()
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(validset, batch_size=valida_batch_size)
# import pdb;pdb.set_trace()
# train_loader_t = DataLoader(trainset, batch_size=batch_size, shuffle=False)

if FROM_SCRATCH:
    '''
    lamda (float): The weight of RUL loss
    alpha (List: [float]): The weights of Capacity loss
    '''
    lamda = 1e-2
    alpha = torch.Tensor([0.1] * 10)

    '''
    model training 
    '''
    tic = time.time()
    # seed_torch(0)
    device = 'cuda'
    # model = CRNN(100, 4, 64, 64)
    # encoder, relationmodel = make_model(h=4,N=1,d_in=14,d_model=64,d_ff=64,d_embedding=32,dropout=0.5)
    encoded_feature_dim = 128
    # encoder = ViT(
    #     dim=128, #512,  # transformer input dimension
    #     in_dim=19,
    #     num_classes=encoded_feature_dim,
    #     # channels=3,
    #     depth=3,
    #     heads=2,
    #     mlp_dim=512, #1024,
    #     dropout=0.1,
    #     emb_dropout=0.1
    # )
    # weights_init(encoder)
    encoder = lstm_encoder(indim=19, hiddendim=128, fcdim=128, outdim=128, n_layers=1, dropout=0.1)
    relationmodel = RelationNetwork(input_size=2*encoded_feature_dim, hidden_size=512)
    encoder = encoder.to(device)
    relationmodel = relationmodel.to(device)

    num_epochs = 10000

    model_load = False
    retrieval_set_size = 32
    trainer = utils.Trainer(lr=3e-4, n_epochs=num_epochs, device=device, patience=1200,
                      lamda=lamda, alpha=alpha, model_name='./model/wx_inner/wx_inner_pretrain', retreival_set=retrieval_set, retrieval_batch_size=retrieval_set_size)
    model, train_loss, valid_loss, total_loss = trainer.train(train_loader, valid_loader, relationmodel=relationmodel, encoder=encoder, load_model=model_load, reference_set_size=retrieval_set_size, retrieval_batch_size=1)

    print(time.time() - tic)

else:  # model has been pre-trained on common dataset
    lamda = 0.0
    train_weight9 = [0., 0.1, 0., 0., 0.1, 0., 0., 0., 0., ]
    valid_weight9 = [0. if (i != 0) else 0.1 for i in train_weight9]
    train_alpha = torch.Tensor(train_weight9 + [0.])
    valid_alpha = torch.Tensor(valid_weight9 + [0.])
    device = 'cuda'

    pretrain_model_path = './model/wx_inner/wx_inner_pretrain_end.pt'
    finetune_model_path = './model/wx_inner/wx_inner_finetune'

    device = 'cuda'
    trainer = Trainer(lr=8e-4, n_epochs=None, device=device, patience=1200,
                      lamda=lamda, alpha=None, model_name=None)

    res_dict = {}

    for name in new_test[:]:

        stride = 1
        test_fea, test_lbl = [], []
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

            model = CRNN(100, 4, 64, 64)
            model = model.to(device)
            # model.load_state_dict(torch.load(pretrain_model_path))

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

            num_epochs = 120
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
        save_obj(res_dict, './result/res_dict')
