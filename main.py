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
from vitmodels import ViT, RelationNetwork, weights_init, lstm_encoder, FFNEncoder
import load_ne
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from torch.utils.data.sampler import RandomSampler

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
'''below is the order when we do not want to upload training log to cloud'''
os.environ['WANDB_MODE'] = 'offline'
import wandb

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings
import argparse
import yaml

# from load_ne import get_train_test_val
# get_train_test_val()

# batch_size = 32
# valida_batch_size = 1
# seriesnum = 1000
# scale_ratios = [1, 2]  # , 3, 4]  # [1, 2, 3]  # must in ascent order, e.g. [1, 2, 3]
# except_ratios = [[2, 2]]
# # except_ratios = [[1, 2], [2, 1],
# #                  [2, 2],
# #                  [1, 3], [3, 1], [3, 3],
# #                  [1, 4], [2, 4], [4, 1], [4, 2], [4, 4]]
# parts_num_per_ratio = 240
# valid_max_len = 10

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--valid-batch-size', type=int, default=1)
parser.add_argument('--train-retrieval-num',
                    help='source domain sequence number when training',
                    type=int,
                    default=20)  # 0.1
parser.add_argument('--seriesnum',
                    help='series number cut off from training sequence',
                    type=int,
                    default=3000)  # 3000
parser.add_argument(
    '--train_fragments_num',
    help='number of retrieval fragments cut off from each'
    'retrieval sequence for comparing when training, since we use the same set for retrieval '
    'and training, we better keep this and --seriesnum the same',
    type=int,
    default=3000)
parser.add_argument('--scale-ratios', type=list,
                    default=[1, 2, 3, 4, 5])  # [1, 2, 3, 4, 5])  # [1, 2]
parser.add_argument('--except-ratios',
                    type=list,
                    default=[[1, 2], [2, 1], [2, 2], [1, 3], [3, 1], [3, 3],
                             [1, 4], [2, 4], [4, 1], [4, 2], [4, 4], [5, 1],
                             [5, 2], [5, 5], [1, 5], [2, 5]])
parser.add_argument('--data-aug-ratios', type=list, default=[1.2, 1.4, 1.6])
parser.add_argument('--parts-num-per-ratio',
                    help='sequence number from each scaling ratio',
                    default=1000)  # 500 240
parser.add_argument('--valid-max-len',
                    type=int,
                    help='sequence number for testing',
                    default=10)
parser.add_argument('--lstm-hidden',
                    type=int,
                    help='lstm hidden layer number',
                    default=128)  # 128
parser.add_argument('--fc-hidden',
                    type=int,
                    help='fully connect layer hidden dimension',
                    default=98)  # 128
parser.add_argument('--fc-out',
                    type=int,
                    help='embedded sequence dimmension',
                    default=64)  # 128
parser.add_argument('--dropout', type=float, default=0.3)  # 0.1
parser.add_argument('--lstm-layer', type=int, default=1)  # 0.1

parser.add_argument('--lr',
                    help='initial learning rate',
                    type=float,
                    default=1e-3)  # 0.1
parser.add_argument('--gama',
                    help='learning rate decay rate',
                    type=float,
                    default=0.9)  # 0.1
args = parser.parse_args()
if __name__ == '__main__':
    FROM_SCRATCH = True
    n_cyc = 30
    in_stride = 3
    fea_num = 100

    v_low = 3.36
    v_upp = 3.60
    q_low = 610
    q_upp = 1190
    rul_factor = 3000.
    cap_factor = 1190
    i_low = -2199
    i_upp = 5498
    pkl_dir = './our_data/'
    # pkl_list = os.listdir(pkl_dir)
    # pkl_list = sorted(pkl_list, key=lambda x: int(x.split('-')[0]) * 10 + int(x[-5]))
    seq_len = 100
    series_lens = [100]

    OURDATA = False
    # print(type(args.scale_ratios), args.scale_ratios)
    # exit(0)
    wandb.init(
        project='ne_noscale',
        config={
            'batch_size': 32,
            'valida_batch_size': 1,
            'seriesnum': 1000,
            'scale_ratios':
            '[1, 2, 3]',  # [1, 2, 3]  # must in ascent order, e.g. [1, 2, 3]
            'except_ratios':
            '[[1, 2], [2, 1], [2, 2], [1, 3], [3, 1], [3, 3], [1, 4], [2, 4], [4, 1], [4, 2], [4, 4]]',
            'parts_num_per_ratio': 240,
            'valid_max_len': 10
        })
    data_aug_scale_ratios = [1.]
    for scale_ratio in args.data_aug_ratios:
        data_aug_scale_ratios += [1 / scale_ratio, scale_ratio]

    if OURDATA:
        new_valid = ['4-3', '5-7', '3-3', '2-3', '9-3', '10-5', '3-2', '3-7']
        new_train = [
            '9-1',
            '2-2',
            '4-7',
            '9-7',
            '1-8',
            '4-6',
            '2-7',
            '8-4',
            '7-2',
            '10-3',
            '2-4',
            '7-4',
            '3-4',
            '5-4',
            '8-7',
            '7-7',
            '4-4',
            '1-3',
            '7-1',
            '5-2',
            '6-4',
            '9-8',
            '9-5',
            '6-3',
            '10-8',
            '1-6',
            '3-5',
            '2-6',
            '3-8',
            '3-6',
            '4-8',
            '7-8',
            '5-1',
            '2-8',
            '8-2',
            '1-5',
            '7-3',
            '10-2',
            '5-5',
            '9-2',
            '5-6',
            '1-7',
            '8-3',
            '4-1',
            '4-2',
            '1-4',
            '6-5',
        ]
        new_test = [
            '9-6', '4-5', '1-2', '10-7', '1-1', '6-1', '6-6', '9-4', '10-4',
            '8-5', '5-3', '10-6', '2-5', '6-2', '3-1', '8-8', '8-1', '8-6',
            '7-6', '6-8', '7-5', '10-1'
        ]

        train_fea, train_ruls, train_batteryids = [], [], []

        batteryid = 0
        for name in new_train + new_valid:
            # tmp_fea, tmp_lbl = dataloader.get_xy(name, n_cyc, in_stride, fea_num, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor)
            tmp_fea, tmp_lbl = dataloader.get_xyv2(name,
                                                   series_lens,
                                                   i_low,
                                                   i_upp,
                                                   v_low,
                                                   v_upp,
                                                   q_low,
                                                   q_upp,
                                                   rul_factor,
                                                   cap_factor,
                                                   pkl_dir,
                                                   raw_features=False,
                                                   seriesnum=args.seriesnum)
            train_fea.append(tmp_fea)
            train_ruls.append(tmp_lbl)
            train_batteryids += [batteryid for _ in range(tmp_fea.shape[0])]
            batteryid += 1

        retrieval_set = {}
        batteryid = 0
        for name in new_train + new_valid:
            retrieval_set[batteryid] = dataloader.get_retrieval_seq(
                name, pkl_dir, rul_factor, seriesnum=5000)
            batteryid += 1

        train_fea = np.vstack(train_fea)
        train_ruls = np.vstack(train_ruls)
        # import pdb;pdb.set_trace()
        train_batteryids = np.array(train_batteryids)
        train_batteryids = train_batteryids.reshape((-1, 1))
        train_lbl = np.hstack((train_ruls, train_batteryids))

        valid_fea, valid_rul, valid_batteryids = [], [], []
        valid_battery_id = 0

        for name in new_test:
            tmp_fea, tmp_lbl = dataloader.get_xyv2(name,
                                                   series_lens,
                                                   i_low,
                                                   i_upp,
                                                   v_low,
                                                   v_upp,
                                                   q_low,
                                                   q_upp,
                                                   rul_factor,
                                                   cap_factor,
                                                   pkl_dir,
                                                   raw_features=False)
            valid_fea.append(tmp_fea[:args.valid_max_len])  # [::stride])
            valid_rul.append(tmp_lbl[:args.valid_max_len])  # [::stride])strid
            valid_batteryids += [
                valid_battery_id for i in range(len(tmp_fea))
            ][:args.valid_max_len]  # [::e]))]
            valid_battery_id += 1
        valid_fea = np.vstack(valid_fea)
        valid_rul = np.vstack(valid_rul)  # .squeeze()
        valid_batteryids = np.array(valid_batteryids)
        valid_batteryids = valid_batteryids.reshape((-1, 1))
        valid_lbl = np.hstack((valid_rul, valid_batteryids))
    else:

        train_fea, train_lbl = load_ne.get_train_test_val(
            series_len=series_lens[0],
            rul_factor=rul_factor,
            dataset_name='trainvalid',
            seqnum=args.seriesnum,
            data_aug_scale_ratios=data_aug_scale_ratios)
        # train_fea = train_fea[:seriesnum]
        # train_lbl = train_lbl[:seriesnum]
        valid_fea, valid_lbl = load_ne.get_train_test_val(
            series_len=series_lens[0],
            rul_factor=rul_factor,
            dataset_name='test',
            seqnum=args.valid_max_len)
        # valid_fea = valid_fea[:valid_max_len]
        # valid_lbl = valid_lbl[:valid_max_len]
        retrieval_set = load_ne.get_retrieval_seq(rul_factor=rul_factor,
                                                  seriesnum=5000)
    print(train_fea.shape, train_lbl.shape, valid_fea.shape, valid_lbl.shape)

    trainset = TensorDataset(torch.Tensor(train_fea), torch.Tensor(train_lbl))
    validset = TensorDataset(torch.Tensor(valid_fea), torch.Tensor(valid_lbl))

    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=True)
    valid_loader = DataLoader(validset, batch_size=args.valid_batch_size)

    directory_based_on_time = str(time.time())

    output_dir = './output/' + directory_based_on_time
    os.mkdir(output_dir)
    with open(output_dir + '/config.yaml', "w", encoding='utf-8') as f:
        yaml.dump(args, f)

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
        device = 'cpu'
        # model = CRNN(100, 4, 64, 64)
        # encoder, relationmodel = make_model(h=4,N=1,d_in=14,d_model=64,d_ff=64,d_embedding=32,dropout=0.5)
        encoded_feature_dim = 128
        # encoder = ViT(
        #     dim=128, #512,  # transformer input dimension
        #     in_dim=13,
        #     num_classes=encoded_feature_dim,
        #     # channels=3,
        #     depth=4,
        #     heads=4,
        #     mlp_dim=512, #1024,
        #     dropout=0.1,
        #     emb_dropout=0.1
        # )
        # weights_init(encoder)
        encoder = lstm_encoder(indim=train_fea.shape[2],
                               hiddendim=args.lstm_hidden,
                               fcdim=args.fc_hidden,
                               outdim=args.fc_out,
                               n_layers=args.lstm_layer,
                               dropout=args.dropout)
        # encoder = FFNEncoder(input_size=9 * 100, hidden_size=args.fc_out)
        # encoder.load_state_dict(torch.load('output/1676706773.6733632/LSTM_relu_b_32_194.pth'))
        relationmodel = RelationNetwork(input_size=2 * args.fc_out,
                                        hidden_size=512)
        # relationmodel = FNNRelationNetwork(input_size=2 * encoded_feature_dim+1, hidden_size=512)
        encoder = encoder.to(device)
        relationmodel = relationmodel.to(device)

        num_epochs = 100

        model_load = False
        retrieval_set_size = 32
        trainer = utils.Trainer(
            lr=args.lr,
            gama=args.gama,
            n_epochs=num_epochs,
            device=device,  #patience=1200,
            # lamda=lamda, alpha=alpha, model_name='./model/wx_inner/wx_inner_pretrain',
            retrieval_set=retrieval_set,
            train_retrieval_size=args.train_retrieval_num,
            retrieval_fragments_num=args.train_fragments_num,
            parts_num_from_each_len=args.parts_num_per_ratio,
            scale_ratios=args.scale_ratios,
            except_ratios=args.except_ratios,
            data_aug_scale_ratios=data_aug_scale_ratios)
        model, train_loss, valid_loss, total_loss = trainer.train(
            train_loader,
            valid_loader,
            relationmodel=relationmodel,
            encoder=encoder,
            wandb=wandb,
            save_path=output_dir)

        print(time.time() - tic)

# else:  # model has been pre-trained on common dataset
#     lamda = 0.0
#     train_weight9 = [0., 0.1, 0., 0., 0.1, 0., 0., 0., 0., ]
#     valid_weight9 = [0. if (i != 0) else 0.1 for i in train_weight9]
#     train_alpha = torch.Tensor(train_weight9 + [0.])
#     valid_alpha = torch.Tensor(valid_weight9 + [0.])
#     device = 'cuda'
#
#     pretrain_model_path = './model/wx_inner/wx_inner_pretrain_end.pt'
#     finetune_model_path = './model/wx_inner/wx_inner_finetune'
#
#     device = 'cuda'
#     trainer = Trainer(lr=8e-4, n_epochs=None, device=device, patience=1200,
#                       lamda=lamda, alpha=None, model_name=None)
#
#     res_dict = {}
#
#     for name in new_test[:]:
#
#         stride = 1
#         test_fea, test_lbl = [], []
#         tmp_fea, tmp_lbl = all_loader[name]['fea'], all_loader[name]['lbl']
#         test_fea.append(tmp_fea[::stride])
#         test_lbl.append(tmp_lbl[::stride])
#         test_fea = np.vstack(test_fea)
#         test_lbl = np.vstack(test_lbl).squeeze()
#
#         batch_size = 20 if len(test_fea) % 20 != 1 else 21
#         rul_true, rul_pred, rul_base, SOH_TRUE, SOH_PRED, SOH_BASE = [], [], [], [], [], []
#
#         for i in range(test_fea.shape[0] // batch_size + 1):
#
#             test_fea_ = test_fea[i * batch_size: i * batch_size + batch_size].transpose(0, 3, 2, 1)
#             test_lbl_ = test_lbl[i * batch_size: i * batch_size + batch_size]
#             testset = TensorDataset(torch.Tensor(test_fea_), torch.Tensor(test_lbl_))
#             test_loader = DataLoader(testset, batch_size=batch_size, )
#
#             if test_fea_.shape[0] == 0: continue
#
#             model = CRNN(100, 4, 64, 64)
#             model = model.to(device)
#             # model.load_state_dict(torch.load(pretrain_model_path))
#
#             _, y_pred, _, _, soh_pred = trainer.test(test_loader, model)
#             rul_base.append(y_pred.cpu().detach().numpy())
#             SOH_BASE.append(soh_pred.cpu().detach().numpy())
#
#             for p in model.soh.parameters():
#                 p.requires_grad = False
#             for p in model.rul.parameters():
#                 p.requires_grad = False
#             for p in model.cnn.parameters():
#                 p.requires_grad = False
#
#             tic = time.time()
#             seed_torch(2021)
#
#             num_epochs = 120
#             model_load = False
#             trainer = FineTrainer(lr=1e-4, n_epochs=num_epochs, device=device, patience=1000,
#                                   lamda=lamda, train_alpha=train_alpha, valid_alpha=valid_alpha,
#                                   model_name=finetune_model_path)
#             model, train_loss, valid_loss, total_loss, added_loss = trainer.train(test_loader, test_loader, model,
#                                                                                   model_load)
#
#             y_true, y_pred, mse_loss, soh_true, soh_pred = trainer.test(test_loader, model)
#             rul_true.append(y_true.cpu().detach().numpy().reshape(-1, 1))
#             rul_pred.append(y_pred.cpu().detach().numpy())
#             SOH_TRUE.append(soh_true.cpu().detach().numpy())
#             SOH_PRED.append(soh_pred.cpu().detach().numpy())
#
#         rul_true = np.vstack(rul_true).squeeze()
#         rul_pred = np.vstack(rul_pred).squeeze()
#         rul_base = np.vstack(rul_base).squeeze()
#         SOH_TRUE = np.vstack(SOH_TRUE)
#         SOH_PRED = np.vstack(SOH_PRED)
#         SOH_BASE = np.vstack(SOH_BASE)
#
#         fig = plt.figure(figsize=(20, 10))
#         plt.subplot(121)
#         plt.plot(rul_true[:] * rul_factor, '.', label='true')
#         plt.plot(rul_pred[:] * rul_factor, '.', label='transfer')
#         plt.legend(fontsize=20)
#         plt.title(f'{name} cycle life ({len(test_fea)}): RUL', fontsize=20)
#         plt.xlabel('Cycle', fontsize=20)
#         plt.ylabel('RUL', fontsize=20)
#         plt.subplot(122)
#         for seq_num in range(9, 10):
#             plt.plot(SOH_TRUE[:, seq_num] * cap_factor, '.', label='true')
#             plt.plot(SOH_PRED[:, seq_num] * cap_factor, '.', label='transfer')
#         plt.legend(fontsize=20)
#         plt.title(f'{name}: Capacity', fontsize=20)
#         plt.xlabel('Cycle', fontsize=20)
#         plt.ylabel('Capacity', fontsize=20)
#         plt.show()
#
#         res_dict.update({name: {
#             'rul': {
#                 'true': rul_true[:] * rul_factor,
#                 'base': rul_base[:] * rul_factor,
#                 'transfer': rul_pred[:] * rul_factor,
#             },
#             'soh': {
#                 'true': SOH_TRUE[:, 9] * cap_factor,
#                 'base': SOH_BASE[:, 9] * cap_factor,
#                 'transfer': SOH_PRED[:, 9] * cap_factor,
#             },
#         }
#         })
#         save_obj(res_dict, './result/res_dict')
