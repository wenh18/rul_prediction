import os
import time
import pickle
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from datetime import datetime
import pandas as pd
# from tool import EarlyStopping
# from sklearn.metrics import roc_auc_score, mean_squared_error
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from torch.utils.data.sampler import RandomSampler
torch.autograd.set_detect_anomaly(True)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings

warnings.filterwarnings('ignore')

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def compress_seqv2(seq, compress_ratio):
    last_idx = seq.size(0) - 1
    sample_ids = [last_idx]
    while last_idx > 0:
        last_idx -= compress_ratio
        sample_ids.append(last_idx)
    sample_ids.sort()
    if sample_ids[0] < 0:
        sample_ids = sample_ids[1:]
    sampled_seq = seq[sample_ids, :]



def compress_seq(target_seq, original_seq, sample_ratio, device):
    '''
    sample_ratio[0]:target sample ratio, sample_ratio[1]:original sample ratio
    '''
    target_sample_last_ids = target_seq.size(0) - 1
    target_sample_ids = [target_sample_last_ids]
    while target_sample_last_ids > 0:
        target_sample_last_ids -= sample_ratio[0]
        target_sample_ids.append(target_sample_last_ids)
    target_sample_ids.sort()
    if target_sample_ids[0] < 0:
        target_sample_ids = target_sample_ids[1:]

    sampled_target_seq = target_seq[target_sample_ids, :]

    original_sample_ids = [int(sample_ratio[1] * i) for i in range(len(target_sample_ids))]
    sampled_original_seq = original_seq[original_sample_ids, :]
    # import pdb;pdb.set_trace()

    sampled_original_seq_num = original_seq.size(0) // sample_ratio[1]
    sampled_whole_original_seq_ids = [sample_ratio[1] * i for i in range(sampled_original_seq_num)]
    sampled_whole_original_seq = original_seq[sampled_whole_original_seq_ids, 0]
    sampled_len = sampled_whole_original_seq.size(0)
    sampled_whole_original_seq = sampled_whole_original_seq.unsqueeze(dim=0)
    sampled_whole_original_seq = sampled_whole_original_seq.unsqueeze(dim=0)
    # stretch the sampled original sequence to the same density of target sequence
    stretched_original_seq = torch.nn.functional.interpolate(sampled_whole_original_seq, size=int(sampled_len*sample_ratio[0]), mode='linear')
    stretched_original_seq = stretched_original_seq.squeeze()
    stretched_original_seq = stretched_original_seq.squeeze()
    normal_seq_len = original_seq.size(0)
    if stretched_original_seq.size(0) >= normal_seq_len:
        stretched_original_seq = stretched_original_seq[:normal_seq_len]
    else:
        zerotensor = torch.zeros(normal_seq_len - stretched_original_seq.size(0)).to(device)
        stretched_original_seq = torch.hstack([stretched_original_seq, zerotensor])

    return sampled_target_seq, sampled_original_seq, stretched_original_seq

def cutoff_zeros(seq):
    real_seq_len = torch.nonzero(seq[:, 0]).size(0)
    return seq[:real_seq_len]

class Trainer():

    def __init__(self, lr, n_epochs, device, patience, lamda, alpha, model_name, retreival_set, scale_ratios=None,
                 parts_num_from_each_len=120, default_target_seq_len=100, train_retrieval_size=10, except_ratios=None):
        """
        Args:
            lr (float): Learning rate
            n_epochs (int): The number of training epoch
            device: 'cuda' or 'cpu'
            patience (int): How long to wait after last time validation loss improved.
            lamda (float): The weight of RUL loss
            alpha (List: [float]): The weights of Capacity loss
            model_name (str): The model save path
        """
        self.lr = lr
        self.n_epochs = n_epochs
        self.device = device
        self.patience = patience
        self.model_name = model_name
        self.lamda = lamda
        self.alpha = alpha
        self.retreival_set = retreival_set

        self.scale_ratios = [1] if scale_ratios is None else scale_ratios  # 1/3., 1/2.5, 1/2., 1/1.5, 1, 1.5, 2, 2.5, 3
        self.parts_num_from_each_len = parts_num_from_each_len
        # self.retrieval_loader, self.retrieval_tensor = self.preprocess_retreival_set(retrieval_batch_size=retrieval_batch_size)
        # self.sample_ratio_pairs = [[1, 1]]#[[1, 1], [1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2], [3, 4], [4, 3]]
        self.end_cap = 880/1190
        self.tolerable_target_seq_lens = [default_target_seq_len]
        self.retrieval_feas, self.retreival_ruls = self.get_retrieval_parts(
            selected_full_seqs=self.retreival_set, target_part_len=default_target_seq_len)
        self.train_retrieval_size = train_retrieval_size

        self.except_ratios = except_ratios
        """retrieval_feas: [len(zoom_retrieval_seq_ratios) × retrieval_battery_num × seq_num × seq_len × feature_num]"""
        # import pdb;pdb.set_trace()
    def get_retrieval_parts(self, selected_full_seqs, target_part_len):
        '''
        selected_seqs: retrieval degradation seqs except the training part
        part_lens: [0.33*target_len + i * 0.1 for i in range(26)]

        return:
        feas: [parts_num_from_each_len*part_lens[i]*feature_num for i in range(len(part_lens[i]))]
        ruls: parts_num_from_each_len X len(part_lens)
        '''
        feature_num = selected_full_seqs[0][0].shape[1]
        feas, ruls = [], []
        for scale_ratio in self.scale_ratios:
            # part_len = int(target_part_len * part_len_ratio)
            all_feas, rul_lbls = [], []
            for selected_full_seq_id in range(len(selected_full_seqs)):
                # full_seq_len = round(rul_factor*selected_full_seqs[selected_full_seq_id][1][0])
                cp_selected_full_seqs = copy.deepcopy(selected_full_seqs[selected_full_seq_id][0])

                cp_selected_full_seqs = cp_selected_full_seqs[scale_ratio-1::scale_ratio, :]
                print('retrieval set size:', cp_selected_full_seqs.shape)
                if cp_selected_full_seqs.shape[0] >= target_part_len:
                    cp_selected_full_seqs_ruls = copy.deepcopy(selected_full_seqs[selected_full_seq_id][1])
                    cp_selected_full_seqs_ruls = cp_selected_full_seqs_ruls[scale_ratio-1::scale_ratio]
                    cp_selected_full_seqs_ruls = cp_selected_full_seqs_ruls / scale_ratio
                    sliced_parts = np.lib.stride_tricks.sliding_window_view(cp_selected_full_seqs,
                                                                            (target_part_len, feature_num))

                    sliced_parts_ruls = cp_selected_full_seqs_ruls[target_part_len - 1:]
                    sliced_parts = sliced_parts.squeeze(1)
                    # rul_factor = 1 / part_len_ratio
                    sliced_parts_ruls = np.array(sliced_parts_ruls).astype(float)
                else:
                    sliced_parts = np.array([])
                    sliced_parts_ruls = np.array([])
                print('sequence number and rul number for scale ratio{} for sequence{}: {} , {}'.format(scale_ratio, selected_full_seq_id, sliced_parts.shape, sliced_parts_ruls.shape))
                all_feas.append(sliced_parts)
                rul_lbls.append(sliced_parts_ruls)
            # all_feas = np.vstack(all_feas)

            feas.append(all_feas)
            ruls.append(rul_lbls)
        # import pdb;pdb.set_trace()
        return feas, ruls

    def randomly_sample_parts(self, batteryidx):
        '''this func takes too much time'''
        feas, ruls = [], []
        for zoom_ratio_idx in range(len(self.retrieval_feas)):
            tmp_feas_pool, tmp_rul_pool = [], []
            for retrieval_battery_idx in range(len(self.retrieval_feas[zoom_ratio_idx])):
                # filter out the battery which the target sequence belongs to
                if retrieval_battery_idx == batteryidx:
                    continue
                tmp_feas_pool.append(self.retrieval_feas[zoom_ratio_idx][retrieval_battery_idx])
                tmp_rul_pool.append(self.retreival_ruls[zoom_ratio_idx][retrieval_battery_idx])
            tmp_feas_pool = np.vstack(tmp_feas_pool)
            tmp_rul_pool = np.hstack(tmp_rul_pool)
            choices = np.random.choice(tmp_feas_pool.shape[0], self.parts_num_from_each_len, replace=False)
            feas.append(tmp_feas_pool[choices, :, :])
            ruls.append(tmp_rul_pool[choices])
        return feas, ruls


    def randomly_sample_partsv2(self, batteryidx):
        feas, ruls = [], []
        # if training:
        parts_num_per_battery = int(self.parts_num_from_each_len / self.train_retrieval_size)
        candidate_battery_ids = []
        for i in range(len(self.retrieval_feas[0])):
            if i != batteryidx:
                candidate_battery_ids.append(i)
        sampled_battery_ids = np.random.choice(candidate_battery_ids, self.train_retrieval_size, replace=False)
        # else:
        #     # parts_num_per_battery = int(self.parts_num_from_each_len / (len(self.retrieval_feas[0]) - 1))
        #     sampled_battery_ids = [i for i in range(len(self.retrieval_feas[0]))]
        for zoom_ratio_idx in range(len(self.retrieval_feas)):
            tmp_feas_pool, tmp_rul_pool = [], []
            for retrieval_battery_idx in sampled_battery_ids:
                if parts_num_per_battery < self.retrieval_feas[zoom_ratio_idx][retrieval_battery_idx].shape[0]:
                    choices = np.random.choice(self.retrieval_feas[zoom_ratio_idx][retrieval_battery_idx].shape[0],
                                               parts_num_per_battery, replace=False)
                    tmp_feas_pool.append(self.retrieval_feas[zoom_ratio_idx][retrieval_battery_idx][choices, :, :])
                    tmp_rul_pool.append(self.retreival_ruls[zoom_ratio_idx][retrieval_battery_idx][choices])
                elif self.retrieval_feas[zoom_ratio_idx][retrieval_battery_idx].shape[0] > 0:
                    tmp_feas_pool.append(self.retrieval_feas[zoom_ratio_idx][retrieval_battery_idx])
                    tmp_rul_pool.append(self.retreival_ruls[zoom_ratio_idx][retrieval_battery_idx])
                # else:
                #     tmp_feas_pool.append(self.retrieval_feas[zoom_ratio_idx][retrieval_battery_idx])
                #     tmp_rul_pool.append(self.retreival_ruls[zoom_ratio_idx][retrieval_battery_idx])
            tmp_feas_pool = np.vstack(tmp_feas_pool)
            tmp_rul_pool = np.hstack(tmp_rul_pool)

            feas.append(tmp_feas_pool)
            ruls.append(tmp_rul_pool)
        return feas, ruls

    def generate_encoded_database(self, encoder, stride=2, end_cyc=500, batchsize=200):
        '''
        used when testing, the encoder has been trained well, so it can encode all the series in retrieval set and wait
        the target sequence to appear
        '''
        encoded_feas, ruls = [], []
        for zoom_ratio_idx in range(len(self.retrieval_feas)):
            tmp_feas_pool, tmp_rul_pool, tmp_encoded_feas_pool = [], [], []
            print('ratio', zoom_ratio_idx)

            for retrieval_battery_idx in range(len(self.retrieval_feas[0])):
                if self.retrieval_feas[zoom_ratio_idx][retrieval_battery_idx].shape[0] > 0:
                    sampled_feas = self.retrieval_feas[zoom_ratio_idx][retrieval_battery_idx][(stride-1)::stride, :, :]
                    tmp_feas_pool.append(sampled_feas[:end_cyc])
                    sampled_ruls = self.retreival_ruls[zoom_ratio_idx][retrieval_battery_idx][(stride-1)::stride]
                    tmp_rul_pool.append(sampled_ruls[:end_cyc])

            # batchwise inference because of the limited gpu memory
            tmp_feas_pool = torch.Tensor(np.vstack(tmp_feas_pool))#.to(self.device)
            seqnum = 0
            while seqnum < tmp_feas_pool.size(0):
                if seqnum + batchsize < tmp_feas_pool.size(0):
                    tmp_feas = tmp_feas_pool[seqnum:seqnum+batchsize, :, :].to(self.device)
                else:
                    tmp_feas = tmp_feas_pool[seqnum:, :, :].to(self.device)
                encoded_tmp_feas = encoder(tmp_feas)
                seqnum += batchsize
                tmp_encoded_feas_pool.append(encoded_tmp_feas)
                # del tmp_feas_pool
            tmp_rul_pool = np.hstack(tmp_rul_pool)
            ruls.append(torch.Tensor(tmp_rul_pool))
            encoded_feas.append(torch.vstack(tmp_encoded_feas_pool))
        # ruls = torch.Tensor(np.hstack(ruls)).to(self.device)
        # encoded_feas = torch.vstack(encoded_feas).to(self.device)
        # ruls = torch.reshape(ruls, (-1, 1))
        return encoded_feas, ruls


    def select_source_seqs(self, battery_seq, batteryidx):
        seqs, batteryids = [], []
        for k, v in self.retreival_set.items():
            if k != batteryidx:
                seqs.append(v)
                batteryids.append(k)
        battery_seq_len = battery_seq.size(1)
        feas, ruls = self.get_retrieval_parts(seqs, battery_seq_len)
        return feas, ruls


    def train(self, train_loader, valid_loader, encoder, relationmodel, wandb, save_path):
        # model = model.to(self.device)
        device = self.device
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=self.lr)
        encoder_lr_scheduler = StepLR(encoder_optimizer, step_size=100, gamma=0.95)
        relationmodel_optimizer = optim.Adam(relationmodel.parameters(), lr=self.lr, )
        relation_lr_scheduler = StepLR(relationmodel_optimizer, step_size=100, gamma=0.98)

        loss_fn = nn.MSELoss().to(self.device)
        # loss_fn = torch.nn.L1Loss()

        # Training
        train_loss = []
        valid_loss = []

        for epoch in range(self.n_epochs):

            # print('training epoch:', epoch)
            # encoder_lr_scheduler.step(epoch)
            # relation_lr_scheduler.step(epoch)
            # encoder.train()
            # relationmodel.train()
            # # y_true, y_pred = [], []
            # # train_losses = []
            # for step, (x, y) in enumerate(train_loader):
            #     x = x.to(device)
            #     loss = 0
            #     for batch_battery_idx in range(y.size(0)):
            #         batteryidx = int(y[batch_battery_idx][2].item())
            #         seqs, ruls = self.randomly_sample_partsv2(batteryidx)
            #         all_scores, all_ruls = [], []
            #         # scale target source
            #         for scaleidx in range(len(self.scale_ratios)):
            #
            #             target_scale_ratio = self.scale_ratios[scaleidx]
            #             scaled_target = x[batch_battery_idx][target_scale_ratio-1::target_scale_ratio, :]
            #             scaled_target = scaled_target.unsqueeze(dim=0)
            #             encoded_target = encoder(scaled_target)
            #
            #             for sampleratioidx in range(len(seqs)):
            #                 ratio_pair = [sampleratioidx+1, scaleidx+1]
            #                 if ratio_pair in self.except_ratios:
            #                     continue
            #                 # print(ratio_pair)
            #                 # if sampleratioidx != 0 and scaleidx == sampleratioidx:
            #                 #     continue
            #                 tensor_seq = torch.Tensor(seqs[sampleratioidx]).cuda()
            #                 encoded_source = encoder(tensor_seq)
            #                 if encoded_source.size() != encoded_target.size():
            #                     unsqueezed_encoded_target = encoded_target.repeat(encoded_source.size(0), 1)
            #                 else:
            #                     unsqueezed_encoded_target = encoded_target
            #                 relation_scores = relationmodel(encoded_source, unsqueezed_encoded_target)
            #                 all_scores.append(relation_scores)
            #
            #                 all_ruls.append(ruls[sampleratioidx] * target_scale_ratio)
            #
            #         all_scores = torch.hstack(all_scores)
            #         all_ruls = torch.Tensor(np.hstack(all_ruls)).cuda()
            #         all_ruls = all_ruls.reshape(-1, 1)
            #
            #         all_scores = F.softmax(all_scores, dim=0)
            #         # all_scores = all_scores / torch.sum(all_scores)
            #         # import pdb;pdb.set_trace()
            #         all_scores = all_scores.unsqueeze(dim=0)
            #         predicted_rul = torch.mm(all_scores, all_ruls)
            #         loss += loss_fn(predicted_rul, y[batch_battery_idx][0].cuda())
            #
            #     loss /= y.size(0)
            #     encoder_optimizer.zero_grad()
            #     relationmodel_optimizer.zero_grad()
            #     loss.backward()
            #     encoder_optimizer.step()
            #     relationmodel_optimizer.step()
            #     train_loss.append(loss.cpu().detach().numpy())
            #
            #     if step % 50 == 0:
            #         print('step:', step, 'train loss:', train_loss[-1], np.average(train_loss))
            #         wandb.log({'loss:': train_loss[-1], 'epoch': epoch})


            print('started to evaluate')

            encoder.eval()
            relationmodel.eval()
            y_true, y_pred = [], []
            with torch.no_grad():

                encoded_source, ruls = self.generate_encoded_database(encoder)

                for step, (x, y) in enumerate(valid_loader):
                    assert y.size(0) == 1
                    x = x.to(device)
                    all_scores, all_ruls = [], []

                    for scaleidx in range(len(self.scale_ratios)):
                        target_scale_ratio = self.scale_ratios[scaleidx]
                        scaled_target = x[:, target_scale_ratio-1::target_scale_ratio, :]
                        encoded_target = encoder(scaled_target)
                        for source_scale_idx in range(len(encoded_source)):
                            ratio_pair = [source_scale_idx + 1, scaleidx + 1]
                            if ratio_pair in self.except_ratios:
                                continue
                            # print(ratio_pair, target_scale_ratio)
                            if encoded_target.size() != encoded_source[source_scale_idx].size():
                                expanded_encoded_target = encoded_target.repeat(encoded_source[source_scale_idx].size(0), 1)
                            relation_scores = relationmodel(encoded_source[source_scale_idx], expanded_encoded_target)
                            all_scores.append(relation_scores)
                            all_ruls.append(ruls[source_scale_idx] * target_scale_ratio)
                            del expanded_encoded_target
                            encoded_source[source_scale_idx].cpu()
                    all_scores = torch.hstack(all_scores)
                    all_ruls = torch.hstack(all_ruls).to(device)

                    maxscores, maxidx = torch.topk(all_scores, 1000)  # 1000
                    selected_ruls = all_ruls[maxidx]
                    # maxscores = all_scores
                    # selected_ruls = all_ruls

                    maxscores = F.softmax(maxscores, dim=0)
                    # maxscores = maxscores / torch.sum(maxscores)
                    maxscores = maxscores.unsqueeze(dim=0)

                    selected_ruls = selected_ruls.reshape(-1, 1)
                    predicted_rul = torch.mm(maxscores, selected_ruls)
                    if step % 100 == 0:
                        print(predicted_rul, y[0][0])
                    y_true.append(y[0][0]*3000)
                    y_pred.append(predicted_rul[0][0].item()*3000)
                    # import pdb;pdb.set_trace()

                error=0
                for i in range(len(y_true)):
                    error += abs(y_true[i]-y_pred[i])/y_true[i]
                print('error:', error/len(y_true))

                y_true = torch.Tensor(y_true)
                y_pred = torch.Tensor(y_pred)
                import matplotlib.pyplot as plt
                plt.plot([i for i in range(len(y_true))], y_true)
                plt.plot([i for i in range(len(y_true))], y_pred)
                plt.show()
                epoch_loss = torch.nn.L1Loss()(y_true, y_pred)
                print(epoch_loss)
                valid_loss.append(epoch_loss)

                # if self.n_epochs > 10:
                if epoch % 1 == 0:
                    print('Epoch number : ', epoch)
                    print(f'-- "train" loss {train_loss[-1]:.4}', f'-- "valid" loss {epoch_loss:.4}')
                    # torch.save(relationmodel.state_dict(), 'VITrelationmodel.pth')
                    name = save_path + '/LSTM_relu_b_32_' + str(int(epoch_loss.item())) + '.pth'
                    torch.save(encoder.state_dict(), name)
                else:
                    print(f'-- "train" loss {train_loss[-1]:.4}', f'-- "valid" loss {epoch_loss:.4}')

    def test(self, test_loader, model):
        model = model.to(self.device)
        device = self.device

        y_true, y_pred, soh_true, soh_pred = [], [], [], []
        model.eval()
        with torch.no_grad():
            for step, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                y_, soh_ = model(x)

                y_pred.append(y_)
                y_true.append(y[:, 0])
                soh_pred.append(soh_)
                soh_true.append(y[:, 1:])

            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)
            soh_true = torch.cat(soh_true, axis=0)
            soh_pred = torch.cat(soh_pred, axis=0)
            mse_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        return y_true, y_pred, mse_loss, soh_true, soh_pred




# import os
# import time
# import pickle
# import math
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import interpolate
# from datetime import datetime
# import pandas as pd
# # from tool import EarlyStopping
# # from sklearn.metrics import roc_auc_score, mean_squared_error
# import copy
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
# from torch import nn
# from torch.autograd import Variable
# from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
# from torch.utils.data.sampler import RandomSampler
# torch.autograd.set_detect_anomaly(True)
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# import warnings
#
# warnings.filterwarnings('ignore')
#
# def seed_torch(seed=1029):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#
# def compress_seqv2(seq, compress_ratio):
#     last_idx = seq.size(0) - 1
#     sample_ids = [last_idx]
#     while last_idx > 0:
#         last_idx -= compress_ratio
#         sample_ids.append(last_idx)
#     sample_ids.sort()
#     if sample_ids[0] < 0:
#         sample_ids = sample_ids[1:]
#     sampled_seq = seq[sample_ids, :]
#
#
#
# def compress_seq(target_seq, original_seq, sample_ratio, device):
#     '''
#     sample_ratio[0]:target sample ratio, sample_ratio[1]:original sample ratio
#     '''
#     target_sample_last_ids = target_seq.size(0) - 1
#     target_sample_ids = [target_sample_last_ids]
#     while target_sample_last_ids > 0:
#         target_sample_last_ids -= sample_ratio[0]
#         target_sample_ids.append(target_sample_last_ids)
#     target_sample_ids.sort()
#     if target_sample_ids[0] < 0:
#         target_sample_ids = target_sample_ids[1:]
#
#     sampled_target_seq = target_seq[target_sample_ids, :]
#
#     original_sample_ids = [int(sample_ratio[1] * i) for i in range(len(target_sample_ids))]
#     sampled_original_seq = original_seq[original_sample_ids, :]
#     # import pdb;pdb.set_trace()
#
#     sampled_original_seq_num = original_seq.size(0) // sample_ratio[1]
#     sampled_whole_original_seq_ids = [sample_ratio[1] * i for i in range(sampled_original_seq_num)]
#     sampled_whole_original_seq = original_seq[sampled_whole_original_seq_ids, 0]
#     sampled_len = sampled_whole_original_seq.size(0)
#     sampled_whole_original_seq = sampled_whole_original_seq.unsqueeze(dim=0)
#     sampled_whole_original_seq = sampled_whole_original_seq.unsqueeze(dim=0)
#     # stretch the sampled original sequence to the same density of target sequence
#     stretched_original_seq = torch.nn.functional.interpolate(sampled_whole_original_seq, size=int(sampled_len*sample_ratio[0]), mode='linear')
#     stretched_original_seq = stretched_original_seq.squeeze()
#     stretched_original_seq = stretched_original_seq.squeeze()
#     normal_seq_len = original_seq.size(0)
#     if stretched_original_seq.size(0) >= normal_seq_len:
#         stretched_original_seq = stretched_original_seq[:normal_seq_len]
#     else:
#         zerotensor = torch.zeros(normal_seq_len - stretched_original_seq.size(0)).to(device)
#         stretched_original_seq = torch.hstack([stretched_original_seq, zerotensor])
#
#     return sampled_target_seq, sampled_original_seq, stretched_original_seq
#
# def cutoff_zeros(seq):
#     real_seq_len = torch.nonzero(seq[:, 0]).size(0)
#     return seq[:real_seq_len]
#
# class Trainer():
#
#     def __init__(self, lr, n_epochs, device, patience, lamda, alpha, model_name, retreival_set, sample_len_ratios=None,
#                  parts_num_from_each_len=120, default_target_seq_len=100, train_retrieval_size=10):
#         """
#         Args:
#             lr (float): Learning rate
#             n_epochs (int): The number of training epoch
#             device: 'cuda' or 'cpu'
#             patience (int): How long to wait after last time validation loss improved.
#             lamda (float): The weight of RUL loss
#             alpha (List: [float]): The weights of Capacity loss
#             model_name (str): The model save path
#         """
#         self.lr = lr
#         self.n_epochs = n_epochs
#         self.device = device
#         self.patience = patience
#         self.model_name = model_name
#         self.lamda = lamda
#         self.alpha = alpha
#         self.retreival_set = retreival_set
#
#         self.part_len_ratios = [1] if sample_len_ratios is None else sample_len_ratios  # 1/3., 1/2.5, 1/2., 1/1.5, 1, 1.5, 2, 2.5, 3
#         self.parts_num_from_each_len = parts_num_from_each_len
#         # self.retrieval_loader, self.retrieval_tensor = self.preprocess_retreival_set(retrieval_batch_size=retrieval_batch_size)
#         # self.sample_ratio_pairs = [[1, 1]]#[[1, 1], [1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2], [3, 4], [4, 3]]
#         self.end_cap = 880/1190
#         self.tolerable_target_seq_lens = [default_target_seq_len]
#         self.retrieval_feas, self.retreival_ruls = self.get_retrieval_parts(
#             selected_full_seqs=self.retreival_set, target_part_len=default_target_seq_len)
#         self.train_retrieval_size = train_retrieval_size
#         """retrieval_feas: [len(zoom_retrieval_seq_ratios) × retrieval_battery_num × seq_num × seq_len × feature_num]"""
#         # import pdb;pdb.set_trace()
#
#
#     def randomly_sample_parts(self, batteryidx):
#         '''this func takes too much time'''
#         feas, ruls = [], []
#         for zoom_ratio_idx in range(len(self.retrieval_feas)):
#             tmp_feas_pool, tmp_rul_pool = [], []
#             for retrieval_battery_idx in range(len(self.retrieval_feas[zoom_ratio_idx])):
#                 # filter out the battery which the target sequence belongs to
#                 if retrieval_battery_idx == batteryidx:
#                     continue
#                 tmp_feas_pool.append(self.retrieval_feas[zoom_ratio_idx][retrieval_battery_idx])
#                 tmp_rul_pool.append(self.retreival_ruls[zoom_ratio_idx][retrieval_battery_idx])
#             tmp_feas_pool = np.vstack(tmp_feas_pool)
#             tmp_rul_pool = np.hstack(tmp_rul_pool)
#             choices = np.random.choice(tmp_feas_pool.shape[0], self.parts_num_from_each_len, replace=False)
#             feas.append(tmp_feas_pool[choices, :, :])
#             ruls.append(tmp_rul_pool[choices])
#         return feas, ruls
#
#
#     def randomly_sample_partsv2(self, batteryidx):
#         feas, ruls = [], []
#         # if training:
#         parts_num_per_battery = int(self.parts_num_from_each_len / self.train_retrieval_size)
#         candidate_battery_ids = []
#         for i in range(len(self.retrieval_feas[0])):
#             if i != batteryidx:
#                 candidate_battery_ids.append(i)
#         sampled_battery_ids = np.random.choice(candidate_battery_ids, self.train_retrieval_size, replace=False)
#         # else:
#         #     # parts_num_per_battery = int(self.parts_num_from_each_len / (len(self.retrieval_feas[0]) - 1))
#         #     sampled_battery_ids = [i for i in range(len(self.retrieval_feas[0]))]
#         for zoom_ratio_idx in range(len(self.retrieval_feas)):
#             tmp_feas_pool, tmp_rul_pool = [], []
#             for retrieval_battery_idx in sampled_battery_ids:
#                 # if training:
#                 choices = np.random.choice(self.retrieval_feas[zoom_ratio_idx][retrieval_battery_idx].shape[0],
#                                            parts_num_per_battery, replace=False)
#                 tmp_feas_pool.append(self.retrieval_feas[zoom_ratio_idx][retrieval_battery_idx][choices, :, :])
#                 tmp_rul_pool.append(self.retreival_ruls[zoom_ratio_idx][retrieval_battery_idx][choices])
#                 # else:
#                 #     tmp_feas_pool.append(self.retrieval_feas[zoom_ratio_idx][retrieval_battery_idx])
#                 #     tmp_rul_pool.append(self.retreival_ruls[zoom_ratio_idx][retrieval_battery_idx])
#             tmp_feas_pool = np.vstack(tmp_feas_pool)
#             tmp_rul_pool = np.hstack(tmp_rul_pool)
#             feas.append(tmp_feas_pool)
#             ruls.append(tmp_rul_pool)
#         return feas, ruls
#
#     def generate_encoded_database(self, encoder, stride=2, end_cyc=500, batchsize=200):
#         encoded_feas, ruls = [], []
#         for zoom_ratio_idx in range(len(self.retrieval_feas)):
#             tmp_feas_pool, tmp_rul_pool = [], []
#             print('ratio', zoom_ratio_idx)
#             for retrieval_battery_idx in range(len(self.retrieval_feas[0])):
#                 sampled_feas = self.retrieval_feas[zoom_ratio_idx][retrieval_battery_idx][(stride-1)::stride, :, :]
#                 tmp_feas_pool.append(sampled_feas[:end_cyc])
#                 sampled_ruls = self.retreival_ruls[zoom_ratio_idx][retrieval_battery_idx][(stride-1)::stride]
#                 tmp_rul_pool.append(sampled_ruls[:end_cyc])
#
#             tmp_feas_pool = torch.Tensor(np.vstack(tmp_feas_pool))#.to(self.device)
#             seqnum = 0
#             while seqnum < tmp_feas_pool.size(0):
#                 if seqnum + batchsize < tmp_feas_pool.size(0):
#                     tmp_feas = tmp_feas_pool[seqnum:seqnum+batchsize, :, :].to(self.device)
#                 else:
#                     tmp_feas = tmp_feas_pool[seqnum:, :, :].to(self.device)
#                 encoded_tmp_feas = encoder(tmp_feas)
#                 seqnum += batchsize
#                 encoded_feas.append(encoded_tmp_feas)
#                 # del tmp_feas_pool
#             tmp_rul_pool = np.hstack(tmp_rul_pool)
#             ruls.append(tmp_rul_pool)
#         ruls = torch.Tensor(np.hstack(ruls)).to(self.device)
#         encoded_feas = torch.vstack(encoded_feas).to(self.device)
#         ruls = torch.reshape(ruls, (-1, 1))
#         return encoded_feas, ruls
#
#     def get_retrieval_parts(self, selected_full_seqs, target_part_len):
#         '''
#         selected_seqs: retrieval degradation seqs except the training part
#         part_lens: [0.33*target_len + i * 0.1 for i in range(26)]
#
#         return:
#         feas: [parts_num_from_each_len*part_lens[i]*feature_num for i in range(len(part_lens[i]))]
#         ruls: parts_num_from_each_len X len(part_lens)
#         '''
#         feature_num = selected_full_seqs[0][0].shape[1]
#         feas, ruls = [], []
#         for part_len_ratio in self.part_len_ratios:
#             part_len = int(target_part_len * part_len_ratio)
#             all_feas, rul_lbls = [], []
#             for selected_full_seq_id in range(len(selected_full_seqs)):
#                 sliced_parts = np.lib.stride_tricks.sliding_window_view(selected_full_seqs[selected_full_seq_id][0],
#                                                                         (part_len, feature_num))
#                 cp_selected_full_seqs_ruls = copy.deepcopy(selected_full_seqs[selected_full_seq_id][1])
#                 sliced_parts_ruls = cp_selected_full_seqs_ruls[part_len - 1:]
#                 sliced_parts = sliced_parts.squeeze()
#                 rul_factor = 1 / part_len_ratio
#                 sliced_parts_ruls = np.array(sliced_parts_ruls).astype(float)
#                 sliced_parts_ruls *= rul_factor
#                 all_feas.append(sliced_parts)
#                 rul_lbls.append(sliced_parts_ruls)
#             # all_feas = np.vstack(all_feas)
#
#
#             feas.append(all_feas)
#             ruls.append(rul_lbls)
#         # import pdb;pdb.set_trace()
#         return feas, ruls
#
#
#     def select_source_seqs(self, battery_seq, batteryidx):
#         seqs, batteryids = [], []
#         for k, v in self.retreival_set.items():
#             if k != batteryidx:
#                 seqs.append(v)
#                 batteryids.append(k)
#         battery_seq_len = battery_seq.size(1)
#         feas, ruls = self.get_retrieval_parts(seqs, battery_seq_len)
#         return feas, ruls
#
#
#     def train(self, train_loader, valid_loader, encoder, relationmodel):
#         # model = model.to(self.device)
#         device = self.device
#         encoder_optimizer = optim.Adam(encoder.parameters(), lr=self.lr)
#         encoder_lr_scheduler = StepLR(encoder_optimizer, step_size=100, gamma=0.98)
#         relationmodel_optimizer = optim.Adam(relationmodel.parameters(), lr=self.lr, )
#         relation_lr_scheduler = StepLR(relationmodel_optimizer, step_size=100, gamma=0.98)
#
#         loss_fn = nn.MSELoss().to(self.device)
#         # loss_fn = torch.nn.L1Loss()
#
#         # Training
#         train_loss = []
#         valid_loss = []
#
#         for epoch in range(self.n_epochs):
#             # encoder.load_state_dict(torch.load('output/LSTM_larger_candidates_b_32_235.pth'))
#             print('training epoch:', epoch)
#             encoder_lr_scheduler.step(epoch)
#             relation_lr_scheduler.step(epoch)
#             encoder.train()
#             relationmodel.train()
#             # y_true, y_pred = [], []
#             # train_losses = []
#             for step, (x, y) in enumerate(train_loader):
#                 x = x.to(device)
#                 loss = 0
#                 for batch_battery_idx in range(y.size(0)):
#                     batteryidx = int(y[batch_battery_idx][2].item())
#                     seqs, ruls = self.randomly_sample_partsv2(batteryidx)
#                     all_scores = []
#                     target = x[batch_battery_idx].unsqueeze(dim=0)
#                     encoded_target = encoder(target)
#                     for sampleratioidx in range(len(seqs)):
#                         tensor_seq = torch.Tensor(seqs[sampleratioidx]).cuda()
#                         encoded_source = encoder(tensor_seq)
#                         if encoded_source.size() != encoded_target.size():
#                             encoded_target = encoded_target.repeat(encoded_source.size(0), 1)
#                         relation_scores = relationmodel(encoded_source, encoded_target)
#                         all_scores.append(relation_scores)
#                     all_scores = torch.hstack(all_scores)
#                     all_ruls = torch.Tensor(np.vstack(ruls)).cuda()
#                     all_ruls = all_ruls.reshape(-1, 1)
#                     scores = F.softmax(all_scores, dim=0)
#                     scores = scores.unsqueeze(dim=0)
#                     # import pdb; pdb.set_trace()
#                     # synthesized_seq = torch.mm(scores, all_transformed_sohs)
#                     predicted_rul = torch.mm(scores, all_ruls)
#                     loss += loss_fn(predicted_rul, y[batch_battery_idx][0].cuda())
#
#                 loss /= y.size(0)
#                 encoder_optimizer.zero_grad()
#                 relationmodel_optimizer.zero_grad()
#                 loss.backward()
#                 encoder_optimizer.step()
#                 relationmodel_optimizer.step()
#                 train_loss.append(loss.cpu().detach().numpy())
#
#                 if step % 50 == 0:
#                     print('step:', step, 'train loss:', train_loss[-1], np.average(train_loss))
#
#
#             print('started to evaluate')
#             encoder.eval()
#             relationmodel.eval()
#             y_true, y_pred = [], []
#             with torch.no_grad():
#
#                 encoded_source, ruls = self.generate_encoded_database(encoder)
#
#                 for step, (x, y) in enumerate(valid_loader):
#                     assert y.size(0) == 1
#                     x = x.to(device)
#                     encoded_target = encoder(x)
#                     encoded_target = encoded_target.repeat(encoded_source.size(0), 1)
#                     relation_scores = relationmodel(encoded_source, encoded_target)
#
#                     maxscores, maxidx = torch.topk(relation_scores, 100)  # 1000
#                     # maxidx = maxidx[0]
#                     selected_ruls = ruls[maxidx, :]
#                     maxscores = F.softmax(maxscores, dim=0)
#                     maxscores = maxscores.unsqueeze(dim=0)
#                     # import pdb;pdb.set_trace()
#                     predicted_rul = torch.mm(maxscores, selected_ruls)
#                     if step % 100 == 0:
#                         print(predicted_rul, y[0][0])
#                     y_true.append(y[0][0]*3000)
#                     y_pred.append(predicted_rul[0][0].item()*3000)
#                     # import pdb;pdb.set_trace()
#
#                 error=0
#                 for i in range(len(y_true)):
#                     error += abs(y_true[i]-y_pred[i])/y_true[i]
#                 print('error:', error/len(y_true))
#
#                 y_true = torch.Tensor(y_true)
#                 y_pred = torch.Tensor(y_pred)
#                 import matplotlib.pyplot as plt
#                 plt.plot([i for i in range(len(y_true))], y_true)
#                 plt.plot([i for i in range(len(y_true))], y_pred)
#                 plt.show()
#                 epoch_loss = torch.nn.L1Loss()(y_true, y_pred)
#                 print(epoch_loss)
#                 valid_loss.append(epoch_loss)
#
#                 # if self.n_epochs > 10:
#                 if epoch % 1 == 0:
#                     print('Epoch number : ', epoch)
#                     print(f'-- "train" loss {train_loss[-1]:.4}', f'-- "valid" loss {epoch_loss:.4}')
#                     # torch.save(relationmodel.state_dict(), 'VITrelationmodel.pth')
#                     name='output/FFN/fnn' + str(int(epoch_loss.item())) + '.pth'
#                     torch.save(encoder.state_dict(), name)
#                 else:
#                     print(f'-- "train" loss {train_loss[-1]:.4}', f'-- "valid" loss {epoch_loss:.4}')
#
#     def test(self, test_loader, model):
#         model = model.to(self.device)
#         device = self.device
#
#         y_true, y_pred, soh_true, soh_pred = [], [], [], []
#         model.eval()
#         with torch.no_grad():
#             for step, (x, y) in enumerate(test_loader):
#                 x = x.to(device)
#                 y = y.to(device)
#                 y_, soh_ = model(x)
#
#                 y_pred.append(y_)
#                 y_true.append(y[:, 0])
#                 soh_pred.append(soh_)
#                 soh_true.append(y[:, 1:])
#
#             y_true = torch.cat(y_true, axis=0)
#             y_pred = torch.cat(y_pred, axis=0)
#             soh_true = torch.cat(soh_true, axis=0)
#             soh_pred = torch.cat(soh_pred, axis=0)
#             mse_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
#         return y_true, y_pred, mse_loss, soh_true, soh_pred
#
#
