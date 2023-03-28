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
from torch.utils.data import Sampler
from load_ne import interp, data_aug
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

    original_sample_ids = [
        int(sample_ratio[1] * i) for i in range(len(target_sample_ids))
    ]
    sampled_original_seq = original_seq[original_sample_ids, :]
    # import pdb;pdb.set_trace()

    sampled_original_seq_num = original_seq.size(0) // sample_ratio[1]
    sampled_whole_original_seq_ids = [
        sample_ratio[1] * i for i in range(sampled_original_seq_num)
    ]
    sampled_whole_original_seq = original_seq[sampled_whole_original_seq_ids,
                                              0]
    sampled_len = sampled_whole_original_seq.size(0)
    sampled_whole_original_seq = sampled_whole_original_seq.unsqueeze(dim=0)
    sampled_whole_original_seq = sampled_whole_original_seq.unsqueeze(dim=0)
    # stretch the sampled original sequence to the same density of target sequence
    stretched_original_seq = torch.nn.functional.interpolate(
        sampled_whole_original_seq,
        size=int(sampled_len * sample_ratio[0]),
        mode='linear')
    stretched_original_seq = stretched_original_seq.squeeze()
    stretched_original_seq = stretched_original_seq.squeeze()
    normal_seq_len = original_seq.size(0)
    if stretched_original_seq.size(0) >= normal_seq_len:
        stretched_original_seq = stretched_original_seq[:normal_seq_len]
    else:
        zerotensor = torch.zeros(normal_seq_len -
                                 stretched_original_seq.size(0)).to(device)
        stretched_original_seq = torch.hstack(
            [stretched_original_seq, zerotensor])

    return sampled_target_seq, sampled_original_seq, stretched_original_seq


def cutoff_zeros(seq):
    real_seq_len = torch.nonzero(seq[:, 0]).size(0)
    return seq[:real_seq_len]


def contrastive_loss(source, pos_sample, tao):
    s = datetime.now()
    assert source.shape[0] == pos_sample.shape[0]
    N = source.shape[0]

    def sim(tensor1, tensor2):
        if tensor1.shape != tensor2.shape:
            tensor1 = tensor1.reshape(1, -1)
            return torch.cosine_similarity(tensor1, tensor2)
        else:
            return torch.cosine_similarity(tensor1, tensor2, dim=0)

    def _l(i, type):
        denominator = 0
        if type == 'src':
            denominator += torch.sum(torch.exp(sim(source[i], source) / tao))
            denominator += torch.sum(
                torch.exp(sim(source[i], pos_sample) / tao))
        else:
            denominator += torch.sum(
                torch.exp(sim(pos_sample[i], pos_sample) / tao))
            denominator += torch.sum(
                torch.exp(sim(pos_sample[i], source) / tao))
        denominator -= math.exp(1 / tao)
        numerator = torch.exp(sim(pos_sample[i], source[i]) / tao)
        return -torch.log(numerator / denominator).item()

    L = 0
    for i in range(N):
        L += _l(i, 'src') + _l(i, 'pos')
    e = datetime.now()
    # print((e-s).microseconds / 10**6)
    return L / (2 * N)


class SeqSampler(Sampler):

    def __init__(self, data_source: TensorDataset, type) -> None:
        super().__init__(data_source)
        self.data = data_source
        self.type = type

    def __iter__(self):
        '''tensors:[features,label]
            features: seq,seq_len,feature_num
            label: seq, feas [rul,len,num]
        '''
        indices_map = {}
        features = self.data.tensors[0]
        labels = self.data.tensors[1]
        for i in range(features.shape[0]):
            tail_dq = features[i][-1][-1]
            origin_dq = features[i][-1][0]
            # rul = labels[i][0]  # tail rul
            # tot_seq_len = labels[i][1]
            # pos = int((tot_seq_len - rul).item())
            pos = '%.3f' % (tail_dq / origin_dq)
            if pos in indices_map.keys():
                indices_map[pos].append(i)
            else:
                indices_map[pos] = [i]
        indices = []
        keys = list(indices_map.keys())
        if self.type == 'train':
            nei_keys = [keys[i + 1] for i in range(len(keys) - 1)]
            nei_keys.append(keys[-2])
            assert len(keys) == len(nei_keys)
            for i in range(len(keys)):
                indices += indices_map[keys[i]]
                indices += indices_map[nei_keys[i]]
        else:
            for i in range(len(keys)):
                indices += indices_map[keys[i]]
        return iter(indices)

    def __len__(self):
        if self.type == 'train':
            return self.data.tensors[0].shape[0] * 2
        else:
            return self.data.tensors[0].shape[0]


def scale_full_seqs_v2(retrieval_set, scale_ratios, rul_factor):
    new_retrieval_set = []
    # feature_num = retrieval_set[0][0].shape[1]

    for k in retrieval_set.keys():
        all_scale_seqs, all_scale_ruls = data_aug(feas=retrieval_set[k][0],
                                                  ruls=retrieval_set[k][1],
                                                  scale_ratios=scale_ratios,
                                                  rul_factor=rul_factor)
        print(len(all_scale_seqs))
    assert 0 == 1
    for seqidx in range(len(retrieval_set)):
        all_scale_seqs, all_scale_ruls = data_aug(
            feas=retrieval_set[seqidx][0],
            ruls=retrieval_set[seqidx][1],
            scale_ratios=scale_ratios,
            rul_factor=rul_factor)
        for idx in range(len(all_scale_seqs)):
            new_retrieval_set.append([
                all_scale_seqs[idx], all_scale_ruls[idx],
                retrieval_set[seqidx][2], retrieval_set[seqidx][3]
            ])
    return new_retrieval_set


class Trainer():

    def __init__(self,
                 lr,
                 gama,
                 n_epochs,
                 device,
                 retrieval_set,
                 scale_ratios=None,
                 parts_num_from_each_len=120,
                 default_target_seq_len=100,
                 train_retrieval_size=10,
                 except_ratios=None,
                 retrieval_fragments_num=10,
                 data_aug_scale_ratios=None,
                 batch_size=32,
                 rulfactor=3000):
        """
        Args:
            lr (float): Learning rate
            n_epochs (int): The number of training epoch
            device: 'cuda' or 'cpu'
            patience (int): How long to wait after last time validation loss improved.
            lamda (float): The weight of RUL loss
            alpha (List: [float]): The weights of Capacity loss
            model_name (str): The model save path
            retrieval_set: a dictionary. Key is the tail point of each slice, and the value is [feature,ruls,seq_len,battery_id]
        """
        self.lr = lr
        self.n_epochs = n_epochs
        self.device = device
        self.rul_factor = rulfactor
        self.batch_size = batch_size
        # self.patience = patience
        # self.model_name = model_name
        # self.lamda = lamda
        # self.alpha = alpha
        # self.retreival_set = retreival_set
        self.gama = gama
        self.scale_ratios = [
            1
        ] if scale_ratios is None else scale_ratios  # 1/3., 1/2.5, 1/2., 1/1.5, 1, 1.5, 2, 2.5, 3
        self.parts_num_from_each_len = parts_num_from_each_len
        # self.retrieval_loader, self.retrieval_tensor = self.preprocess_retreival_set(retrieval_batch_size=retrieval_batch_size)
        # self.sample_ratio_pairs = [[1, 1]]#[[1, 1], [1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2], [3, 4], [4, 3]]
        self.end_cap = 880 / 1190
        self.retrieval_fragments_num = retrieval_fragments_num
        self.tolerable_target_seq_lens = [default_target_seq_len]

        # self.retrieval_set = self.scale_full_seqs_v2(retrieval_set,
        #                                              data_aug_scale_ratios,
        #                                              rulfactor)
        self.retrieval_set = dict(
            sorted(retrieval_set.items(), key=lambda x: float(x[0])))

        # self.retrieval_feas, self.retreival_ruls = self.get_retrieval_parts(
        #     selected_full_seqs=self.retrieval_set, target_part_len=default_targemt_seq_len)

        # self.retrieval_feas, self.retrieval_ruls = self.get_retrieval_fragments(
        #     target_part_len=default_target_seq_len)
        self.train_retrieval_size = train_retrieval_size

        self.except_ratios = except_ratios
        """retrieval_feas: [len(zoom_retrieval_seq_ratios) × retrieval_battery_num × seq_num × seq_len × feature_num]"""

    def scale_full_seqs(self, retrieval_set, scale_ratios, rul_factor):
        new_retrieval_set = []
        # feature_num = retrieval_set[0][0].shape[1]
        for seqidx in range(len(retrieval_set)):
            all_scale_seqs, all_scale_ruls = data_aug(
                feas=retrieval_set[seqidx][0],
                ruls=retrieval_set[seqidx][1],
                scale_ratios=scale_ratios,
                rul_factor=rul_factor)
            for idx in range(len(all_scale_seqs)):
                new_retrieval_set.append([
                    all_scale_seqs[idx], all_scale_ruls[idx],
                    retrieval_set[seqidx][2], retrieval_set[seqidx][3]
                ])
        return new_retrieval_set

    def generate_encoded_database(self,
                                  encoder,
                                  stride=2,
                                  end_cyc=500,
                                  batchsize=200):
        '''
        used when testing, the encoder has been trained well, so it can encode all the series in retrieval set and wait
        the target sequence to appear
        '''
        encoded_feas, ruls = [], []
        for zoom_ratio_idx in range(len(self.retrieval_feas)):
            tmp_feas_pool, tmp_rul_pool, tmp_encoded_feas_pool = [], [], []
            print('ratio', zoom_ratio_idx)

            for retrieval_battery_idx in range(len(self.retrieval_feas[0])):
                if self.retrieval_feas[zoom_ratio_idx][
                        retrieval_battery_idx].shape[0] > 0:
                    sampled_feas = self.retrieval_feas[zoom_ratio_idx][
                        retrieval_battery_idx][(stride - 1)::stride, :, :]
                    tmp_feas_pool.append(sampled_feas[:end_cyc])
                    sampled_ruls = self.retrieval_ruls[zoom_ratio_idx][
                        retrieval_battery_idx][(stride - 1)::stride]
                    tmp_rul_pool.append(sampled_ruls[:end_cyc])

            # batchwise inference because of the limited gpu memory
            tmp_feas_pool = torch.Tensor(
                np.vstack(tmp_feas_pool))  # .to(self.device)
            seqnum = 0
            while seqnum < tmp_feas_pool.size(0):
                if seqnum + batchsize < tmp_feas_pool.size(0):
                    tmp_feas = tmp_feas_pool[seqnum:seqnum +
                                             batchsize, :, :].to(self.device)
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

    def generate_encoded_databasev2(self,
                                    encoder,
                                    stride=2,
                                    end_cyc=500,
                                    batchsize=400):
        '''
        used when testing, the encoder has been trained well, so it can encode all the series in retrieval set and wait
        the target sequence to appear
        '''
        encoded_feas, ruls = [], []
        for batteryidx in range(len(self.retrieval_feas)):
            tmp_feas_pool, tmp_rul_pool, tmp_encoded_feas_pool = [], [], []
            print('original battery idx', batteryidx)

            # for retrieval_battery_idx in range(len(self.retrieval_feas)):
            #     if self.retrieval_feas[zoom_ratio_idx][retrieval_battery_idx].shape[0] > 0:
            #         sampled_feas = self.retrieval_feas[zoom_ratio_idx][retrieval_battery_idx][(stride-1)::stride, :, :]
            #         tmp_feas_pool.append(sampled_feas[:end_cyc])
            #         sampled_ruls = self.retrieval_ruls[zoom_ratio_idx][retrieval_battery_idx][(stride - 1)::stride]
            #         tmp_rul_pool.append(sampled_ruls[:end_cyc])

            # batchwise inference because of the limited gpu memory
            tmp_feas_pool = torch.Tensor(
                self.retrieval_feas[batteryidx])  # .to(self.device)
            seqnum = 0
            while seqnum < tmp_feas_pool.size(0):
                if seqnum + batchsize < tmp_feas_pool.size(0):
                    tmp_feas = tmp_feas_pool[seqnum:seqnum +
                                             batchsize, :, :].to(self.device)
                else:
                    tmp_feas = tmp_feas_pool[seqnum:, :, :].to(self.device)
                encoded_tmp_feas = encoder(tmp_feas)
                seqnum += batchsize
                tmp_encoded_feas_pool.append(encoded_tmp_feas)
                # del tmp_feas_pool
            # tmp_rul_pool = np.hstack(tmp_rul_pool)
            ruls.append(torch.Tensor(self.retrieval_ruls[batteryidx]))
            encoded_feas.append(torch.vstack(tmp_encoded_feas_pool))
        # ruls = torch.Tensor(np.hstack(ruls)).to(self.device)
        # encoded_feas = torch.vstack(encoded_feas).to(self.device)
        # ruls = torch.reshape(ruls, (-1, 1))
        return encoded_feas, ruls

    def generate_encoded_databasev3(self,
                                    encoder,
                                    stride=2,
                                    end_cyc=500,
                                    batchsize=400):
        '''
        used when testing, the encoder has been trained well, so it can encode all the series in retrieval set and wait
        the target sequence to appear
        '''
        new_retrieval_set = {}
        for k in self.retrieval_set.keys():
            tmp_feas = self.retrieval_set[k][0]
            encoded_tmp_feas = encoder(torch.Tensor(tmp_feas).cuda())
            rul = self.retrieval_set[k][1]
            seq_len = self.retrieval_set[k][2]
            battery_id = self.retrieval_set[k][3]
            new_retrieval_set[k] = [encoded_tmp_feas, rul, seq_len, battery_id]
        return new_retrieval_set

    def select_source_seqs(self, battery_seq, batteryidx):
        seqs, batteryids = [], []
        for k, v in self.retreival_set.items():
            if k != batteryidx:
                seqs.append(v)
                batteryids.append(k)
        battery_seq_len = battery_seq.size(1)
        feas, ruls = self.get_retrieval_parts(seqs, battery_seq_len)
        return feas, ruls

    def train(self, train_loader, valid_loader, encoder, relationmodel, wandb,
              save_path):
        # model = model.to(self.device)
        device = self.device
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=self.lr)
        encoder_lr_scheduler = StepLR(encoder_optimizer,
                                      step_size=1,
                                      gamma=self.gama)
        relationmodel_optimizer = optim.Adam(
            relationmodel.parameters(),
            lr=self.lr,
        )
        relation_lr_scheduler = StepLR(relationmodel_optimizer,
                                       step_size=100,
                                       gamma=self.gama)

        loss_fn = nn.MSELoss().to(self.device)
        # loss_fn = torch.nn.L1Loss().to(self.device)
        # loss_fn = torch.nn.L1Loss()

        # Training
        train_loss = []
        valid_loss = []

        for epoch in range(self.n_epochs):

            encoder_lr_scheduler.step(epoch)
            relation_lr_scheduler.step(epoch)
            print('training epoch:', epoch, 'learning rate:',
                  encoder_optimizer.param_groups[0]['lr'])
            encoder.train()
            relationmodel.train()
            y_true, y_pred = [], []
            train_losses = []
            for step, (x, y) in enumerate(train_loader):
                x = x.to(device)
                x_source = x[:self.batch_size]
                x_nei = x[self.batch_size:]
                assert x_source.shape == x_nei.shape
                encoded_target = encoder(x_source)
                encoded_neighbor = encoder(x_nei)

                loss = contrastive_loss(encoded_target, encoded_neighbor,
                                        0.5) * 0.1

                # print(x.shape, y.shape)
                # for i in range(x.shape[0]):
                #     print('%.3f' % (x[i][-1][0] / x[i][-1][-1]))
                # tail_point = int(y[0][1] - y[0][0])
                tail_point = '%.3f' % (x[-1][-1][0] / x[-1][-1][-1])

                if tail_point not in self.retrieval_set.keys():
                    print(f"No key {tail_point}")
                    continue
                retrieval_sub_set = self.retrieval_set[tail_point]
                retrieval_set_keys = list(self.retrieval_set.keys())
                key_idx = retrieval_set_keys.index(tail_point)
                offset = 1
                retrieval_count = retrieval_sub_set[0].shape[0]
                if retrieval_count < x_source.shape[0]:
                    while retrieval_count < x_source.shape[0]:
                        if key_idx > 0:
                            before_nei = self.retrieval_set[retrieval_set_keys[
                                key_idx - offset]]
                            for i in range(len(retrieval_sub_set)):
                                retrieval_sub_set[i] = np.vstack(
                                    (retrieval_sub_set[i], before_nei[i]))
                            retrieval_count += before_nei[0].shape[0]
                        if retrieval_count >= x_source.shape[0]:
                            break
                        if key_idx < len(retrieval_set_keys) - 1:
                            next_nei = self.retrieval_set[retrieval_set_keys[
                                key_idx + offset]]
                            for i in range(len(retrieval_sub_set)):
                                retrieval_sub_set[i] = np.vstack(
                                    (retrieval_sub_set[i], next_nei[i]))
                            retrieval_count += next_nei[0].shape[0]
                        offset += 1

                seqs = retrieval_sub_set[0]
                ruls = retrieval_sub_set[1]

                for i in range(x_source.shape[0]):

                    # tensor_target = x[batch_battery_idx].unsqueeze(dim=0)#.to(device)
                    # encoded_target = encoder(tensor_target)
                    tensor_source = torch.Tensor(seqs).to(device)
                    print("seq", tensor_source.shape)
                    encoded_source = encoder(tensor_source)

                    if encoded_source.size() != encoded_target[i].size():
                        repeated_encoded_target = encoded_target[i].repeat(
                            encoded_source.size(0), 1)
                    else:
                        repeated_encoded_target = encoded_target[i]
                    relation_scores = relationmodel(encoded_source,
                                                    repeated_encoded_target)
                    # all_scores.append(relation_scores)
                    # all_ruls.append(ruls)

                    # all_scores = torch.hstack(all_scores)
                    all_scores = F.softmax(relation_scores, dim=0)
                    all_scores = all_scores.unsqueeze(dim=0)

                    all_ruls = torch.Tensor(ruls).cuda()
                    all_ruls = all_ruls.reshape(-1, 1)

                    all_ruls /= self.rul_factor

                    # all_scores = all_scores / torch.sum(all_scores)
                    # import pdb;pdb.set_trace()
                    predicted_rul = torch.mm(all_scores, all_ruls)
                    # c_l = contrastive_loss(encoded_source, encoded_nei, 0.5)
                    # print("contrastive loss: ", c_l)
                    # loss += c_l
                    loss += loss_fn(predicted_rul,
                                    y[i][0].cuda() / self.rul_factor)

                # print("tot_loss", loss)
                loss /= y.size(0)
                encoder_optimizer.zero_grad()
                relationmodel_optimizer.zero_grad()
                loss.backward()
                encoder_optimizer.step()
                relationmodel_optimizer.step()
                train_loss.append(loss.cpu().detach().numpy())

                if step % 40 == 0:
                    print('step:', step, 'train loss:', train_loss[-1],
                          np.average(train_loss))
                    wandb.log({
                        'loss:': np.average(train_loss),
                        'epoch': epoch
                    })

            print('started to evaluate')
            if True:
                encoder.eval()
                relationmodel.eval()
                y_true, y_pred = [], []
                with torch.no_grad():

                    encoded_retrieval_set = self.generate_encoded_databasev3(
                        encoder)
                    # encoded_source, ruls = self.generate_encoded_databasev3(
                    #     encoder)

                    for step, (x, y) in enumerate(valid_loader):
                        assert y.size(0) == 1
                        x = x.to(device)
                        all_scores, all_ruls = [], []
                        encoded_target = encoder(x)

                        for k, v in encoded_retrieval_set.items():
                            encoded_source = v[0]
                            encoded_ruls = torch.Tensor(v[1])
                            if encoded_target.size() != encoded_source.size():
                                expanded_encoded_target = encoded_target.repeat(
                                    encoded_source.size(0), 1)
                            else:
                                expanded_encoded_target = encoded_target
                            relation_scores = relationmodel(
                                encoded_source, expanded_encoded_target)
                            all_scores.append(relation_scores)
                            all_ruls.append(encoded_ruls)
                            del expanded_encoded_target
                            encoded_source.cpu()

                        all_scores = torch.hstack(all_scores)
                        all_ruls = torch.hstack(all_ruls).to(device)

                        maxscores, maxidx = torch.topk(all_scores,
                                                       1000)  # 1000
                        selected_ruls = all_ruls[maxidx]
                        # maxscores = all_scores
                        # selected_ruls = all_ruls

                        maxscores = F.softmax(maxscores, dim=0)
                        # maxscores = maxscores / torch.sum(maxscores)
                        maxscores = maxscores.unsqueeze(dim=0)

                        selected_ruls = selected_ruls.reshape(-1, 1)
                        predicted_rul = torch.mm(maxscores, selected_ruls)
                        if step % 100 == 0:
                            # if True:
                            print(predicted_rul, y[0][0])
                        y_true.append(y[0][0])
                        y_pred.append(predicted_rul[0][0].item())
                        # import pdb;pdb.set_trace()

                    error = 0
                    for i in range(len(y_true)):
                        error += abs(y_true[i] - y_pred[i]) / y_true[i]
                    print('error:', error / len(y_true))

                    y_true = torch.Tensor(y_true)
                    y_pred = torch.Tensor(y_pred)
                    # import matplotlib.pyplot as plt
                    # plt.plot([i for i in range(len(y_true))], y_true)
                    # plt.plot([i for i in range(len(y_true))], y_pred)
                    # plt.show()
                    epoch_loss = torch.nn.L1Loss()(y_true, y_pred)
                    print(epoch_loss)
                    valid_loss.append(epoch_loss)

                    # if self.n_epochs > 10:
                    if epoch % 1 == 0:
                        print('Epoch number : ', epoch)
                        print(f'-- "train" loss {train_loss[-1]:.4}',
                              f'-- "valid" loss {epoch_loss:.4}')
                        # torch.save(relationmodel.state_dict(), 'VITrelationmodel.pth')
                        name = save_path + '/LSTM_relu_b_32_' + str(
                            int(epoch_loss.item())) + '.pth'
                        torch.save(encoder.state_dict(), name)
                    else:
                        print(f'-- "train" loss {train_loss[-1]:.4}',
                              f'-- "valid" loss {epoch_loss:.4}')

    def test(self, train_loader, valid_loader, encoder, relationmodel,
             save_path):
        device = self.device
        valid_loss = []

        encoder.eval()
        relationmodel.eval()
        y_true, y_pred = [], []
        with torch.no_grad():

            encoded_source, ruls = self.generate_encoded_databasev2(encoder)

            for step, (x, y) in enumerate(valid_loader):
                assert y.size(0) == 1
                x = x.to(device)
                all_scores, all_ruls = [], []
                encoded_target = encoder(x)
                for original_battery_idx in range(len(encoded_source)):
                    if encoded_target.size(
                    ) != encoded_source[original_battery_idx].size():
                        expanded_encoded_target = encoded_target.repeat(
                            encoded_source[original_battery_idx].size(0), 1)
                    relation_scores = relationmodel(
                        encoded_source[original_battery_idx],
                        expanded_encoded_target)
                    all_scores.append(relation_scores)
                    all_ruls.append(ruls[original_battery_idx])
                    del expanded_encoded_target
                    encoded_source[original_battery_idx].cpu()

                all_scores = torch.hstack(all_scores)
                all_ruls = torch.hstack(all_ruls).to(device)

                maxscores, maxidx = torch.topk(all_scores, 1000)  # 1000
                selected_ruls = all_ruls[maxidx]

                maxscores = F.softmax(maxscores, dim=0)
                # maxscores = maxscores / torch.sum(maxscores)
                maxscores = maxscores.unsqueeze(dim=0)

                selected_ruls = selected_ruls.reshape(-1, 1)
                predicted_rul = torch.mm(maxscores, selected_ruls)
                if step % 100 == 0:
                    print(predicted_rul, y[0][0])
                y_true.append(y[0][0] * 3000)
                y_pred.append(predicted_rul[0][0].item() * 3000)
                # import pdb;pdb.set_trace()

            error = 0
            for i in range(len(y_true)):
                error += abs(y_true[i] - y_pred[i]) / y_true[i]
            print('error:', error / len(y_true))
            true_predict_list = []
            for i in range(len(y_true)):
                true_predict_list.append([y_true[i], y_pred[i]])
            true_predict_list = sorted(true_predict_list, key=lambda x: x[0])
            y_true, y_pred = [], []
            for i in range(len(true_predict_list)):
                y_true.append(true_predict_list[i][0])
                y_pred.append(true_predict_list[i][1])
            y_true = torch.Tensor(y_true)
            y_pred = torch.Tensor(y_pred)
            import matplotlib.pyplot as plt
            plt.plot([i for i in range(len(y_true))], y_true)
            plt.plot([i for i in range(len(y_true))], y_pred)
            plt.show()
            epoch_loss = torch.nn.L1Loss()(y_true, y_pred)
            print(epoch_loss)
            valid_loss.append(epoch_loss)
