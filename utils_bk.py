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

    def __init__(self, lr, n_epochs, device, patience, lamda, alpha, model_name, retreival_set, retrieval_batch_size=16):
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

        self.retrieval_loader, self.retrieval_tensor = self.preprocess_retreival_set(retrieval_batch_size=retrieval_batch_size)
        self.sample_ratio_pairs = [[1, 1]]#[[1, 1], [1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2], [3, 4], [4, 3]]
        self.end_cap = 880/1190

    def preprocess_retreival_set(self, retrieval_batch_size):
        max_len = 0
        for batteryid, batteryfea in self.retreival_set.items():
            print(batteryfea[-1][0])
            # import pdb;pdb.set_trace()
            if batteryfea.shape[0] > max_len:
                max_len = batteryfea.shape[0]
                self.battery_fea_dim = batteryfea.shape[1]
        retrieval_tensor = torch.empty(0, max_len, self.battery_fea_dim)
        retreival_ids = []
        for batteryid, batteryfea in self.retreival_set.items():
            self.retreival_set[batteryid] = torch.Tensor(self.retreival_set[batteryid])
            if batteryfea.shape[0] < max_len:
                padding_tensor = torch.zeros(max_len-batteryfea.shape[0], self.battery_fea_dim)
                self.retreival_set[batteryid] = torch.cat((self.retreival_set[batteryid], padding_tensor), dim=0)
            self.retreival_set[batteryid] = self.retreival_set[batteryid].unsqueeze(dim=0)
            retrieval_tensor = torch.cat((retrieval_tensor, self.retreival_set[batteryid]), dim=0)
            retreival_ids.append(batteryid)

        retrieval_dataset = TensorDataset(retrieval_tensor, torch.Tensor(retreival_ids))
        retrieval_loader = DataLoader(retrieval_dataset, batch_size=retrieval_batch_size, shuffle=False)

        print('retrieval set size:', retrieval_tensor.size())

        return retrieval_loader, retrieval_tensor



    def train(self, train_loader, valid_loader, encoder, relationmodel, load_model, reference_set_size=16, loss_type='mse', filter_synthesized=True, retrieval_batch_size=1):
        # model = model.to(self.device)
        device = self.device
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=self.lr)
        encoder_lr_scheduler = StepLR(encoder_optimizer, step_size=100, gamma=0.98)
        relationmodel_optimizer = optim.Adam(relationmodel.parameters(), lr=self.lr, )
        relation_lr_scheduler = StepLR(relationmodel_optimizer, step_size=100, gamma=0.98)

        model_name = self.model_name
        lamda = self.lamda
        alpha = self.alpha

        loss_fn = nn.MSELoss()
        # early_stopping = EarlyStopping(self.patience, verbose=True)
        loss_fn.to(self.device)

        # Training
        train_loss = []
        valid_loss = []
        total_loss = []

        if loss_type == 'mse':
            loss_criterion = torch.nn.MSELoss().to(device)
        # else:
        #     loss_criterion = torch.nn.
        self.retrieval_tensor = self.retrieval_tensor.to(device)
        for epoch in range(self.n_epochs):
            print('training epoch:', epoch)
            encoder_lr_scheduler.step(epoch)
            relation_lr_scheduler.step(epoch)
            encoder.train()
            relationmodel.train()
            # y_true, y_pred = [], []
            # train_losses = []
            for step, (x, y) in enumerate(train_loader):
                x = x.to(device)
                # self.retreival_tensor = self.retreival_tensor.to(device)
                # encoder.eval()
                encoded_retrieval_seqs, retrieval_idx = [], 0
                # with torch.no_grad():
                    # while retrieval_idx < self.retreival_tensor.size(0):
                    #     retrieval_battery_feas = self.retreival_tensor[retrieval_idx:retrieval_idx+retrieval_batch_size].to(device)
                    #     # retrieval_battery_feas = retrieval_battery_feas.unsqueeze(dim=0)
                    #     # import pdb;pdb.set_trace()
                    #       # this is important for preventing GPU memory explosion
                    #     encoded_retrieval_fea = encoder(retrieval_battery_feas)
                    #     # import pdb;pdb.set_trace()
                    #     encoded_retrieval_seqs.append(encoded_retrieval_fea)
                    #     retrieval_idx += retrieval_batch_size
                    #     # del retrieval_battery_feas
                    # '''dataloader version'''
                    # for retrieval_step, (retrieval_seq, retrieval_batteryidx) in enumerate(self.retrieval_loader):
                    #     retrieval_seq = retrieval_seq.to(device)
                    #     encoded_retrieval_fea = encoder(retrieval_seq)
                    #     encoded_retrieval_seqs.append(encoded_retrieval_fea)
                # encoded_retrieval_seqs = torch.vstack(encoded_retrieval_seqs).to(device)
                # encoder.train()
                # import pdb;pdb.set_trace()
                relation_features, reference_ids = [], []  # all the compared batteries in one batch
                batteryids = []
                compared_candidates = list(self.retreival_set.keys())
                # x_fea = encoder(x)
                loss = 0
                for batch_battery_idx in range(y.size(0)):
                    batteryidx = int(y[batch_battery_idx][1].item())
                    real_x = cutoff_zeros(x[batch_battery_idx])
                    # batteryidx is the battery id in the whole dataset
                    candidate_choices = copy.deepcopy(compared_candidates)
                    candidate_choices.remove(batteryidx)
                    # batteryids.append(batteryidx)
                    choices = np.random.choice(candidate_choices, reference_set_size, replace=False)
                    reference_ids.append(choices)

                    '''
                    this part makes data augmentation manually
                    all_source_feas: sample_ratio_pairs * choices * seq_len * feature_num
                    all_transformed_target_feas: sample_ratio_pairs * 1 * seq_len * feature_num
                    all_transformed_sohs: sample_ratio_pairs * choices * seq_len * feature_num
                    '''
                    all_transformed_source_feas, all_transformed_target_feas, all_transformed_sohs = [], [], []

                    for sampleratioidx in range(len(self.sample_ratio_pairs)):
                        source_feas, target_feas, transformed_sohs = [], [], []

                        for choiceids in range(len(choices)):
                            target_seq, source_seq, transformed_soh = compress_seq(real_x, self.retrieval_tensor[choiceids], self.sample_ratio_pairs[sampleratioidx], device)
                            source_feas.append(source_seq.unsqueeze(dim=0))
                            transformed_sohs.append(transformed_soh.unsqueeze(dim=0))

                            if choiceids == (len(choices) - 1):
                                target_feas.append(target_seq.unsqueeze(dim=0))
                        all_transformed_source_feas.append(torch.vstack(source_feas))
                        all_transformed_target_feas.append(torch.vstack(target_feas))
                        # import pdb;pdb.set_trace()
                        all_transformed_sohs.append(torch.vstack(transformed_sohs))
                    all_transformed_sohs = torch.vstack(all_transformed_sohs)
                    all_scores = []
                    for sampleratioidx in range(len(self.sample_ratio_pairs)):
                        source_fea = all_transformed_source_feas[sampleratioidx].to(device)
                        encoded_source = encoder(source_fea)
                        # import pdb;pdb.set_trace()
                        target_fea = all_transformed_target_feas[sampleratioidx].to(device)
                        encoded_target = encoder(target_fea)
                        encoded_target = encoded_target.repeat(encoded_source.size(0), 1)
                        # import pdb;pdb.set_trace()
                        relation_scores = relationmodel(encoded_source, encoded_target)
                        # relation_feature = torch.cat((encoded_source, encoded_target), dim=1).to(device)
                        # relation_scores = relationmodel(relation_feature)
                        all_scores.append(relation_scores)
                    # import pdb;pdb.set_trace()

                    all_scores = torch.hstack(all_scores)
                    scores = F.softmax(all_scores, dim=0)
                    scores = scores.unsqueeze(dim=0)
                    # import pdb; pdb.set_trace()
                    synthesized_seq = torch.mm(scores, all_transformed_sohs)
                    # import pdb;pdb.set_trace()
                    loss += loss_criterion(synthesized_seq, self.retrieval_tensor[batteryidx][:, 0].unsqueeze(dim=0))
                    # x_relation_feas = torch.empty(0, x_fea.size(1)*2).to(device)
                    # for choice in choices:
                    #     relation_feature = torch.cat((x_fea[batch_battery_idx], encoded_retrieval_seqs[choice]), dim=0)
                    #     relation_feature = relation_feature.unsqueeze(dim=0)
                    #     x_relation_feas = torch.cat((x_relation_feas, relation_feature), dim=0)
                    # relation_features.append(x_relation_feas)


                # loss = 0
                #
                # for relation_fea_idx in range(x.size(0)):
                #     relation_scores = relationmodel(relation_features[relation_fea_idx])
                #     # import pdb;pdb.set_trace()
                #     relation_scores = F.softmax(relation_scores, dim=0)
                #     # import pdb;pdb.set_trace()
                #     candidate_choices = torch.tensor(reference_ids[relation_fea_idx]).to(device)#.unsqueeze(dim=0)
                #     candidates_values = torch.index_select(self.retrieval_tensor, 0, candidate_choices)
                #     # candidates_values = candidates_values.T
                #     relation_scores = relation_scores.T
                #     # import pdb;pdb.set_trace()
                #     # only the discharge capacity part is used to compute the loss
                #     synthesized_seq = torch.mm(relation_scores, candidates_values[:, :, 0])
                #     # put the cycles that are after the end capacity to zero, such is the type of the retrieval tensor.
                #     if filter_synthesized:
                #         synthesized_seq = torch.nn.Threshold(self.end_cap, 0)(synthesized_seq)
                #         # synthesized_seq -= self.end_cap * torch.ones_like(synthesized_seq)
                #         # synthesized_seq = torch.nn.ReLU()(synthesized_seq)
                #         # synthesized_seq += self.end_cap * torch.ones_like(synthesized_seq)
                #     # only the discharge capacity part is used to compute the loss
                #     # import pdb;pdb.set_trace()
                #     loss += loss_criterion(synthesized_seq, self.retrieval_tensor[batteryids[relation_fea_idx]][:, 0])

                loss /= y.size(0)
                loss *= 100
                encoder_optimizer.zero_grad()
                relationmodel_optimizer.zero_grad()
                loss.backward()
                # y_, soh_ = model(x)
                #
                # loss = lamda * loss_fn(y_.squeeze(), y[:, 0])
                #
                # for i in range(y.shape[1] - 1):
                #     loss += loss_fn(soh_[:, i], y[:, i + 1]) * alpha[i]
                # torch.nn.utils.clip_grad_norm(encoder.parameters(), 0.5)
                # torch.nn.utils.clip_grad_norm(relationmodel.parameters(), 0.5)
                # loss.backward()
                # optimizer.step()
                encoder_optimizer.step()
                relationmodel_optimizer.step()

                train_loss.append(loss.cpu().detach().numpy())

                if step % 1 == 0:
                    print('step:', step, 'train loss:', train_loss[-1], np.average(train_loss))

                # predicted_rul = self.retreival_tensor.size(1)
                # for batteryidx in range(x.size(0)):
                #     for cycleidx in range(synthesized_seq[batteryidx].size(0)):
                #         if synthesized_seq[0][cycleidx][0] < self.end_cap:
                #             predicted_rul = cycleidx - 1
                #             break
                # y_true.append(int(y[0][0]))
                # y_pred.append(predicted_rul)

                # y_pred.append(y_)
                # y_true.append(y[:, 0])

            # y_true = torch.cat(y_true, axis=0)
            # y_pred = torch.cat(y_pred, axis=0)

            # epoch_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
            # train_loss.append(epoch_loss)
            #
            # losses = np.mean(losses)
            # total_loss.append(losses)

            # validate
            # model.eval()
            print('started to evaluate')
            encoder.eval()
            relationmodel.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for step, (x, y) in enumerate(valid_loader):
                    # print(x[:, :, 0])
                    # if step % 50 != 0:
                    #     continue
                    # print('evaluate step:', step)
                    assert y.size(0) == 1
                    x = x.to(device)
                    y = y.to(device)
                    # y_, soh_ = model(x)
                    all_scores, stretched_sohs = [], []
                    real_x = cutoff_zeros(x[0])
                    for sampleratioidx in range(len(self.sample_ratio_pairs)):
                        transformed_target, transformed_source = [], []
                        for retrievalids in range(self.retrieval_tensor.size(0)):
                            source_seq = self.retrieval_tensor[retrievalids].to(device)
                            target_seq, source_seq, transformed_soh = compress_seq(real_x, source_seq, self.sample_ratio_pairs[sampleratioidx], device)
                            transformed_source.append(source_seq.unsqueeze(dim=0))
                            if retrievalids == 0:
                                transformed_target.append(target_seq.unsqueeze(dim=0))
                            stretched_sohs.append(transformed_soh.unsqueeze(dim=0))
                        transformed_target = transformed_target[0].to(device)
                        transformed_source = torch.vstack(transformed_source).to(device)
                        encoded_source = encoder(transformed_source)
                        encoded_target = encoder(transformed_target)
                        encoded_target = encoded_target.repeat(encoded_source.size(0), 1)
                        # relation_scores = torch.cat((encoded_source, encoded_target), dim=1).to(device)
                        relation_scores = relationmodel(encoded_source, encoded_target)
                        # import pdb;pdb.set_trace()
                        all_scores.append(relation_scores)
                    stretched_sohs = torch.vstack(stretched_sohs)
                    all_scores = torch.hstack(all_scores)

                    all_scores = F.softmax(all_scores, dim=0)
                    all_scores = all_scores.unsqueeze(dim=0)
                    # import pdb;
                    # pdb.set_trace()
                    synthesized_seq = torch.mm(all_scores, stretched_sohs)
                    # x_feature = encoder(x)
                    # # import pdb;pdb.set_trace()
                    # x_features = x_feature.repeat(self.retrieval_tensor.size(0), 1, 1)
                    # x_features = x_features.squeeze(dim=1)
                    # # encoder.eval()
                    # encoded_retrieval_seqs, retrieval_idx = [], 0
                    # # while retrieval_idx < self.retreival_tensor.size(0):
                    # #     retrieval_battery_feas = copy.deepcopy(
                    # #         self.retreival_tensor[retrieval_idx:retrieval_idx + retrieval_batch_size]).to(device)
                    # #     # retrieval_battery_feas = retrieval_battery_feas.unsqueeze(dim=0)
                    # #     encoded_retrieval_fea = encoder(retrieval_battery_feas)
                    # #     encoded_retrieval_seqs.append(encoded_retrieval_fea)
                    # #     retrieval_idx += retrieval_batch_size
                    # #     del retrieval_battery_feas
                    # '''dataloader version'''
                    # for retrieval_step, (retrieval_seq, retrieval_batteryidx) in enumerate(self.retrieval_loader):
                    #     retrieval_seq = retrieval_seq.to(device)
                    #     encoded_retrieval_fea = encoder(retrieval_seq)
                    #     encoded_retrieval_seqs.append(encoded_retrieval_fea)
                    # encoded_retrieval_seqs = torch.vstack(encoded_retrieval_seqs)
                    # # import pdb;pdb.set_trace()
                    # relation_feature = torch.cat((x_features, encoded_retrieval_seqs), dim=1)
                    # scores = relationmodel(relation_feature)
                    # scores = F.softmax(scores, dim=0)

                    # scores = scores.T
                    # synthesized_seq = torch.mm(scores, self.retrieval_tensor[:, :, 0])
                    rul_end, rul_start = self.retrieval_tensor.size(1), real_x.size(0) - 1
                    # start_flag = False
                    # import pdb;pdb.set_trace()
                    for cycleidx in range(synthesized_seq.size(1)):
                        # import pdb;pdb.set_trace()
                        # if synthesized_seq[0][cycleidx].item() < x[0][-1][0].item() and not start_flag:
                        #     rul_start = cycleidx
                        #     start_flag = True
                        if synthesized_seq[0][cycleidx].item() < self.end_cap:
                            rul_end = cycleidx - 1
                            break
                    # import pdb;pdb.set_trace()
                    y_true.append(int(y[0][0]))
                    y_pred.append(int(rul_end - rul_start))
                    print(step, y_true[-1], y_pred[-1])

                y_true = torch.Tensor(y_true)
                y_pred = torch.Tensor(y_pred)
                # import pdb;pdb.set_trace()
                epoch_loss = torch.nn.MSELoss()(y_true, y_pred)
                valid_loss.append(epoch_loss)

                # if self.n_epochs > 10:
                if (epoch % 1 == 0 and epoch != 0):
                    print('Epoch number : ', epoch)
                    print(f'-- "train" loss {train_loss[-1]:.4}', f'-- "valid" loss {epoch_loss:.4}')
                    torch.save(relationmodel.state_dict(), 'relationmodel.pth')
                    torch.save(encoder.state_dict(), 'encoder.pth')
                else:
                    print(f'-- "train" loss {train_loss[-1]:.4}', f'-- "valid" loss {epoch_loss:.4}')

            # early_stopping(epoch_loss, model, f'{model_name}_best.pt')
            # if early_stopping.early_stop:
            #     break

        # if load_model:
        #     model.load_state_dict(torch.load(f'{model_name}_best.pt'))
        # else:
        #     torch.save(model.state_dict(), f'{model_name}_end.pt')

        # return model, train_loss, valid_loss, total_loss

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


