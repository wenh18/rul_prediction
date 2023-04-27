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


# save dict
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


# load dict
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def interp(v, q, num):
    f = interpolate.interp1d(v, q, kind='linear')
    v_new = np.linspace(v[0], v[-1], num)
    q_new = f(v_new)
    vq_new = np.concatenate((v_new.reshape(-1, 1), q_new.reshape(-1, 1)), axis=1)
    return q_new


def integral(x, y):
    result = 0
    for i in range(len(x) - 1):
        result += (y[i] + y[i + 1]) / 2 * (x[i + 1] - x[i])
    return result


def preprocess(data, t):
    datamean = np.mean(data)
    datastdvar = math.sqrt(np.var(data))
    data_s, data_k, e_k, p_k = 0, 0, 0, 0
    for i in range(len(data)):
        data_s += (data[i] - datamean) ** 3
        data_k += (data[i] - datamean) ** 4
    data_s = data_s / (len(data) * (datastdvar ** 3))
    data_k = data_k / (len(data) * (datastdvar ** 4))
    data_square = [data[i] ** 2 for i in range(len(data))]
    e_k = integral(t, data_square)
    # import pdb;pdb.set_trace()
    p_k = math.log(1 / (t[-1] - t[0]) * e_k)
    return [datamean, datastdvar, data_s, data_k, e_k, p_k]

def preprocessv3(data, t):
    datamean = np.mean(data)
    datastdvar = math.sqrt(np.var(data))
    return [datamean, datastdvar]


def get_xyv2(name, series_lens, i_low, i_upp, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor, pkl_dir,
             raw_features=True, fill_with_zero=True, seriesnum=None):
    """
    Args:
        n_cyc (int): The previous cycles number for model input
        in_stride (int): The interval in the previous cycles number
        fea_num (int): The number of interpolation
        v_low (float): Voltage minimum for normalization
        v_upp (float): Voltage maximum for normalization
        q_low (float): Capacity minimum for normalization
        q_upp (float): Capacity maximum for normalization
        rul_factor (float): The RUL factor for normalization
        cap_factor (float): The capacity factor for normalization
        seriesnum: The number of series sliced from this degradation curve
    """
    # print("loading", name)
    if not os.path.exists(pkl_dir + name + 'v3.npy'):
        A = load_obj(pkl_dir + name)[name]
        A_rul = A['rul']
        A_dq = A['dq']
        A_df = A['data']
        all_fea = []
        all_idx = list(A_dq.keys())[9:]
        ruls = []
        for cyc in all_idx:
            if cyc % 500 == 0:
                print(cyc)
            feature = [A_dq[cyc] / cap_factor]
            time, v, i, q, dv, di, dq, dtime = [], [], [], [], [], [], [], []
            for timeidx in range(len(A_df[cyc]['Status'])):
                if 'discharge' in A_df[cyc]['Status'][timeidx]:
                    time.append(A_df[cyc]['Time (s)'][timeidx])
                    v.append((A_df[cyc]['Voltage (V)'][timeidx] - v_low) / (v_upp - v_low))
                    i.append((A_df[cyc]['Current (mA)'][timeidx] - i_low) / (i_upp - i_low))
                    q.append((A_df[cyc]['Capacity (mAh)'][timeidx] - q_low) / (q_upp - q_low))
                    if timeidx < len(A_df[all_idx[0]]['Voltage (V)']):
                        # import pdb;pdb.set_trace()
                        dv.append((A_df[cyc]['Voltage (V)'][timeidx] - A_df[all_idx[0]]['Voltage (V)'][timeidx]) / (v_upp - v_low))
                        di.append((A_df[cyc]['Current (mA)'][timeidx] - A_df[all_idx[0]]['Current (mA)'][timeidx]) / (i_upp - i_low))
                        dq.append((A_df[cyc]['Capacity (mAh)'][timeidx] - A_df[all_idx[0]]['Capacity (mAh)'][timeidx]) / (q_upp - q_low))
                        dtime.append(A_df[cyc]['Time (s)'][timeidx])
            feature += preprocessv3(v, time)
            feature += preprocessv3(i, time)
            feature += preprocessv3(q, time)
            feature += preprocessv3(dv, dtime)
            feature += preprocessv3(di, dtime)
            feature += preprocessv3(dq, dtime)
            all_fea.append(feature)
            ruls.append(A_rul[cyc])

        np.save(pkl_dir + name + 'v3.npy', all_fea)
        np.save(pkl_dir + name + '_rulv3.npy', ruls)
    else:
        all_fea = np.load(pkl_dir + name + 'v3.npy', allow_pickle=True)
        A_rul = np.load(pkl_dir + name + '_rulv3.npy', allow_pickle=True)
        # import pdb;pdb.set_trace()
    if raw_features:
        return np.array(all_fea), A_rul
    feature_num = len(all_fea[0])
    all_series, all_ruls = np.empty((0, np.max(series_lens), feature_num)), np.empty((0, 2))
    for series_len in series_lens:
        # series_num = len(all_fea) // series_len
        # series = np.lib.stride_tricks.as_strided(np.array(all_fea), (series_num, series_len, feature_num))
        series = np.lib.stride_tricks.sliding_window_view(all_fea, (series_len, feature_num))
        series = series.squeeze()
        full_series = []
        if series_len < np.max(series_lens) and fill_with_zero:
            zeros = np.zeros((np.max(series_lens) - series_len, feature_num))
            for seriesidx in range(series.shape[0]):
                # import pdb;pdb.set_trace()
                full_series.append(np.concatenate((series[seriesidx], zeros)))
        elif series_len == np.max(series_lens):
            full_series = series
        # ruls = np.array(A_rul[series_len - 1:]) / rul_factor
        # series.tolist()
        full_series = np.array(full_series)

        full_seq_len = len(A_rul)

        if isinstance(A_rul, dict):
            tmp = []
            for k, v in A_rul.items():
                if k >= series_len:
                    tmp.append([v, full_seq_len])
            ruls = tmp
        else:
            ruls = A_rul[series_len - 1:].tolist()
            for i in range(len(ruls)):
                ruls[i] = [ruls[i], full_seq_len]
        # import pdb;pdb.set_trace()
        # print(all_series.shape, all_ruls.shape)
        all_series = np.append(all_series, full_series, axis=0)
        ruls = np.array(ruls).astype(float)
        ruls /= rul_factor
        all_ruls = np.append(all_ruls, ruls, axis=0)
    if seriesnum is not None:
        all_series = all_series[:seriesnum]
        all_ruls = all_ruls[:seriesnum]
    return all_series, all_ruls


def get_retrieval_seq(name, pkl_dir, rul_factor, seriesnum=None):
    '''
    gets degradation curve that starts from the first cycle
    '''
    all_fea = np.load(pkl_dir + name + 'v3.npy', allow_pickle=True)
    A_rul = np.load(pkl_dir + name + '_rulv3.npy', allow_pickle=True).astype(float)
    seq_len = len(A_rul)
    A_rul /= rul_factor
    if seriesnum is not None:
        all_fea = all_fea[:seriesnum]
        A_rul = A_rul[:seriesnum]
    return all_fea, A_rul, seq_len


def get_xy_from_start(name, series_lens, i_low, i_upp, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor, pkl_dir,
             raw_features=True, fill_with_zero=True):
    '''
    gets degradation curve that starts from the first cycle
    '''
    all_fea = np.load(pkl_dir + name + 'v2.npy', allow_pickle=True)
    A_rul = np.load(pkl_dir + name + '_rulv2.npy', allow_pickle=True)
    feature_num = len(all_fea[0])
    all_series, all_ruls = [], []
    for series_len in series_lens:
        # series = np.lib.stride_tricks.sliding_window_view(all_fea, (series_len, feature_num))
        series = all_fea[:series_len, :]
        # series = series.squeeze()
        # full_series = []
        if series_len < np.max(series_lens) and fill_with_zero:
            zeros = np.zeros((np.max(series_lens) - series_len, feature_num))
            # for seriesidx in range(series.shape[0]):
            full_series = np.concatenate((series, zeros))
        elif series_len == np.max(series_lens):
            full_series = series
        # ruls = np.array(A_rul[series_len - 1:]) / rul_factor
        # series.tolist()
        # full_series = np.array([full_series])
        # import pdb;pdb.set_trace()
        rul = A_rul[series_len - 1]
        all_series.append(full_series)
        all_ruls.append(rul)
    all_series = np.array(all_series)
    all_ruls = np.array(all_ruls)
    # battery_ids = [name for i in range]
    return all_series, all_ruls



def save_full_seqs(name, n_cyc, in_stride, fea_num, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor):
    """
    Args:
        n_cyc (int): The previous cycles number for model input
        in_stride (int): The interval in the previous cycles number
        fea_num (int): The number of interpolation
        v_low (float): Voltage minimum for normalization
        v_upp (float): Voltage maximum for normalization
        q_low (float): Capacity minimum for normalization
        q_upp (float): Capacity maximum for normalization
        rul_factor (float): The RUL factor for normalization
        cap_factor (float): The capacity factor for normalization
    """
    print('dealing with {}'.format(name))
    A = load_obj(f'./our_data/{name}')[name]
    A_rul = A['rul']
    A_dq = A['dq']
    A_df = A['data']
    all_idx = list(A_dq.keys())[9:]
    all_fea, rul_lbl, cap_lbl = [], [], []
    for cyc in all_idx:
        tmp = A_df[cyc]
        tmp = tmp.loc[tmp['Status'].apply(lambda x: not 'discharge' in x)]

        left = (tmp['Current (mA)'] < 5000).argmax() + 1
        right = (tmp['Current (mA)'] < 1090).argmax() - 2

        tmp = tmp.iloc[left:right]

        tmp_v = tmp['Voltage (V)'].values
        tmp_q = tmp['Capacity (mAh)'].values
        tmp_t = tmp['Time (s)'].values
        v_fea = interp(tmp_t, tmp_v, fea_num)
        q_fea = interp(tmp_t, tmp_q, fea_num)

        tmp_fea = np.hstack((v_fea.reshape(-1, 1), q_fea.reshape(-1, 1)))

        all_fea.append(np.expand_dims(tmp_fea, axis=0))
        rul_lbl.append(A_rul[cyc])
        cap_lbl.append(A_dq[cyc])
    all_fea = np.vstack(all_fea)
    rul_lbl = np.array(rul_lbl)
    cap_lbl = np.array(cap_lbl)
    # import pdb;pdb.set_trace()
    all_fea_c = all_fea.copy()
    all_fea_c[:, :, 0] = (all_fea_c[:, :, 0] - v_low) / (v_upp - v_low)
    all_fea_c[:, :, 1] = (all_fea_c[:, :, 1] - q_low) / (q_upp - q_low)
    dif_fea = all_fea_c - all_fea_c[0:1, :, :]
    all_fea = np.concatenate((all_fea, dif_fea), axis=2)

    all_fea[:, :, 0] = (all_fea[:, :, 0] - v_low) / (v_upp - v_low)
    all_fea[:, :, 1] = (all_fea[:, :, 1] - q_low) / (q_upp - q_low)

    extracted_fea = []
    for cycidx in range(all_fea.shape[0]):
        tmp_extracted_fea = [cap_lbl[cycidx]]
        for i in range(4):
            tmp_extracted_fea += preprocessv3(all_fea[cycidx, :, i], None)
        extracted_fea.append(tmp_extracted_fea)

    extracted_fea = np.array(extracted_fea)

    extracted_fea[:, 0] = extracted_fea[:, 0] / cap_factor
    # all_lbl = all_lbl[seq_len - 1:]

    np.save('our_data/' + name + 'v4.npy', extracted_fea)
    np.save('our_data/' + name + '_rulv4.npy', rul_lbl)

    # all_fea = np.lib.stride_tricks.sliding_window_view(all_fea, (
    # n_cyc, fea_num, 4))  # build sliding windows (x, window_shape), e.g. [0, 1, 2, 3] -> [0, 1, 2], [1, 2, 3]
    # cap_lbl = np.lib.stride_tricks.sliding_window_view(cap_lbl, (n_cyc,))
    # all_fea = all_fea.squeeze(axis=(1, 2,))
    # rul_lbl = rul_lbl[n_cyc - 1:]
    # all_fea = all_fea[:, (in_stride - 1)::in_stride, :, :]
    # cap_lbl = cap_lbl[:, (in_stride - 1)::in_stride, ]

    # all_fea_new = np.zeros(all_fea.shape)
    # all_fea_new[:, :, :, 0] = (all_fea[:, :, :, 0] - v_low) / (v_upp - v_low)
    # all_fea_new[:, :, :, 1] = (all_fea[:, :, :, 1] - q_low) / (q_upp - q_low)
    # all_fea_new[:, :, :, 2] = all_fea[:, :, :, 2]
    # all_fea_new[:, :, :, 3] = all_fea[:, :, :, 3]
    # print(f'{name} length is {all_fea_new.shape[0]}',
    #       'v_max:', '%.4f' % all_fea_new[:, :, :, 0].max(),
    #       'q_manp.lx:', '%.4f' % all_fea_new[:, :, :, 1].max(),
    #       'dv_max:', '%.4f' % all_fea_new[:, :, :, 2].max(),
    #       'dq_max:', '%.4f' % all_fea_new[:, :, :, 3].max())
    # rul_lbl = rul_lbl / rul_factor
    # cap_lbl = cap_lbl / cap_factor

    # return all_fea_new, np.hstack((rul_lbl.reshape(-1, 1), cap_lbl))


def get_xyv3(name, seq_len, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor,
             raw_features=True, fill_with_zero=True, fea_num=100):
    """
    Args:
        n_cyc (int): The previous cycles number for model input
        in_stride (int): The interval in the previous cycles number
        fea_num (int): The number of interpolation
        v_low (float): Voltage minimum for normalization
        v_upp (float): Voltage maximum for normalization
        q_low (float): Capacity minimum for normalization
        q_upp (float): Capacity maximum for normalization
        rul_factor (float): The RUL factor for normalization
        cap_factor (float): The capacity factor for normalization
    """
    A = load_obj(f'./our_data/{name}')[name]
    A_rul = A['rul']
    A_dq = A['dq']
    A_df = A['data']
    all_idx = list(A_dq.keys())[9:]
    all_fea, rul_lbl, cap_lbl = [], [], []
    for cyc in all_idx:
        tmp = A_df[cyc]
        tmp = tmp.loc[tmp['Status'].apply(lambda x: not 'discharge' in x)]

        left = (tmp['Current (mA)'] < 5000).argmax() + 1
        right = (tmp['Current (mA)'] < 1090).argmax() - 2

        tmp = tmp.iloc[left:right]

        tmp_v = tmp['Voltage (V)'].values
        tmp_q = tmp['Capacity (mAh)'].values
        tmp_t = tmp['Time (s)'].values
        v_fea = interp(tmp_t, tmp_v, fea_num)
        q_fea = interp(tmp_t, tmp_q, fea_num)

        tmp_fea = np.hstack((v_fea.reshape(-1, 1), q_fea.reshape(-1, 1)))

        all_fea.append(np.expand_dims(tmp_fea, axis=0))
        rul_lbl.append(A_rul[cyc])
        cap_lbl.append(A_dq[cyc])
    all_fea = np.vstack(all_fea)
    rul_lbl = np.array(rul_lbl)
    cap_lbl = np.array(cap_lbl)
    # import pdb;pdb.set_trace()
    all_fea_c = all_fea.copy()
    all_fea_c[:, :, 0] = (all_fea_c[:, :, 0] - v_low) / (v_upp - v_low)
    all_fea_c[:, :, 1] = (all_fea_c[:, :, 1] - q_low) / (q_upp - q_low)
    dif_fea = all_fea_c - all_fea_c[0:1, :, :]
    all_fea = np.concatenate((all_fea, dif_fea), axis=2)  # eg. [2048, 100, 4]

    all_fea = np.lib.stride_tricks.sliding_window_view(all_fea, (seq_len, fea_num, 4))  # build sliding windows (x, window_shape), e.g. [0, 1, 2, 3] -> [0, 1, 2], [1, 2, 3]
    cap_lbl = np.lib.stride_tricks.sliding_window_view(cap_lbl, (seq_len,))
    all_fea = all_fea.squeeze(axis=(1, 2,))
    rul_lbl = rul_lbl[seq_len - 1:]
    # all_fea = all_fea[:, (in_stride - 1)::in_stride, :, :]
    # cap_lbl = cap_lbl[:, (in_stride - 1)::in_stride, ]
    #
    # all_fea_new = np.zeros(all_fea.shape)
    # all_fea_new[:, :, :, 0] = (all_fea[:, :, :, 0] - v_low) / (v_upp - v_low)
    # all_fea_new[:, :, :, 1] = (all_fea[:, :, :, 1] - q_low) / (q_upp - q_low)
    # all_fea_new[:, :, :, 2] = all_fea[:, :, :, 2]
    # all_fea_new[:, :, :, 3] = all_fea[:, :, :, 3]
    print(f'{name} length is {all_fea.shape[0]}',
          'v_max:', '%.4f' % all_fea[:, :, :, 0].max(),
          'q_manp.lx:', '%.4f' % all_fea[:, :, :, 1].max(),
          'dv_max:', '%.4f' % all_fea[:, :, :, 2].max(),
          'dq_max:', '%.4f' % all_fea[:, :, :, 3].max())
    rul_lbl = rul_lbl / rul_factor
    cap_lbl = cap_lbl / cap_factor

    return all_fea, np.hstack((rul_lbl.reshape(-1, 1), cap_lbl))


class Seriesset(torch.utils.data.Dataset):
    def __init__(self, features, ruls, batteryids):
        self.features = features
        self.ruls = ruls
        self.batteryids = batteryids

    def __getitem__(self, index):
        return self.features[index], self.ruls[index], self.batteryids[index]

    def __len__(self):
        return self.features.size(0)


if __name__ == '__main__':
    new_valid = ['4-3', '5-7', '3-3', '2-3', '9-3', '10-5', '3-2', '3-7']
    new_train = ['9-1', '2-2', '4-7', '9-7', '1-8', '4-6', '2-7', '8-4', '7-2', '10-3', '2-4', '7-4', '3-4',
                 '5-4', '8-7', '7-7', '4-4', '1-3', '7-1', '5-2', '6-4', '9-8', '9-5', '6-3', '10-8', '1-6', '3-5',
                 '2-6', '3-8', '3-6', '4-8', '7-8', '5-1', '2-8', '8-2', '1-5', '7-3', '10-2', '5-5', '9-2', '5-6',
                 '1-7',
                 '8-3', '4-1', '4-2', '1-4', '6-5', ]
    new_test = ['9-6', '4-5', '1-2', '10-7', '1-1', '6-1', '6-6', '9-4', '10-4', '8-5', '5-3', '10-6',
                '2-5', '6-2', '3-1', '8-8', '8-1', '8-6', '7-6', '6-8', '7-5', '10-1']
