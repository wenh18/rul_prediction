import numpy as np
import scipy.io as sio
import h5py
import matplotlib.pyplot as plt
from scipy import interpolate
from copy import deepcopy
from scipy import stats
from scipy.optimize import leastsq
from scipy.stats import pearsonr
import pickle
from dataloader import preprocessv3


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def change(arr, t, num):
    x_new = np.linspace(t[0], t[-1], num)
    f_linear = interpolate.interp1d(t, arr)
    y_new = f_linear(x_new)
    return y_new


def filter_out_training_extremes(data, ruls, threshold=0.05, rounds=10):
    for _ in range(rounds):
        dataidx = 1
        while dataidx < data.shape[0]:
            if data[dataidx][0] > data[dataidx - 1][0] + threshold:
                data = np.vstack((data[:dataidx], data[dataidx + 1:]))
                # ruls = np.hstack((ruls[:dataidx], ruls[dataidx + 1:]))
            if data[dataidx][0] < data[dataidx - 1][0] - threshold:
                data = np.vstack((data[:dataidx], data[dataidx + 1:]))
                # ruls = np.hstack((ruls[:dataidx], ruls[dataidx ]))
            dataidx += 1
        ruls = ruls[-data.shape[0]:]
    return data, ruls


def interp(x, y, num, ruls, rul_factor):
    ynew = []
    for i in range(y.shape[1]):
        f = interpolate.interp1d(x, y[:, i], kind='linear')
        x_new = np.linspace(x[0], x[-1], num)
        ytmp = f(x_new)
        ynew.append(ytmp)
    ynew = np.vstack(ynew)
    ynew = ynew.T
    newruls = [i for i in range(1, ynew.shape[0] + 1)]
    newruls.reverse()
    newruls = np.array(newruls).astype(float)
    newruls /= rul_factor
    new_right_end_value = ruls[-1] * (num/len(x))
    for i in range(len(newruls)):
        newruls[i] += new_right_end_value
    return ynew, newruls

def data_aug(feas, ruls, scale_ratios, rul_factor):
    augmented_feas, augmented_ruls = [], []
    for scaleratio in scale_ratios:
        if int(scaleratio * feas.shape[0]) <= 100:
            continue
        augmented, rul = interp([i for i in range(feas.shape[0])], feas,
                                int(scaleratio*feas.shape[0]), ruls, rul_factor)
        augmented_feas.append(augmented)
        augmented_ruls.append(rul)
    return augmented_feas, augmented_ruls

def split_seq(fullseq, rul_labels, seqlen, seqnum):
    if isinstance(fullseq, list):
        all_fea, all_lbls = [], []
        for seqidx in range(len(fullseq)):
            tmp_all_fea = np.lib.stride_tricks.sliding_window_view(fullseq[seqidx], (seqlen, 9))

            tmp_all_fea = tmp_all_fea.squeeze()
            tmp_lbls = rul_labels[seqidx][seqlen - 1:]
            tmp_fullseqlen = rul_labels[seqidx][0]
            fullseqlens = np.array([tmp_fullseqlen for _ in range(tmp_all_fea.shape[0])])
            # print(tmp_lbls.shape, fullseqlens.shape, fullseq[seqidx].shape, rul_labels[seqidx].shape)
            lbls = np.vstack((tmp_lbls, fullseqlens)).T
            if seqnum <= tmp_all_fea.shape[0]:
                all_fea.append(tmp_all_fea[:seqnum])
                all_lbls.append(lbls[:seqnum])
            else:
                all_fea.append(tmp_all_fea)
                all_lbls.append(lbls)
        all_fea = np.vstack(all_fea)
        all_lbls = np.vstack(all_lbls)
        return all_fea, all_lbls
    else:
        all_fea = np.lib.stride_tricks.sliding_window_view(fullseq, (seqlen, 9))
        all_fea = all_fea.squeeze()
        # ruls = rul_labels[seqlen-1:]
        fullseqlen = rul_labels[0]
        lbls = rul_labels[seqlen - 1:]
        fullseqlens = np.array([fullseqlen for _ in range(all_fea.shape[0])])
        lbls = np.vstack((lbls, fullseqlens)).T
        if seqnum <= all_fea.shape[0]:
            return all_fea[:seqnum], lbls[:seqnum]
        else:
            return all_fea, lbls


def get_train_test_val(series_len=100, rul_factor=3000, dataset_name='train', seqnum=500, data_aug_scale_ratios=None):

    metadata = np.load('ne_data/meta_data.npy', allow_pickle=True)
    if dataset_name == 'train':
        set = metadata[0]
    elif dataset_name == 'valid':
        set = metadata[1]
    elif dataset_name == 'trainvalid':
        set = metadata[0] + metadata[1]
    else:
        set = metadata[2]

    allseqs, allruls, batteryids = [], [], []
    batteryid = 0
    for batteryname in set:
        seqname = 'ne_data/' + batteryname + 'v3.npy'
        lblname = 'ne_data/' + batteryname + '_rulv3.npy'
        seq = np.load(seqname, allow_pickle=True)
        lbls = np.load(lblname, allow_pickle=True)
        lbls = lbls / rul_factor
        if data_aug_scale_ratios is not None:
            seqs, ruls = data_aug(seq, lbls, data_aug_scale_ratios, rul_factor)
            feas, ruls = split_seq(seqs, ruls, series_len, seqnum)
        else:
            feas, ruls = split_seq(seq, lbls, series_len, seqnum)

        allseqs.append(feas)
        allruls.append(ruls)
        batteryids += [batteryid for _ in range(feas.shape[0])]
        batteryid += 1
    batteryids = np.array(batteryids).reshape((-1, 1))
    allruls = np.vstack(allruls)
    allruls = np.hstack((allruls, batteryids))
    allseqs = np.vstack(allseqs)

    return allseqs, allruls


def get_retrieval_seq(pkl_dir='ne_data/', rul_factor=3000, seriesnum=None, dataset_name='trainvalid'):
    '''
    gets degradation curve that starts from the first cycle
    '''
    retrieval_set = {}
    metadata = np.load('ne_data/meta_data.npy', allow_pickle=True)
    if dataset_name == 'train':
        set = metadata[0]
    elif dataset_name == 'trainvalid':
        set = metadata[0] + metadata[1]
    batteryid = 0
    for name in set:
        all_fea = np.load(pkl_dir + name + 'v3.npy', allow_pickle=True)
        A_rul = np.load(pkl_dir + name + '_rulv3.npy', allow_pickle=True).astype(float)
        seq_len = len(A_rul)
        A_rul /= rul_factor
        if seriesnum is not None and seriesnum < all_fea.shape[0]:
            all_fea = all_fea[:seriesnum]
            A_rul = A_rul[:seriesnum]
        retrieval_set[batteryid] = [all_fea, A_rul, seq_len, batteryid]
        batteryid += 1
    return retrieval_set


if __name__ == '__main__':
    path1 = './ne_data/2017-05-12_batchdata_updated_struct_errorcorrect.mat'
    path2 = './ne_data/2017-06-30_batchdata_updated_struct_errorcorrect.mat'
    path3 = './ne_data/2018-04-12_batchdata_updated_struct_errorcorrect.mat'

    temp1 = h5py.File(path1, 'r')
    temp2 = h5py.File(path2, 'r')
    temp3 = h5py.File(path3, 'r')

    batch1 = temp1['batch']
    batch2 = temp2['batch']
    batch3 = temp3['batch']

    cycle_life = dict()

    '''the first batch'''
    temp = temp1
    batch = batch1
    for bat_num in range(batch['cycles'].shape[0]):
        a = int(list(temp[batch['cycle_life'][bat_num, 0]])[0][0])
        # cl=temp[batch['cycle_life'][bat_num,0]].value
        cycle_life.update({'a' + str(bat_num): a})

    '''the second batch'''
    temp = temp2
    batch = batch2
    for bat_num in range(batch['cycles'].shape[0]):
        a = int(list(temp[batch['cycle_life'][bat_num, 0]])[0][0])
        cycle_life.update({'b' + str(bat_num): a})

    '''the third batch'''
    temp = temp3
    batch = batch3
    for bat_num in range(batch['cycles'].shape[0]):
        if bat_num != 23 and bat_num != 32:
            a = int(list(temp[batch['cycle_life'][bat_num, 0]])[0][0])
            cycle_life.update({'c' + str(bat_num): a})
            continue
    cycle_life.update({'c23': 2190})
    cycle_life.update({'c32': 2238})

    # There are four cells from batch1 that carried into batch2, we'll remove the data from batch2
    # and put it with the correct cell from batch1
    cycle_life['a0'] = cycle_life['a0'] + cycle_life['b7'] - 1
    cycle_life['a1'] = cycle_life['a1'] + cycle_life['b8'] - 1
    cycle_life['a2'] = cycle_life['a2'] + cycle_life['b9'] - 1
    cycle_life['a3'] = cycle_life['a3'] + cycle_life['b15'] - 1
    cycle_life['a4'] = cycle_life['a4'] + cycle_life['b16'] - 1

    # remove batteries that do not reach 80% capacity
    del cycle_life['a8']
    del cycle_life['a10']
    del cycle_life['a12']
    del cycle_life['a13']
    del cycle_life['a22']

    # remove data from a that carried into b
    del cycle_life['b7']
    del cycle_life['b8']
    del cycle_life['b9']
    del cycle_life['b15']
    del cycle_life['b16']

    # remove noisy channels from c
    del cycle_life['c37']
    del cycle_life['c2']
    del cycle_life['c23']
    del cycle_life['c32']
    del cycle_life['c38']
    del cycle_life['c39']
    print('loaded batteries')
    fea_num = 100
    n_cyc = 100
    in_stride = 10
    stride = 1

    v_low = 3.36
    v_upp = 3.60
    q_low = 0.61
    q_upp = 1.19
    lbl_factor = 3000
    aux_factor = 1190

    a0_4 = {}
    ay0_4 = {}
    list_a = [0, 1, 2, 3, 4]
    list_b = [7, 8, 9, 15, 16]
    for i, num in enumerate(list_a):
        print(i, num)
        fea_list = []
        label_list = []
        fea_list.append(None)
        label_list.append(None)
        b_num = list_b[i]
        bat_life = cycle_life['a' + str(num)]
        cyc_num = int(list(temp1[batch1['cycle_life'][num, 0]])[0][0]) - 1
        for j in range(1, cyc_num):
            # I = list(temp1[temp1[batch1['cycles'][num, 0]]['I'][j, :][0]])[0]
            # try:
            #     left_id = 0
            #     left = np.where(np.abs(I - 1) < 0.001)[0][left_id]
            #     while list(temp[temp[batch['cycles'][bat_num, 0]]['Qc'][j - 1, :][0]])[0][left] < 0.4:
            #         left_id += 1
            #         left = np.where(np.abs(I - 1) < 0.001)[0][left_id]
            #     right = np.where(np.abs(I - 1) < 0.001)[0][-1]
            # except:
            #     continue
            # if right - left <= 1:
            #     continue

            t = list(temp1[temp1[batch1['cycles'][num, 0]]['t'][j, :][0]])[0]#[left:right]
            V = list(temp1[temp1[batch1['cycles'][num, 0]]['V'][j, :][0]])[0]#[left:right]
            Vc = change(V, t, fea_num)
            Qc = list(temp1[temp1[batch1['cycles'][num, 0]]['Qc'][j, :][0]])[0]#[left:right]
            Qc = change(Qc, t, fea_num)
            QD = list(temp1[batch1['summary'][num, 0]]['QDischarge'][0, :])[j]
            tmp_fea = np.hstack((Vc.reshape(-1, 1), Qc.reshape(-1, 1)))

            fea_list.append(tmp_fea)
            label_list.append(QD)
        new_num = int(list(temp2[batch2['cycle_life'][b_num, 0]])[0][0]) - 1
        for j in range(new_num):

            I = list(temp2[temp2[batch2['cycles'][b_num, 0]]['I'][j, :][0]])[0]
            try:
                left_id = 0
                left = np.where(np.abs(I - 1) < 0.001)[0][left_id]
                while list(temp[temp[batch['cycles'][bat_num, 0]]['Qc'][j - 1, :][0]])[0][left] < 0.4:
                    left_id += 1
                    left = np.where(np.abs(I - 1) < 0.001)[0][left_id]
                right = np.where(np.abs(I - 1) < 0.001)[0][-1]
            except:
                continue
            if right - left <= 1:
                continue

            t = list(temp2[temp2[batch2['cycles'][b_num, 0]]['t'][j, :][0]])[0]#[left:right]
            V = list(temp2[temp2[batch2['cycles'][b_num, 0]]['V'][j, :][0]])[0]#[left:right]
            Vc = change(V, t, fea_num)
            Qc = list(temp2[temp2[batch2['cycles'][b_num, 0]]['Qc'][j, :][0]])[0]#[left:right]
            Qc = change(Qc, t, fea_num)
            QD = list(temp2[batch2['summary'][b_num, 0]]['QDischarge'][0, :])[j]
            tmp_fea = np.hstack((Vc.reshape(-1, 1), Qc.reshape(-1, 1)))

            fea_list.append(tmp_fea)
            label_list.append(QD)
        a0_4.update({num: fea_list})
        ay0_4.update({num: label_list})

    numBat1 = 0
    numBat2 = 0
    numBat3 = 0
    for key in cycle_life.keys():
        if 'a' in key:
            numBat1 += 1
        elif 'b' in key:
            numBat2 += 1
        elif 'c' in key:
            numBat3 += 1
    numBat = numBat1 + numBat2 + numBat3

    # Train and Test Split
    # If you are interested in using the same train/test split as the paper, use the indices specified below
    test_ind = np.hstack((np.arange(0, (numBat1 + numBat2), 2), 83))
    train_ind = np.arange(1, (numBat1 + numBat2 - 1), 2)
    secondary_test_ind = np.arange(numBat - numBat3, numBat)

    # print(len(train_ind),len(test_ind), len(secondary_test_ind))

    cycle_train = []
    cycle_test = []
    cycle_secondary = []

    for i, key in enumerate(cycle_life.keys()):
        if i in train_ind:
            cycle_train.append(key)
        elif i in test_ind:
            cycle_test.append(key)
        elif i in secondary_test_ind:
            cycle_secondary.append(key)

    print(len(cycle_train), len(cycle_test), len(cycle_secondary))


    # np.save('ne_data/meta_data.npy', [cycle_train, cycle_test, cycle_secondary])
    # exit(0)
    def get_xy(cyc_num, seq_len=100, pkl_dir='ne_data/'):
        fea = dict()
        label = dict()
        for i in cyc_num:
            key = i
            bat_life = cycle_life[key]
            fea_i = []
            label_i = []
            aux_lbl = []
            for j in range(11, bat_life):
                if key[0] == 'a':
                    temp = temp1
                    batch = batch1
                elif key[0] == 'b':
                    temp = temp2
                    batch = batch2
                else:
                    temp = temp3
                    batch = batch3

                bat_num = int(key[1:])
                if key[0] == 'a' and bat_num in [0, 1, 2, 3, 4]:
                    try:
                        tmp_fea = a0_4[bat_num][j - 1]
                        QD = ay0_4[bat_num][j - 1]
                    except:
                        continue
                else:
                    # I = list(temp[temp[batch['cycles'][bat_num, 0]]['I'][j - 1, :][0]])[0]
                    # try:
                    #     left_id = 0
                    #     left = np.where(np.abs(I - 1) < 0.001)[0][left_id]
                    #     while list(temp[temp[batch['cycles'][bat_num, 0]]['Qc'][j - 1, :][0]])[0][left] < 0.4:
                    #         left_id += 1
                    #         left = np.where(np.abs(I - 1) < 0.001)[0][left_id]
                    #     right = np.where(np.abs(I - 1) < 0.001)[0][-1]
                    # except:
                    #     continue
                    # if right - left <= 1:
                    #     continue

                    t = list(temp[temp[batch['cycles'][bat_num, 0]]['t'][j - 1, :][0]])[0]#[left:right]
                    V = list(temp[temp[batch['cycles'][bat_num, 0]]['V'][j - 1, :][0]])[0]#[left:right]
                    Qc = list(temp[temp[batch['cycles'][bat_num, 0]]['Qc'][j - 1, :][0]])[0]#[left:right]
                    Vc = change(V, t, fea_num)
                    Qc = change(Qc, t, fea_num)
                    QD = list(temp[batch['summary'][bat_num, 0]]['QDischarge'][0, :])[j - 1]
                    tmp_fea = np.hstack((Vc.reshape(-1, 1), Qc.reshape(-1, 1)))

                fea_i.append(np.expand_dims(tmp_fea, axis=0))
                label_i.append(bat_life - j)
                aux_lbl.append(QD)

            all_fea = np.vstack(fea_i)
            all_lbl = np.array(label_i).reshape(-1)
            aux_lbl = np.array(aux_lbl)

            all_fea_c = all_fea.copy()
            all_fea_c[:, :, 0] = (all_fea_c[:, :, 0] - v_low) / (v_upp - v_low)
            all_fea_c[:, :, 1] = (all_fea_c[:, :, 1] - q_low) / (q_upp - q_low)
            dif_fea = all_fea_c - all_fea_c[0:1, :, :]
            all_fea = np.concatenate((all_fea, dif_fea), axis=2)

            extracted_fea = []
            for cycidx in range(all_fea.shape[0]):
                tmp_extracted_fea = [aux_lbl[cycidx]]
                for i in range(4):
                    tmp_extracted_fea += preprocessv3(all_fea[cycidx, :, i], None)
                extracted_fea.append(tmp_extracted_fea)
            extracted_fea = np.array(extracted_fea)
            # all_lbl = all_lbl[seq_len - 1:]

            np.save(pkl_dir + key + 'v4.npy', extracted_fea)
            np.save(pkl_dir + key + '_rulv4.npy', all_lbl)

            # all_fea = np.lib.stride_tricks.sliding_window_view(extracted_fea, (seq_len, 9))
            # # aux_lbl = np.lib.stride_tricks.sliding_window_view(aux_lbl, (n_cyc,))
            # # all_fea = all_fea.squeeze(axis=(1, 2,))
            #
            # # all_fea = all_fea[::stride]
            # # all_fea = all_fea[:, ::in_stride, :, :]
            # # all_lbl = all_lbl[::stride]
            # # aux_lbl = aux_lbl[::stride]
            # # aux_lbl = aux_lbl[:, ::in_stride, ]
            #
            # all_fea_new = np.zeros(all_fea.shape)
            # all_fea_new[:, :, :, 0] = (all_fea[:, :, :, 0] - v_low) / (v_upp - v_low)
            # all_fea_new[:, :, :, 1] = (all_fea[:, :, :, 1] - q_low) / (q_upp - q_low)
            # all_fea_new[:, :, :, 2] = all_fea[:, :, :, 2]
            # all_fea_new[:, :, :, 3] = all_fea[:, :, :, 3]
            # print(f'{key} length is {all_fea_new.shape[0]}',
            #       'v_max:', '%.4f' % all_fea_new[:, :, :, 0].max(),
            #       'q_max:', '%.4f' % all_fea_new[:, :, :, 1].max(),
            #       'dv_max:', '%.4f' % all_fea_new[:, :, :, 2].max(),
            #       'dq_max:', '%.4f' % all_fea_new[:, :, :, 3].max())
            # all_lbl = all_lbl / lbl_factor
            # aux_lbl = aux_lbl / aux_factor

        return fea, label


    def save_extracted_feas(cycle_train, cycle_test, cycle_secondary):
        get_xy(cycle_train)
        get_xy(cycle_test)
        get_xy(cycle_secondary)


    save_extracted_feas(cycle_train, cycle_test, cycle_secondary)

# def get_dataset():
