import matplotlib.pyplot as plt

import numpy as np
import torch


'''
全局变量
'''
Windows_Len = 0.9
Slide_Times = 20


def compute_nearest_neighbor(P,Q):
    '''
    对于P中的每一个点P_i，找到其在Q中的最近邻，返回对应的索引值
    :param P: P:(Np, features=32)
    :param Q: Q:(Nq,features=32)
    :return: P_index, P2Q_index
    '''
    Np = P.shape[0]
    P_index = list(range(Np))
    P2Q_index = []
    for i in range(Np):
        P_i = P[i, :] # [1,32]
        Alpha = Q - P_i #[Nq,32]
        alpha = torch.pow(torch.norm(Alpha, dim=1), 2).cpu().detach().numpy()

        index = np.argmin(alpha)
        P2Q_index.append(index)
    return P_index, P2Q_index

def compute_window_alignment(P_index,Q_index,Q2P_index,win_len = 0.8, slide_times = 20):
    '''
    计算测试集和【一个训练集】的平均对齐长度
    :param P_index: 训练集编码P的index
    :param Q_index: 测试集编码Q的index
    :param Q2P_index: Q指向P的index
    :param win_len: 窗长，百分比，0-1
    :param slide_times: 滑动次数（平均次数）
    :param log: logging
    :return: alignment_p_len, alignment_q_len
    '''
    Np = len(P_index)
    Nq = len(Q_index)
    window_len = int(win_len * Nq)
    alignment_q_len = window_len
    slide_range = Nq - window_len
    slide_step = slide_range/slide_times

    P_LEN = []
    for slide in range(slide_times):
        start = int(slide * slide_step)
        end = int(slide * slide_step + window_len) if int(slide * slide_step + window_len) < Nq else Nq-1
        alignment_p_len = Q2P_index[end] - min(Q2P_index[start:end]) + 1
        P_LEN.append(alignment_p_len)
    alignment_p_len = np.mean(np.array(P_LEN))
    return alignment_p_len, alignment_q_len

def compute_rul_and_weight(P_index,Q_index,Q2P_index):
    '''
    计算测试集与【一个训练集】对齐的寿命和权重
    :param P_index:
    :param Q_index:
    :param Q2P_index:
    :return: rul, weight
    '''
    sub_p_len, sub_q_len = compute_window_alignment(P_index,Q_index,Q2P_index,win_len = Windows_Len, slide_times = Slide_Times)
    proportion = sub_q_len / sub_p_len
    T_e_j = Q2P_index[-1]
    Np = len(P_index)
    rul = proportion * (Np - T_e_j)
    weight = 1 - abs(sub_q_len-sub_p_len) / sub_q_len

    return rul, weight

def compute_basic_rul(P_index,Q_index,Q2P_index):
    '''
    计算测试集与【一个训练集】对齐的寿命，不划窗
    :param P_index: 训练集编码P的index
    :param Q_index: 测试集编码Q的index
    :param Q2P_index: Q指向P的index
    :return: basic_rul
    '''
    T_s_index = Q2P_index[0]
    T_e_index = Q2P_index[-1]
    alignment_p_len = T_e_index - T_s_index + 1
    alignment_q_len = len(Q_index)
    num_p = len(P_index)
    basic_rul = (num_p-T_e_index)/alignment_p_len * alignment_q_len
    return basic_rul


def embedding(net,p,q):
    '''
    对p,q进行编码得到P，Q
    :param net: 编码网络
    :param p: 训练集p [len_p,features=14]
    :param q: 测试集q [len_q,features=14]
    :return: P [len_p, 32]   Q [len_q, 32]
    '''
    net = net.eval()
    P = net(p)  # x[N,C_in,L_in]
    Q = net(q)
    return P,Q

def embedding2(net,p,q):
    '''
    对p,q进行编码得到P，Q
    :param net: 编码网络
    :param p: 训练集p [len_p,features=14]
    :param q: 测试集q [len_q,features=14]
    :return: P [len_p, 32]   Q [len_q, 32]
    '''
    p = p.view(p.shape[0], 1, -1)
    q = q.view(q.shape[0], 1, -1)
    assert p.shape[2] == q.shape[2]
    net = net.eval()
    P = net(p)
    Q = net(q)
    return P,Q

def Compute_RUL(net, train_set, test):
    '''
    把测试集和【所有训练集】对齐，计算剩余寿命
    :param net: 编码网络
    :param train_set: 训练集集合
    :param test: 测试集
    :return: mean_rul, weighted_rul
    '''
    q = test
    RUL = []
    weight = []
    BASIC_RUL = []
    for i in range(len(train_set)):
        # print(i+1,end=' ')
        p = train_set[i]
        P, Q = embedding(net,p,q)
        P_index, P2Q_index = compute_nearest_neighbor(P, Q)
        Q_index, Q2P_index = compute_nearest_neighbor(Q, P)

        # 计算测试集与【一个训练集】对齐的寿命和权重
        rul, w = compute_rul_and_weight(P_index,Q_index,Q2P_index)
        basic_rul = compute_basic_rul(P_index,Q_index,Q2P_index)

        RUL.append(rul)
        weight.append(w)
        BASIC_RUL.append((basic_rul))


    mean_basic_rul = int(np.mean(BASIC_RUL))
    RUL = np.array(RUL)
    weight = np.array(weight)



    weight_tensor = torch.tensor(weight, dtype=torch.float32)
    softmax = torch.nn.Softmax(dim=0)
    weight_tensor = softmax(weight_tensor)
    RUL_tensor = torch.tensor(RUL, dtype=torch.float32)
    weighted_RUL = torch.dot(weight_tensor, RUL_tensor)
    w_rul = int(weighted_RUL.item())
    return mean_basic_rul, w_rul

if __name__ == '__main__':
    ##
    pass


