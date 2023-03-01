import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import time
from easydict import EasyDict as edict
from model import make_model
CONFIG = edict()
CONFIG.DEVICE = 'cuda'
CONFIG.LOSS_TYPE = 'regression_mse_var'
CONFIG.ENCODING_DIM = 32

CONFIG.DATA = edict()
CONFIG.DATA.NAME = 'MIT'

CONFIG.TRAIN = edict()
CONFIG.TRAIN.EPOCH = 2000
CONFIG.TRAIN.BATCH_SIZE = 4
CONFIG.TRAIN.LEARNING_RATE = 0.001
CONFIG.TRAIN.PRINT_LOSS_PERIOD = 1
CONFIG.TRAIN.PRINT_TRAIN_INFO = False

CONFIG.TEST = edict()
CONFIG.TEST.VISUALIZE_ALIGNMENT = True
CONFIG.TEST.VISUALIZE_RATE = 0.5
CONFIG.TEST.VISUALIZE_EVERY = 5
CONFIG.TEST.PLTSAVE = True

import logging



def compute_alignment_loss(embs,
                           batch_size,
                           steps=None,
                           seq_lens=None,
                           stochastic_matching=False,
                           normalize_embeddings=False,
                           loss_type='regression_mse_var',
                           similarity_type='l2',
                           temperature=0.1,
                           variance_lambda=0.001,
                           huber_delta=0.1,
                           normalize_indices=True):
    '''Computes alignment loss between sequences of embeddings '''
    '''  
    训练代码直接沿用的下面这篇论文的源码，在我们的论文中也讲了，可以自行下载
    
    可参考@InProceedings{Dwibedi_2019_CVPR,
                            author = {Dwibedi, Debidatta and Aytar, Yusuf and Tompson, Jonathan and Sermanet, Pierre and Zisserman, Andrew},
                            title = {Temporal Cycle-Consistency Learning},
                            booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
                            month = {June},
                            year = {2019},
                          }
    
    '''
    return 0

def load_MIT_data():
    '''
    :return: list = [mat1,mat2,...,matn]  mat1=[N1,input_features=14]
    '''
    paths = ['../data/MITdata/2017-05-12-npy','../data/MITdata/2017-06-30-npy','../data/MITdata/2018-04-12-npy']
    MIT_data = []
    for path in paths:
        i = 0
        files = os.listdir(path)
        save_folder_name = path.split('/')[-1]
        for file in files:
            i += 1
            file_path = path + '/' + file
            data = np.load(file_path)
            mat_tensor = torch.tensor(data, dtype=torch.float32, device='cuda')
            # print(mat_tensor.shape)
            MIT_data.append(mat_tensor)
    return MIT_data

def train_model(net,data,log,save_folder):
    CONFIG.DATA.LEN = len(data)
    CONFIG.TRAIN.BATCH_NUM = CONFIG.DATA.LEN // CONFIG.TRAIN.BATCH_SIZE
    optimizer = torch.optim.Adam(net.parameters(), lr=CONFIG.TRAIN.LEARNING_RATE)
    net = net.train()
    min_loss = 10
    LOSS = []
    last_best = None
    for epoch in range(2000):
        batch_loss = []

        for num in range(CONFIG.TRAIN.BATCH_NUM):
            batch_data = data[num * CONFIG.TRAIN.BATCH_SIZE:(num + 1) * CONFIG.TRAIN.BATCH_SIZE]
            embs = []
            for data_i in batch_data:
                embs.append(net(data_i))

            loss = compute_alignment_loss(embs, CONFIG.TRAIN.BATCH_SIZE, loss_type=CONFIG.LOSS_TYPE)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())


        LOSS.append(np.mean(batch_loss))
        log.info('epoch:{:}, loss:{:} '.format(epoch,np.mean(batch_loss)))

        '''   保存模型  '''
        time_now = time.strftime("%Y-%m-%d", time.localtime())
        save_name = save_folder + '/' + time_now +'_len' + str(CONFIG.DATA.LEN) + '_bs' + str(
                CONFIG.TRAIN.BATCH_SIZE) + '_epoch' + str(epoch + 1) + '.pth'
        if min_loss > np.mean(batch_loss):
            if last_best is not None:
                os.remove(last_best)
            min_loss = np.mean(batch_loss)
            best = save_folder + '/' + time_now + ' best_model' + ' epoch_' + str(epoch + 1) + '.pth'
            torch.save(net.state_dict(), best)
            last_best = best
        if (epoch + 1) % 100 == 0:
            torch.save(net.state_dict(), save_name)



def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    import random
    random.seed(0)
    log = logging.getLogger('my_log')
    data = load_MIT_data()
    print(f'the number of batteries is : {len(data)}')
    random.shuffle(data)

    for num in [60,40,20,10]:
        train_data = data[:num]
        path = f'result'
        make_dir(path)

        net = make_model(h=4,d_in=14,N=1,d_model=64,d_ff=64,d_out=32,dropout=0.5).to('cuda')
        train_model(net, train_data,log,path)



