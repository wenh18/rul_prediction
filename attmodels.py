import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import math,copy
import numpy as np


# class embedding()


def clones(module,N):
    '''
    produce N identical layers
    :param module:
    :param N:
    :return:
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def my_attention(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.t()) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class my_MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(my_MultiHeadedAttention, self).__init__()
        self.d_k = d_model // h
        self.h = h
        self.d_model = d_model
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.fc = nn.Linear(h*d_model,d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):

        query, key, value = [l(x).view(-1, self.d_model) for l, x in zip(self.linears, (query, key, value))]


        X = []
        for _ in range(self.h):
            x, self.attn = my_attention(query, key, value,dropout=self.dropout)
            X.append(x)

        out = torch.cat(X, dim=1)
        return self.fc(out)

class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
        #return self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=2500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)   #[2500,14]
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):

        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff,dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class FC1Layer(nn.Module):
    def __init__(self,d_in,d_model):
        super(FC1Layer,self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.layer = nn.Sequential(
            nn.Linear(d_in,d_model),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)

class EncodingNet(nn.Module):
    def __init__(self,fc1,encoder,src_embed,fc):
        super(EncodingNet,self).__init__()
        self.fc1 = fc1
        self.encoder = encoder
        self.src_embed = src_embed
        self.fc = fc

    def forward(self,src):
        src1 = self.fc1(src)
        y = self.encoder(self.src_embed(src1))
        out = self.fc(y.view(y.shape[0],1,-1))
        return out

class CovNet(nn.Module):
    def __init__(self,input_channel=1,d_out=32):
        super(CovNet,self).__init__() # N*1*64
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel,16,5), # in_channels, out_channels, kernel_size
            # N*16*60
            nn.MaxPool1d(2,stride=2),  # N*16*30
            nn.LeakyReLU(),
            nn.Conv1d(16, 32, 5), # N*32*26
            nn.MaxPool1d(2, stride=2), # N*32*13
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32*13,128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,d_out),
        )
    def forward(self,x):
        y = self.conv(x)
        out = self.fc(y.view(x.shape[0],-1))
        return out

def make_encoder(h=4,N=1,d_in=14,d_model=64,d_ff=64,d_out=32,dropout=0):
    c = copy.deepcopy
    attn = my_MultiHeadedAttention(h,d_model,dropout)
    ff = PositionwiseFeedForward(d_model,d_ff,dropout)
    position = PositionalEncoding(d_model,dropout)
    fc1 = FC1Layer(d_in,d_model)
    cov = CovNet(d_out=d_out)
    model = EncodingNet(
        fc1,
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        c(position),
        cov
    )
    return model

class RelationNet(nn.Module):

    def __init__(self,fc1,encoder,src_embed,fc, num_features):
        super(RelationNet,self).__init__()
        self.fc1 = fc1
        self.encoder = encoder
        self.src_embed = src_embed
        self.fc = fc
        # self.average_pooling = nn.AdaptiveAvgPool1d
        self.linear = nn.Linear(num_features, 1)

    def forward(self,src):
        src1 = self.fc1(src)
        y = self.encoder(self.src_embed(src1))
        out = self.fc(y.view(y.shape[0],1,-1))
        out_t = out.T
        fea = nn.functional.adaptive_avg_pool1d(out_t, 1)
        fea = fea.T
        score = self.linear(fea)
        return score

def make_relation_model(h=4,N=1,d_in=14,d_model=64,d_ff=64,d_out=32,dropout=0):
    c = copy.deepcopy
    attn = my_MultiHeadedAttention(h,d_model,dropout)
    ff = PositionwiseFeedForward(d_model,d_ff,dropout)
    position = PositionalEncoding(d_model,dropout)
    fc1 = FC1Layer(d_in,d_model)
    cov = CovNet(d_out=d_out)
    model = RelationNet(
        fc1,
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        c(position),
        cov,
        d_in
    )
    return model

def make_model(h=4,N=1,d_in=14,d_model=64,d_ff=64,d_embedding=32,dropout=0):
    encoder = make_encoder(h,N,d_in,d_model,d_ff,d_embedding,dropout)
    relationdecoder = make_relation_model(h,N,d_embedding,d_model,d_ff,d_embedding,dropout)
    return encoder, relationdecoder

# class ScoreNet(nn.Module):
#     def __init__(self,h=4,N=1,d_in=14,d_model=64,d_ff=64,d_out=32,dropout=0):
#         super(ScoreNet,self).__init__()
#         self.encoder = make_encoder(h, N, d_in, d_model, d_ff, d_out, dropout)
#         self.relationnet = make_relation_model(h, N, d_in, d_model, d_ff, d_out, dropout)
#
#     def forward(self, src, target):
#         for src_series in src:
#
#         return score

if __name__ == '__main__':
    x = torch.randn(1000,14)
    x2 = torch.rand(1500,14)
    encoder, relationmodel = make_model(h=4,N=1,d_in=14,d_model=64,d_ff=64,d_embedding=32,dropout=0.5)
    # relationmodel = make_relation_model(h=4,N=1,d_in=32,d_model=64,d_ff=64,d_out=32,dropout=0.5)
    y = encoder(x)
    print(y.shape, x.shape)
    y1 = relationmodel(y)
    # import pdb;pdb.set_trace()
    print(y.shape)
    print(y1.shape)






