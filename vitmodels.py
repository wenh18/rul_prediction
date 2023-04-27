import torch
from torch import nn
import math
from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
import torch.nn.functional as F

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() *
                    -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim,
                                           dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(
                        dim,
                        Attention(dim,
                                  heads=heads,
                                  dim_head=dim_head,
                                  dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):

    def __init__(self,
                 num_classes,
                 in_dim,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 pool='cls',
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch_dim = channels * patch_height * patch_width
        assert pool in {
            'cls', 'mean'
        }, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),  # torch.Size([1, 49, 3072])
        #     nn.Linear(in_dim, dim),
        # )

        self.to_patch_embedding = nn.Linear(in_dim, dim)

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding = PositionalEmbedding(d_model=dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim,
                                       dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim),
                                      nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding(x)
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class lstm_encoder(nn.Module):

    def __init__(self, indim, hiddendim, fcdim, outdim, n_layers, dropout=0.4):
        super(lstm_encoder, self).__init__()
        self.lstm1 = torch.nn.LSTM(input_size=indim,
                                   hidden_size=hiddendim,
                                   batch_first=True,
                                   bidirectional=False,
                                   num_layers=n_layers)
        # self.lstm2 = torch.nn.LSTM(input_size=hiddendim, hidden_size=hiddendim, batch_first=True, bidirectional=False, num_layers=n_layers)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(hiddendim * n_layers, fcdim)
        self.bn1 = torch.nn.LayerNorm(normalized_shape=fcdim)
        self.fc2 = torch.nn.Linear(fcdim, outdim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.fill_(0)

    def forward(self, x):
        out, (h, c) = self.lstm1(x)
        # out, (h, c) = self.lstm2(h)
        h = h.reshape(x.size(0), -1)
        h = self.dropout(h)
        # h = h.squeeze()
        x = self.fc1(h)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = F.sigmoid(x)
        return x


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)

    def forward(self, source, target):
        # import pdb;pdb.set_trace()
        # n_examples = a.size()[0] * a.size()[1]
        # a = a.reshape(n_examples, -1)
        # b = b.reshape(n_examples, -1)
        cosine = F.cosine_similarity(source, target)
        return cosine


class FNNRelationNetwork(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(
            FNNRelationNetwork,
            self,
        ).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)

    def forward(self, source, target):
        x = torch.cat((source, target), 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)  # hiddensize->hiddensize
        x = self.fc2(x)
        return x


class FFNEncoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(
            FFNEncoder,
            self,
        ).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)

    def forward(self, seq):
        # print("SEQ: ", seq.shape)
        x = seq.reshape(seq.size(0), -1)
        # print("X: ", x.shape)
        x = self.fc1(x)
        # print("X1: ", x.shape)
        # x = self.bn1(x)
        x = F.relu(x)  # hiddensize->hiddensize
        x = self.fc2(x)
        # print("X2: ", x.shape)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


if __name__ == '__main__':
    # efficient_transformer = Linformer(
    #     dim=128,
    #     seq_len=49 + 1,  # 7x7 patches + 1 cls-token
    #     depth=12,
    #     heads=8,
    #     k=64
    # )
    rm = RelationNetwork(1000, 2000)
    x = torch.rand(4, 1000)
    y = rm(x)
    print(y.size())
    exit(0)
    model = ViT(
        dim=512,  # transformer input dimension
        in_dim=14,
        num_classes=1000,
        channels=3,
        depth=6,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1)
    import numpy as np

    x = torch.Tensor(np.random.rand(1, 1500, 14))
    y = model(x)
    print(y.shape)