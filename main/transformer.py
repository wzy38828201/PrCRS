from turtle import hideturtle
import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

def gelu(x):
    sqrt_two = 1.4142135623730951
    return x * 0.5 * (1.0 + torch.erf(x / sqrt_two))

ACTIVATION = {
    'relu': torch.relu,
    'selu': torch.selu,
    'gelu': gelu
}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000, _type='abs'):
        super(PositionalEncoding, self).__init__()
        assert _type in ['abs', 'learn']
        self.type = _type
        if _type == 'abs':
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)
        else:
            self.pe = nn.Embedding(max_len, d_model)
            torch.nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x):
        if self.type == 'abs':
            x = x + self.pe[:x.size(0), :]
        else:
            batch = x.size(0)
            seq_len = x.size(1)
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)[None, :].repeat(batch, 1)
            x = x + self.pe(position_ids)
        return x

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(self.linear_layer.weight,
                                      gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_normal_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)

class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                 bias=True, w_init='linear', wn=False, groups=1):
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias, groups=groups)

        if wn:
            self.conv = weight_norm(self.conv)

        nn.init.xavier_normal_(self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.conv(x)
        return x

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale):
        attention = torch.bmm(q, k.transpose(1, 2))
        attention = attention * scale
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.5,
                 mth_activation=False,
                 mth_layer_norm=True,
                 mth_linear=True):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.mth_linear = mth_linear
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.mth_activation = mth_activation
        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.mth_layer_norm = mth_layer_norm
        # multi-head attention之后需要做layer norm
        if mth_layer_norm:
            self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        if not self.mth_activation:
            key = self.linear_k(key)
            value = self.linear_v(value)
            query = self.linear_q(query)
        else:
            key = F.relu(self.linear_k(key), inplace=True)
            value = F.relu(self.linear_v(value), inplace=True)
            query = F.relu(self.linear_q(query), inplace=True)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = torch.tensor((key.size(-1) // num_heads) ** -0.5)
        # print("query.size=", query.size())
        # print("key.size=", key.size())
        # print("value.size=", value.size())
        context, attention = self.dot_product_attention(query, key, value, scale)

        # concat heads
        output = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        if self.mth_linear:
            output = self.linear_final(output)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        if self.mth_layer_norm:
            output = self.layer_norm(residual + output)

        return output, attention


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self, embedding_dim, num_heads=4, dropout=0.5, activation="gelu", 
                mth_activation=False, mth_layer_norm=True, mth_linear=True,):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(embedding_dim, num_heads, dropout=dropout,
                                            mth_activation=mth_activation,
                                            mth_layer_norm=mth_layer_norm,
                                            mth_linear=mth_linear)

        self.ds = Linear(embedding_dim, embedding_dim // 16)  # [B,T,C]
        self.conv1 = torch.nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 1, 3, padding=1)
        self.us = Linear(embedding_dim // 16, embedding_dim)

        self.activation = ACTIVATION[activation]
        self.res_weight = torch.nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))
        self.layer_norm1 = torch.nn.LayerNorm(embedding_dim)
        self.layer_norm2 = torch.nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # x = x.transpose(1, 2) # [B,T,C]
        _x = x
        x_in = x
        x_l, x_r = x_in[:, :-1, :], x_in[:, 1:, :]
        # x_l, x_r = x_in, x_in
        x, w = self.slf_attn(query=x_in, key=x_l, value=x_r)
        x = self.layer_norm1(_x + x * self.res_weight[0])  # add and norm
        
        _x = x
        x = self.ds(x)
        x = x.unsqueeze(-1).transpose(1, 3)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = x.transpose(1, 3).squeeze(-1)
        x = self.us(x)
        return self.layer_norm2(_x + x * self.res_weight[1])

    
