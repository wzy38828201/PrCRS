import torch
import os
import numpy as np
from torch import nn

from squeezeformer.encoder import SqueezeformerBlock
from transformer import FFTBlock

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
        nn.init.xavier_normal_(self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.conv(x)
        return x
    
class tranMedical(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_class=3):
        super(tranMedical, self).__init__()
        self.conv = Conv(in_channels=input_dim, out_channels=hidden_dim, kernel_size=5, stride=2)
        self.fftblock = SqueezeformerBlock(encoder_dim =hidden_dim, num_attention_heads=1)
        # self.fftblock = FFTBlock(embedding_dim=hidden_dim, num_heads=1)
        # self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim//2, bidirectional=True, batch_first=True)
        self.fc1 = Linear(in_dim=hidden_dim, out_dim=hidden_dim//4)
        self.fc = Linear(in_dim=hidden_dim//4, out_dim=num_class)
        self.activate = nn.ReLU()

    def forward(self, x):
        # print('x: ', x.shape)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.activate(x)
        # print('x: ', x.shape)
        x = x.permute(0, 2, 1)
        x = self.fftblock(x)

        # hidden, x = self.rnn(x)
        # x = torch.mean(hidden, dim=1)

        x = torch.mean(x, dim=1)
        x = self.fc1(x)
        x = self.fc(x)
        
        return x

