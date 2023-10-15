
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn import preprocessing
import math
import torch.utils.data as Data
from matplotlib import pyplot as plt
from My_utils.evaluation_scheme import evaluation
from My_utils.preprocess_data import divide_data
import scipy.optimize as optimize
import warnings
import torch.nn.functional as F
import time
from torch.nn.utils import weight_norm
import pandas as pd
import numpy as np
from PyEMD import EEMD
from scipy.signal import hilbert

# paras
seq_len=360
pre_len=180
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########       decomposed block     ###########
def decomposed_block(data):
    eemd = EEMD()
    eemd.eemd(data)  # need array
    t = np.arange(0, len(data), 1)
    imfs, res = eemd.get_imfs_and_residue()

    # calculate the mean instaneous frequency##############
    analytic_signal = hilbert(imfs)
    inst_phase = np.unwrap(np.angle(analytic_signal))
    inst_freqs = np.diff(inst_phase) / (2 * np.pi * (t[1] - t[0]))
    inst_freqs = np.concatenate((inst_freqs, inst_freqs[:, -1].reshape(inst_freqs[:, -1].shape[0], 1)),
                                axis=1)  ##row vector!!
    MIF = np.mean(inst_freqs, axis=1)

    # transform to period
    sample_interval = 60  # min
    period = sample_interval / (60 * MIF)
    return period


########       Frequency_enhanced block  ##################

def Frequency_enhanced(time_series, period):  # time_series shape(1,1,timestep)   also called pattern_identification
    time_series = time_series.cpu()
    x_list = np.arange(1, time_series.size(2) + 1, 1)
    y_list = time_series.squeeze().numpy()
    pattern = 0
    para_list = []
    pattern_list = []
    res_list = []
    for i in range(len(period) - 1, -1, -1):
        global T
        T = period[i]
        a = [5, 5, 5, 7, 7, 9, 9]
        p = np.random.rand(a[i])
        para, _ = optimize.curve_fit(func_fourier, x_list, y_list, p0=p)
        y_fit = [func_fourier(a, *para) for a in x_list]
        pattern = np.array(y_fit) + np.array(pattern)
        time_series = time_series - np.array(y_fit)
        res = time_series
        para_list.append(para)
        pattern_list.append(np.array(y_fit))
        res_list.append(res)

    return pattern_list, res_list[-1]


############   fourier reconstruct in time domain  ##########

def func_fourier(x, *b):
    w = T
    ret = 0
    a = b[:-2]
    for deg in range(0, int(len(a) / 2) + 1):
        ret += a[deg] * np.cos(deg * 2 * math.pi / w * x) + a[len(a) - deg - 1] * np.sin(deg * 2 * math.pi / w * x)
    return ret + b[-1]

################    TCN_block  ##############
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride=1, dilation=None,
                 padding=3, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        if dilation is None:
            dilation = 2
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        # FC=nn.Linear(2,1).to(device)
        # Fusion=FC(self.network(x)).squeeze()
        # pattern=torch.mean(self.network(x[:,:,:-1]), dim=2)
        # data=torch.mean(self.network(x[:, :, -1].unsqueeze(2)), dim=2)
        # TEMP=torch.cat([torch.flatten(pattern).unsqueeze(1), torch.flatten(data).unsqueeze(1)], dim=1)
        # Fusion = FC(TEMP)
        # Fusion=torch.mean(torch.cat([pattern.unsqueeze(2),data],dim=2), dim=2)

        # return self.network(x)
        # return Fusion
        # return pattern
        # return data
        # return torch.sum(self.network(x), dim=2,keepdim=True)


        # outs=torch.mean(self.network(x), dim=2)

        outs = self.network(x)
        temp=outs.resize(outs.shape[0],outs.shape[1],23,9)
        a=0
        for i in range(9):
            a=a+temp[:,:,:,i]
        final_out=a/9
        return final_out

############   weighted loss func  ##########

def weight_loss(y, pre_y):

    func = nn.MSELoss(reduce = False)  #return vector
    weight=torch.tensor(np.linspace(1, 0.0001, y.size(2))).float().to(device)
    b_weight=weight.repeat(y.size(0),1).unsqueeze(1)
    loss=func(y, pre_y)
    adj_loss=torch.mean(b_weight*loss)

    return adj_loss

