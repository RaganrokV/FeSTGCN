# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from PyEMD import EMD,EEMD
from scipy.signal import hilbert
import math
from sklearn import preprocessing

from torch_geometric_temporal.nn.recurrent.dcrnn import DConv
from torch.nn.utils import weight_norm
import torch
from torch import nn
import scipy.optimize as optimize

import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
import os
import random
warnings.filterwarnings("ignore")

#%%
def decomposed_block( data, sample_interval):
    eemd = EEMD()
    period = []
    for i in range(23):
        eemd.eemd(data[:, i, 0].numpy())  # need array tensor to array
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
        # sample_interval unit min
        period.append(sample_interval / (60 * MIF)[:8])

    return period


def adjust_period(period, target_value):
    for i in range(len(period)):
        arr = period[i]
        scale = target_value / arr[4]  # 计算缩放比例
        arr *= scale  # 按比例缩放所有元素
        arr[4] = target_value  # 将第5个元素设置为目标值
        period[i] = arr
    return period


def func_fourier( x, *b):
    w = T
    ret = 0
    a = b[:-2]
    for deg in range(0, int(len(a) / 2) + 1):
        ret += a[deg] * np.cos(deg * 2 * math.pi / w * x) + a[len(a) - deg - 1] * np.sin(deg * 2 * math.pi / w * x)
    return ret + b[-1]


def Frequency_enhanced(time_series, period):  # time_series shape(1,1,timestep)   also called pattern_identification
    time_series = time_series.cpu().numpy()  # 将 time_series 转换为 numpy 数组
    batch_size = time_series.shape[0]  # 获取 batch 大小
    x_list = np.arange(1, time_series.shape[1] + 1, 1)  # 根据时间序列的维度获取 x_list
    y_list = time_series
    pattern = np.zeros_like(y_list)  # 初始化 pattern 数组
    para_list = []
    pattern_list = []
    res_list = []
    for i in range(len(period) - 1, -1, -1):
        global T
        T = period[i]
        a = [5, 5, 5, 7, 7, 7, 9, 9]
        p = np.random.rand(a[i])
        para = []
        for j in range(batch_size):
            para_j, _ = optimize.curve_fit(func_fourier, x_list, y_list[j], p0=p)  # 对每个 batch 中的时间序列拟合参数
            para.append(para_j)
        para = np.array(para)
        y_fit = np.array([func_fourier(x_list, *para[j]) for j in range(batch_size)])  # 根据参数计算拟合值
        pattern += y_fit
        time_series -= y_fit
        res = time_series
        para_list.append(para)
        pattern_list.append(y_fit)
        res_list.append(res)

    return pattern_list, res_list[-1]
#%%
class FeSTGCN_net(torch.nn.Module):
    def __init__(self, in_channels, embed_channels, out_channels,adj_matrix,adj_period):
        super(FeSTGCN_net, self).__init__()
        """:param"""
        self.ADJ=torch.tensor(adj_matrix).to(device)
        self.embed_size=12
        self.weight_inflation=25

        self.Spatial_cell = DConv(in_channels, embed_channels, 2)
        # self.Temporal_cell =nn.LSTM(in_channels,embed_channels,num_layers=2,dropout=0.5,
        #                             batch_first=True,bidirectional=True)
        self.Temporal_cell=TemporalConvNet(num_inputs=in_channels, num_channels=[64,embed_channels],
                                           kernel_size=2, dropout=0.1)

        self.Frequency_cell=Frequency_net(in_channels, out_channels, adj_period)

        """decoder part"""
        self.decoder_rnn=torch.nn.GRU(self.embed_size,
                          embed_channels//2,
                          num_layers=1,
                          batch_first=True)
        self.decoder_lin= nn.Linear(embed_channels//2, out_channels)

        self.Att=nn.MultiheadAttention(in_channels, 2, batch_first=True)

        self.linear=torch.nn.Linear(embed_channels*3, out_channels)
        # self.embedding = torch.nn.Linear(embed_channels * 3, self.embed_size)
        self.embedding = torch.nn.Linear(embed_channels +embed_channels+9, self.embed_size)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()



    def forward(self, x, edge_index, edge_weight ):

        """att weight"""
        _, attn_output_weights = self.Att(x, x, x)

        Mean_weight = torch.mean(self.ADJ * attn_output_weights, dim=0)
        index = torch.nonzero(Mean_weight, as_tuple=True)
        attention_scorce = Mean_weight[index[0], index[1]]
        Att_weigth=edge_weight*attention_scorce

        Spatial_info = self.Spatial_cell(x, edge_index, self.weight_inflation*Att_weigth) #Spatial_info(B, N_nodes, F_out)
        Spatial_info = F.relu(Spatial_info)

        Temporal_info = self.Temporal_cell(x.permute(0,2,1))
        Temporal_info = F.relu(Temporal_info.permute(0,2,1))

        Frequency_info = self.Frequency_cell(x)
        Frequency_info = F.relu(Frequency_info)


        CAT = torch.cat((Temporal_info, Spatial_info,Frequency_info), dim=2)

        embeding=self.embedding(CAT)

        h_feature, _=self.decoder_rnn(embeding)
        h_feature = F.relu(h_feature)
        Fusion=self.decoder_lin(h_feature)

        return Fusion

"""frequency block"""
class Frequency_net(nn.Module):
    def __init__(self,  seq_channels, out_channels,adj_period):
        super(Frequency_net, self).__init__()
        self.period=adj_period
        self.linear = torch.nn.Linear(seq_channels, out_channels)

    def forward(self, x):

        Freq_info = []

        for i in range(x.shape[1]):
            pattern_list, res = Frequency_enhanced(x[:, i, :], self.period[i])
            pattern_array = np.stack(pattern_list, axis=2)
            # 在第三个维度末尾添加 res
            Freq = torch.tensor(np.concatenate((pattern_array, np.expand_dims(res, axis=2))
                                               , axis=2)).float().to(device)

            TEMP = self.linear(Freq.permute(0, 2, 1))
            Freq_info.append(TEMP)


        Frequency = torch.stack(Freq_info, dim=2)[:,:,:,-1].permute(0, 2, 1)

        return Frequency


"""temporal block"""
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
    def __init__(self, n_inputs=24, n_outputs=1, kernel_size=2, stride=1, dilation=None, padding=3, dropout=0.2):
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
        return self.network(x)

#%%  hyper parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len=96
pre_len=1
batch_size=16
# set seed for reproductive 42 is the answer to the universe
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
#%%
data_csv = pd.read_csv(r'1-Fed-traffic/periodic_sample.csv')
adj=pd.read_csv(r'1-Fed-traffic/ADJ.csv')
adj_matrix=adj

period = np.load('1-Fed-traffic/period.npy')
adj_period=adjust_period(period,24)

Normalization = preprocessing.MinMaxScaler()
Norm_TS = Normalization.fit_transform(data_csv.values)
#%%
Fe_Net_dict1 = torch.load('1-Fed-traffic/models/FeSTGCN_1.pt')
Fe_Net1 = FeSTGCN_net(in_channels=seq_len, embed_channels=32,
                  out_channels=1,adj_matrix=adj_matrix.values,
                  adj_period=adj_period).to(device)
Fe_Net1.load_state_dict(Fe_Net_dict1)

Fe_Net_dict2 = torch.load('1-Fed-traffic/models/FeSTGCN_2.pt')
Fe_Net2 = FeSTGCN_net(in_channels=seq_len, embed_channels=32,
                  out_channels=2,adj_matrix=adj_matrix.values,
                  adj_period=adj_period).to(device)
Fe_Net2.load_state_dict(Fe_Net_dict2)

Fe_Net_dict4 = torch.load('1-Fed-traffic/models/FeSTGCN_4.pt')
Fe_Net4 = FeSTGCN_net(in_channels=seq_len, embed_channels=32,
                  out_channels=4,adj_matrix=adj_matrix.values,
                  adj_period=adj_period).to(device)
Fe_Net4.load_state_dict(Fe_Net_dict4)
#%%
def get_weight(Fe_Net):
    weight_matrix = torch.mean(torch.abs(Fe_Net.embedding.weight.data), dim=0)[-8:].cpu()

    return weight_matrix

weight_matrix1=get_weight(Fe_Net1)
weight_matrix2=get_weight(Fe_Net2)
weight_matrix4=get_weight(Fe_Net4)
# 将权重矩阵转换为DataFrame类型
weight_df = pd.DataFrame({'1-th step': weight_matrix1.numpy(),
                          '24-th step': weight_matrix2.numpy(),
                          '168-th step': weight_matrix4.numpy()})

# 设置行名和列名
weight_df.index = ['Peak', 'Semi-commuting', 'Semi-daily', 'Daily', 'Weekend', 'Weekday', 'Transition', 'Residual']
weight_df.columns = ['1-th step', '2-th step', '4-th step']
sorted_df = weight_df.assign(Sum=weight_df.sum(axis=1)).sort_values(by='Sum', ascending=True).drop(columns='Sum')
#%%
# 画图
plt.rcParams["legend.fontsize"] = 30
fig, ax = plt.subplots(figsize=(12, 6))
ax.grid(ls="--", lw=0.3, dashes=(8, 4), color="lightgray")
ax.hlines(y=sorted_df.index, xmin=sorted_df.min(axis=1), xmax=sorted_df.max(axis=1), color='skyblue')
ax.plot(sorted_df, sorted_df.index, "--D", markersize=15, label=sorted_df.columns)

plt.yticks(fontsize=22, font='Times New Roman')
plt.xticks(fontsize=22, font='Times New Roman')
ax.set_xlim(0, 1.8)
plt.legend(fontsize=30,
           prop={'family': 'Times New Roman'}, loc='lower right')

plt.xlabel('Importance', fontsize=30, font='Times New Roman')
fig.tight_layout()
plt.savefig(r"1-Fed-traffic/figs/importance.svg", dpi=600)
plt.show()

#%%
# Preds=[]
# GT=[]
# for c in range(10):
#
#     for j in range(5):  # 分别计算5辆车
#
#         """sample"""
#
#         car_test, RDR_std, RDR_mean, EC_std, EC_mean, SOC_std, SOC_mean = get_perm_data(sample_snapshot[j], c)
#         """prediction"""
#         Metric = []
#         all_simu = []
#         all_real = []
#         all_X = []
#         for i, (x, y) in enumerate(car_test):
#
#             sum_pred = []
#             for k in range(10):
#                 ET = BET[k].eval()
#                 sum_pred.append(ET(x.to(device)).float())
#
#             pred = sum([weight * pred for weight,
#                                           pred in zip(weights_normalized_A, sum_pred)])
#
#             # pred = sum_pred / num_models
#             Norm_pred = pred.data.cpu().numpy()
#             all_simu.append(Norm_pred)
#             all_real.append(y.squeeze(1))
#             all_X.append(x)
#
#         Preds.append(np.vstack(all_simu) * RDR_std + RDR_mean)
#         GT.append(np.vstack(all_real) * RDR_std + RDR_mean)
#
# #%%
# import matplotlib.pyplot as plt
#
# os.environ['PYTHONHASHSEED'] = str(42)
# np.random.seed(42)
# random.seed(42)
# labels = ['Voltage', 'Current', 'Max_V', 'Min_V', 'Max_T', 'Min_T', 'Speed', 'SOC', 'Degradation', 'EC']
# plt.rcParams['font.family'] = 'Times New Roman'
# fig, axes = plt.subplots(2, 5, figsize=(15, 8))
# fig.tight_layout()
#
# for i in range(10):  # 调整循环范围为10
#     merge_pred = np.vstack([GT[i][:], Preds[i][:]])
#     observed_diff = abs(Preds[i].mean() - merge_pred.mean())
#     extreme_values = []
#     sample_d = []
#
#     for _ in range(10000):
#         sample_mean = np.random.choice(merge_pred[:, 0], size=Preds[i].shape[0]).mean()
#         sample_diff = abs(sample_mean - merge_pred.mean())
#         sample_d.append(sample_diff)
#         extreme_values.append(sample_diff >= observed_diff)
#
#     p = np.sum(extreme_values) / 10000
#
#     y, x, _ = axes.flat[i].hist(sample_d, alpha=0.6)
#     axes.flat[i].vlines(observed_diff, 0, max(y), colors='red', linestyles='dashed')
#     axes.flat[i].text(0.92, 0.95, f"{labels[i]}\nP-value: {p:.3f}",
#                       ha='right', va='top', transform=axes.flat[i].transAxes,
#                       fontsize=17, fontname='Times New Roman')
#
# fig.subplots_adjust(top=0.9, bottom=0.1)
#
# # 添加正下方的标题
# fig.text(0.5, 0.02, "(a) Causality test for Vehicle A", ha='center', fontsize=22)
#
#
# plt.savefig(r"4-SOC_prediction/FIGS/causation_A.svg", dpi=600)
# # plt.xlabel("Simulation distributions and relative p-values")
# plt.show()
