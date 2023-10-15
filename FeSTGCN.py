# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from PyEMD import EMD,EEMD
from scipy.signal import hilbert
import math
from sklearn import preprocessing
import scipy.sparse as sp
from torch_geometric_temporal.nn.recurrent.dcrnn import DConv
from torch.nn.utils import weight_norm
import torch
from torch import nn
import scipy.optimize as optimize
from My_utils.evaluation_scheme import evaluation
from My_utils.preprocess_data import  divide_data
import torch.utils.data as Data
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings

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
# data_csv = pd.read_csv(r'periodic_sample.csv') #for debug
# adj=pd.read_csv(r'ADJ.csv')
# adj_matrix=adj

Normalization = preprocessing.MinMaxScaler()
Norm_TS = Normalization.fit_transform(data_csv.values)
#%%
"""divide data"""
train_size=3500 #80%
val_size=450 #10%
trainX, trainY, valX, valY, testX, testY = divide_data(data=Norm_TS, train_size=train_size,
                                           val_size=val_size,
                                           seq_len=seq_len, pre_len=pre_len)
trainX, trainY = trainX.transpose(1, 2).float(), trainY.transpose(1, 2).float()
valX, valY = valX.transpose(1, 2).float(), valY.transpose(1, 2).float()
testX, testY = testX.transpose(1, 2).float(), testY.transpose(1, 2).float()

adj = sp.coo_matrix(adj)
values = adj.data
indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
edge_index = torch.LongTensor(indices)  # PyG框架需要的coo形式
edge_index=torch.LongTensor(edge_index).to(device)


edge_weight=np.ones(edge_index.shape[1])
edge_weight=torch.FloatTensor(edge_weight).to(device)

train_dataset = Data.TensorDataset(trainX, trainY)
val_dataset = Data.TensorDataset(valX, valY)
test_dataset = Data.TensorDataset(testX, testY)

train_loader = Data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size, shuffle=False)
val_loader = Data.DataLoader(dataset=val_dataset,
                               batch_size=batch_size, shuffle=False)
test_loader = Data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size, shuffle=False)

#%%
# period=decomposed_block(trainX, 15)
# np.save('1-Fed-traffic/period.npy', period)

# 从 npy 文件中读取 period 对象
period = np.load('1-Fed-traffic/period.npy')
# period = np.load('period.npy')
#%%
adj_period=adjust_period(period,24)
net = FeSTGCN_net(in_channels=seq_len, embed_channels=32,
                  out_channels=pre_len,adj_matrix=adj_matrix.values,
                  adj_period=adj_period).to(device)
optimizer = torch.optim.AdamW(net.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.98)
loss_func = nn.MSELoss()

# %%
best_val_loss = float("inf")
best_model = None
#           train
train_loss_all = []
net.train()  # Turn on the train mode
total_loss = 0.

for epoch in range(50):
    train_loss = 0
    train_num = 0
    for step, (x, y) in enumerate(train_loader):

        time_start = time.time()

        optimizer.zero_grad()

        x, y = x.to(device), y.to(device)

        pre_y = net(x, edge_index.to(device), edge_weight.to(device))

        loss = loss_func(pre_y, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)  # 梯度裁剪，放backward和step直接
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        train_num += x.size(0)

        time_end = time.time()
        time_c = (time_end - time_start) * 100

        total_loss += loss.item()
        log_interval = int(len(trainX) / batch_size / 5)
        if (step + 1) % log_interval == 0 and (step + 1) > 0:
            cur_loss = total_loss / log_interval
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | '
                  'loss {:5.5f} | time {:8.2f}'.format(
                epoch, (step + 1), len(trainX) // batch_size, scheduler.get_last_lr()[0],
                cur_loss, time_c))
            total_loss = 0

    if (epoch + 1) % 5 == 0:
        print('-' * 89)
        print('end of epoch: {}, Loss:{:.7f}'.format(epoch + 1, loss.item()))
        print('-' * 89)
        train_loss_all.append(train_loss / train_num)

        # print_grad_norms(T_model)  ##

    # 验证阶段
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = net(data, edge_index.to(device), edge_weight.to(device))
            val_loss += loss_func(output, target).item()

    net.train()  # 将模型设置回train()模式

    print('Epoch: {} Validation Loss: {:.6f}'.format(epoch + 1, val_loss / (train_num / 8)))

    if val_loss / (train_num / 8) < best_val_loss:
        best_val_loss = val_loss / (train_num / 8)
        best_model = net
        print('best_Epoch: {} Validation Loss: {:.6f}'.format(epoch + 1, best_val_loss))

    scheduler.step()
#%%
torch.save(best_model.state_dict(), '1-Fed-traffic/models/FeSTGCN_{}.pt'.format(pre_len))
#%%
Fe_Net_dict = torch.load('1-Fed-traffic/models/FeSTGCN_1.pt')
Fe_Net = FeSTGCN_net(in_channels=seq_len, embed_channels=32,
                  out_channels=1,adj_matrix=adj_matrix.values,
                  adj_period=adj_period).to(device)
Fe_Net.load_state_dict(Fe_Net_dict)
Fe_Net.eval()
#%%
# Fe_Net=best_model.eval()
pred = Fe_Net(testX.float().to(device),edge_index,edge_weight)
Norm_pred = pred.data.cpu().numpy()
all_simu=Normalization.inverse_transform(Norm_pred[:,:,-1])
all_real=Normalization.inverse_transform(testY.data.numpy()[:,:,-1])
#%%  ALL STEPS

Metric=[]
for i in range(all_simu.shape[0]):
    MAE, RMSE, MAPE, R2 = evaluation(all_real[i, :], all_simu[i, :])
    Metric.append([MAE, RMSE, MAPE,R2])

M = np.mean(np.array(Metric), axis=0)
M_sec = pd.DataFrame(Metric)
print(M)

#%%
import matplotlib.cm as cm
import scipy.optimize as optimize
import matplotlib
from matplotlib import pyplot as plt
def f_1(x, A, B):
    return A * x + B

nx, ny=all_real.reshape(-1,1).squeeze(),all_simu.reshape(-1,1).squeeze()
nbins = 300
#STEP1结果
H, xedges, yedges = np.histogram2d(nx*60*15, ny*60*15, bins=nbins)
H = np.rot90(H)
H = np.flipud(H)
Hmasked = np.ma.masked_where(H==0,H)
#可视化绘制
fig, ax1 = plt.subplots()
#将颜色映射到 vmin~vmax 之间
norm = matplotlib.colors.Normalize(vmin=0, vmax=15)

im1 = ax1.pcolormesh(xedges, yedges, Hmasked, cmap=cm.get_cmap('jet'),norm=norm)
ax1.set_xlim(0, 800)
ax1.set_ylim(0, 800)
ax1.axline([0, 0], [1, 1],ls='--',c='k',lw=2)
A1, B1 = optimize.curve_fit(f_1, nx*60*15, ny*60*15)[0]
X1_ = np.arange(0, 800, 0.01)#30和75要对应x0的两个端点，0.01为步长
Y1_ = A1 * X1_ + B1
ax1.plot(X1_, Y1_, ls='-',c='r',lw=2)

plt.text(780, 20, '(f) FeSTGCN', fontsize=20,font='Times New Roman',
         verticalalignment='bottom', horizontalalignment='right')
plt.text(10, 750, 'Observed={:.2f}*Predicted+{:.2f}'.format(A1, B1), fontsize=20,font='Times New Roman',
         verticalalignment='top', horizontalalignment='left')
plt.text(10, 650, 'MAE=16.569', fontsize=20,font='Times New Roman',
         verticalalignment='top', horizontalalignment='left')
plt.text(10, 550, 'N=8487', fontsize=20,font='Times New Roman',
         verticalalignment='top', horizontalalignment='left')

# cbar = fig.colorbar(im1, ax=ax1)
# cbar.set_label('Count', fontproperties='Times New Roman', fontsize=18)  # 设置 colorbar 标签
#


plt.xticks(np.arange(0, 801, 200), fontproperties='Times New Roman', size=20)
plt.yticks(np.arange(0, 801, 200), fontproperties='Times New Roman', size=20)

plt.xlabel('Predicted',font='Times New Roman',fontsize=20)  # 替换为您的X轴标签
plt.ylabel('Observed',font='Times New Roman',fontsize=20)  # 替换为您的Y轴标签
plt.tight_layout()
plt.savefig(r"1-Fed-traffic/figs/FeSTGCN_scatter.svg", dpi=600)
plt.show()

#%%
weight_matrix = torch.mean(Fe_Net.embedding.weight.data,dim=0)[-8:].cpu()
weight_matrix_norm = (weight_matrix - weight_matrix.min()) / (weight_matrix.max() - weight_matrix.min())
# 绘制柱状图
plt.bar(range(len(weight_matrix_norm)), weight_matrix_norm)
plt.xlabel('Feature Index')
plt.ylabel('Weight')
plt.title('Feature Importance')
plt.show()
#%%




