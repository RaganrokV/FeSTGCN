# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import preprocessing
import scipy.sparse as sp
from torch_geometric_temporal import ChebConvAttention, TemporalConv
from torch_geometric_temporal.nn.recurrent.dcrnn import DConv
import torch
from torch import nn
from My_utils.evaluation_scheme import evaluation
from My_utils.preprocess_data import  divide_data
import torch.utils.data as Data
import time
from torch_geometric_temporal.nn.recurrent import GConvGRU,TGCN2,GCLSTM,A3TGCN2
import torch.nn.functional as F
#%%

class my_TGCN2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, out_size,batch_size):
        super(my_TGCN2, self).__init__()
        """:param"""

        self.T_Conv = TGCN2(in_channels,out_channels, batch_size,improved=True)
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=out_channels,
            num_layers=2,
            dropout=0.2,
            batch_first=True)

        self.linear1 = torch.nn.Linear(out_channels, 1)
        self.linear2 = nn.Linear(out_channels, 1)
        self.linear3 = nn.Linear(2, out_size)

    def forward(self, x, edge_index, edge_weight):
        temp, _ = self.gru(x)
        B, S, H = temp.size()
        outs = self.linear2(temp)
        gru_out = outs.reshape(B, S, -1)

        h = self.T_Conv(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear1(h)

        CAT=torch.cat((gru_out,h),dim=2)
        fusion = self.linear3(CAT)

        return fusion


class my_ASTGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, out_size,batch_size):
        super(my_ASTGCN, self).__init__()
        """:param"""
        self.periods=3

        self.AST = A3TGCN2(in_channels//self.periods,out_size,periods=self.periods, batch_size=batch_size)
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=out_channels,
            num_layers=2,
            dropout=0.2,
            batch_first=True)


        self.linear1 = torch.nn.Linear(out_size, 1)
        self.linear2 = nn.Linear(out_channels, 1)
        self.linear3 = nn.Linear(2, out_size)

    def forward(self, x, edge_index, edge_weight):
        temp, _ = self.gru(x)
        B, S, H = temp.size()
        outs = self.linear2(temp)
        gru_out = outs.reshape(B, S, -1)

        h = self.AST(x.reshape(x.shape[0],x.shape[1],8,3),edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear1(h)

        CAT = torch.cat((gru_out, h), dim=2)
        fusion = self.linear3(CAT)

        return fusion

class my_DCRNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, out_size):
        super(my_DCRNN, self).__init__()
        """:param"""

        self.Dconv = DConv(in_channels,out_size,K=3)
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=out_channels,
            num_layers=2,
            batch_first=True)

        self.linear1 = torch.nn.Linear(out_size, 1)
        self.linear2 = nn.Linear(out_channels, 1)
        self.linear3 = nn.Linear(2, out_size)

    def forward(self, x, edge_index, edge_weight):
        temp, _ = self.gru(x)
        B, S, H = temp.size()
        outs = self.linear2(temp)
        gru_out = outs.reshape(B, S, -1)

        h = self.Dconv(x,edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear1(h)

        CAT = torch.cat((gru_out, h), dim=2)
        fusion = self.linear3(CAT)

        return fusion

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len=24
pre_len=4
batch_size=64
train_size=3500 #80%
# set seed for reproductive 42 is the answer to the universe
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
#%%
data_csv = pd.read_csv(r'3-Unfixed sampling/periodic_sample.csv')
adj=pd.read_csv(r'3-Unfixed sampling//ADJ.csv')

# data_csv = pd.read_csv(r'periodic_sample.csv') #for debug
# adj=pd.read_csv(r'ADJ.csv')
adj_matrix=adj
Normalization = preprocessing.MinMaxScaler()
Norm_TS = Normalization.fit_transform(data_csv.values)
#%%
"""divide data"""
trainX, trainY, testX, testY = divide_data(data=Norm_TS, train_size=train_size,
                                           seq_len=seq_len, pre_len=pre_len)
trainX, trainY = trainX.transpose(1, 2).float(), trainY.transpose(1, 2).float()
testX, testY = testX.transpose(1, 2).float(), testY.transpose(1, 2).float()


adj = sp.coo_matrix(adj)
values = adj.data
indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
edge_index = torch.LongTensor(indices)  # PyG框架需要的coo形式
edge_index=torch.LongTensor(edge_index).to(device)


edge_weight=np.ones(edge_index.shape[1])
edge_weight=torch.FloatTensor(edge_weight).to(device)

train_dataset = Data.TensorDataset(trainX, trainY)
test_dataset = Data.TensorDataset(testX, testY)

train_loader = Data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size, shuffle=False)
test_loader = Data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size, shuffle=False)
#%%TGCN
# net = my_TGCN2(in_channels=seq_len, out_channels=64,out_size=pre_len,batch_size=batch_size).to(device)
# net = my_ASTGCN(in_channels=seq_len, out_channels=100,
#                out_size=pre_len,batch_size=batch_size).to(device)
net = my_DCRNN(in_channels=seq_len, out_channels=64,out_size=pre_len).to(device)
optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.98)
loss_func = nn.MSELoss()



best_val_loss = float("inf")
best_model = None
#           train
train_loss_all = []
net.train()  # Turn on the train mode
total_loss = 0.

for epoch in range(500):

    cost = 0
    train_loss = 0
    train_num = 0
    for step, (x, y) in enumerate(train_loader):

        time_start = time.time()

        optimizer.zero_grad()


        x, y = x.to(device), y.to(device)  #batch train


        pre_y = net(x,edge_index.to(device),edge_weight.to(device))

        loss = loss_func(pre_y, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)  # 梯度裁剪，放backward和step直接，小模型可以不考虑用于缓解梯度爆炸
        optimizer.step()
        train_loss += loss.item() * x.size(0)
        train_num += x.size(0)

        time_end = time.time()
        time_c = (time_end - time_start)*10

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

    if train_loss < best_val_loss:
        best_val_loss = train_loss
        best_model = net

    scheduler.step()

#%%
"""forecasting"""
best_model.eval()  # 转换成测试模式
pred = best_model(testX.float().to(device),edge_index,edge_weight)
Norm_pred = pred.data.cpu().numpy()
"""evaluation"""
all_simu=Normalization.inverse_transform(Norm_pred[:,:,-1])
all_real=Normalization.inverse_transform(testY.data.numpy()[:,:,-1])
Metric=[]
for i in range(all_simu.shape[0]):
    MAE, RMSE, MAPE, R2 = evaluation(all_real[i, :], all_simu[i, :])
    Metric.append([MAE, RMSE, MAPE, R2])

M = np.mean(np.array(Metric), axis=0)
M_sec = pd.DataFrame(Metric)

print(M)
