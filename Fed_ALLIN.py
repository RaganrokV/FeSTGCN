import pandas as pd
import numpy as np
import torch
from torch import nn
from My_utils.evaluation_scheme import evaluation
from My_utils.preprocess_data import divide_data
from data_preprocessing import normalized,get_dataloader
from Fed_TCN_Net import decomposed_block,weight_loss,TemporalConvNet
from train_and_test import train_net,test_net
from Fed_TCN_Net import Frequency_enhanced
import warnings
import torch.utils.data as Data
from torch.nn.utils import weight_norm
import time
from sklearn import preprocessing
warnings.filterwarnings("ignore")
#%%  hyper parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len=360
pre_len=180  #7:00:22:00
batch_size=144
train_size=12000
# set seed for reproductive 42 is the answer to the universe
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
#%%  prepare data
data_csv = pd.read_csv(r'Fed-traffic/PeMS.csv')
Normalization = preprocessing.MinMaxScaler()
Norm_TS = Normalization.fit_transform(data_csv.iloc[: ,:23])
#%%
data_csv = pd.read_csv(r'Fed-traffic/PeMS.csv')
# data_csv = pd.read_csv(r'PeMS.csv')#for debug
TRX,TRY,TEX,TEY,TRP,TEP=[],[],[],[],[],[]
for i in range(23):
    fre_info = decomposed_block(data_csv.iloc[:8640, i].values)
    Norm_TS, Normalization = normalized(data_csv.iloc[:, i].values)
    # frequency enhanced block and put into loader#
    trainX, trainY, testX, testY = divide_data(data=Norm_TS, train_size=train_size,
                                               seq_len=seq_len, pre_len=pre_len)
    trainX, trainY = trainX.transpose(1, 2).float(), trainY.transpose(1, 2).float()
    testX, testY = testX.transpose(1, 2).float(), testY.transpose(1, 2).float()

    TRX.append(trainX)
    TRY.append(trainY)
    TEX.append(testX)
    TEY.append(testY)

    #   pattern info
    train_patterns = []
    train_res = []
    for s in range(len(trainX)):
        train_P, train_R = Frequency_enhanced(trainX[s].unsqueeze(0), fre_info[:7])
        train_patterns.append(np.array(train_P))
        train_res.append(np.array(train_R))

    test_patterns = []
    test_res = []
    for d in range(len(testX)):
        test_P, test_R = Frequency_enhanced(testX[d].unsqueeze(0), fre_info[:7])
        test_patterns.append(np.array(test_P))
        test_res.append(np.array(test_R))

    # unify shape
    train_pattern = torch.tensor(np.array(train_patterns)).float()
    train_res = torch.tensor(np.array(train_res)).squeeze(1).float()

    test_pattern = torch.tensor(np.array(test_patterns)).float()
    test_res = torch.tensor(np.array(test_res)).squeeze(1).float()

    train_PI = torch.cat((train_pattern, train_res), dim=1)
    test_PI = torch.cat((test_pattern, test_res), dim=1)
    #  nomalize data
    train_PI = nn.functional.normalize(train_PI, dim=2)  # dim=2，是对第三个维度，也就是每一行操作
    test_PI = nn.functional.normalize(test_PI, dim=2)

    TRP.append(train_PI)
    TEP.append(test_PI)
    print(i)


#%%
trainX=torch.stack(TRX).squeeze().permute(1,0,2)
trainY=torch.stack(TRY).squeeze().permute(1,0,2)
testX=torch.stack(TEX).squeeze().permute(1,0,2)
testY=torch.stack(TEY).squeeze().permute(1,0,2)

train_PI=torch.stack(TRP).permute(1, 0, 2,3)
train_PI=train_PI.resize(train_PI.shape[0],
                                 train_PI.shape[1]*train_PI.shape[2],
                                 train_PI.shape[3])
test_PI=torch.stack(TEP).permute(1, 0, 2,3)
test_PI=test_PI.resize(test_PI.shape[0],
                                 test_PI.shape[1]*test_PI.shape[2],
                                 test_PI.shape[3])

# np.savez('Fed-traffic/my_data.npz', trainX,trainY,testX,testY,train_PI,test_PI)
#%%
data=np.load('Fed-traffic/my_data.npy.npz')
trainX,trainY,testX,testY,train_PI,test_PI=\
    data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3'],data['arr_4'],data['arr_5']
#%%
trainX,trainY,testX,testY,train_PI,test_PI=torch.tensor(trainX),torch.tensor(trainY),\
                                           torch.tensor(testX),torch.tensor(testY),\
                                           torch.tensor(train_PI),torch.tensor(test_PI)
train_PI=train_PI.permute(1, 0, 2,3)
train_PI=train_PI.resize(train_PI.shape[0],
                                 train_PI.shape[1]*train_PI.shape[2],
                                 train_PI.shape[3])
test_PI=test_PI.permute(1, 0, 2,3)
test_PI=test_PI.resize(test_PI.shape[0],
                                 test_PI.shape[1]*test_PI.shape[2],
                                 test_PI.shape[3])

train_dataset = Data.TensorDataset(trainX, trainY, train_PI)
test_dataset = Data.TensorDataset(testX, testY, test_PI)

# put into loader
train_loader = Data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size, shuffle=False)
test_loader = Data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size, shuffle=False)
#%%################    TCN_block  ##############
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


        outs = self.network(x)
        # FC = nn.Linear(outs.shape[2], outs.shape[2]/9).to(device)
        # Fusion = FC(outs)
        temp=outs.resize(outs.shape[0],outs.shape[1],23,9)
        final_out=torch.mean(temp,dim=3)
        return final_out

def weight_loss(y, pre_y):

    func = nn.MSELoss(reduce = False)  #return vector
    weight=torch.tensor(np.linspace(1, 0.0001, y.size(2))).float().to(device)
    b_weight=weight.repeat(y.size(0),1).unsqueeze(1)
    loss=func(y, pre_y)
    adj_loss=torch.mean(b_weight*loss)

    return adj_loss
#%% Fed-TCN

Fed_TCN = TemporalConvNet(num_inputs=seq_len, num_channels=[32,pre_len], kernel_size=2, dropout=0.3).to(device)
optimizer = torch.optim.RMSprop(Fed_TCN .parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.99)
# loss_func = nn.MSELoss()
loss_func=weight_loss
#%% Train and test net
net=Fed_TCN
best_val_loss = float("inf")
best_model = None

train_loss_all = []
# net.train()  # Turn on the train mode
net.train()  # Turn on the train mode
total_loss = 0.
trainX_len=train_size-seq_len-pre_len
for epoch in range(100):
    train_loss = 0
    train_num = 0
    for step, (x, y, PI) in enumerate(train_loader):

        time_start = time.time()

        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        data = torch.cat((PI.to(device), x), dim=1)
        pre_y = net(data.permute(0, 2, 1))

        loss = loss_func(y, pre_y.permute(0, 2, 1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.05)
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        train_num += x.size(0)

        time_end = time.time()
        time_c = time_end - time_start

        total_loss += loss.item()
        log_interval = int(trainX_len / batch_size / 5)
        if (step + 1) % log_interval == 0 and (step + 1) > 0:
            cur_loss = total_loss / log_interval
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | '
                  'loss {:5.5f} | time {:8.2f}'.format(
                epoch, (step + 1), trainX_len // batch_size, scheduler.get_lr()[0],
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
best_model.eval()  # 转换成测试模式
data_test = torch.cat((test_PI.float().to(device), testX.float().to(device)), dim=1)
pred = best_model(data_test.permute(0, 2, 1).to(device))  #

Norm_pred = pred.squeeze().permute(0, 2, 1).data.cpu().numpy()
#%% evaluate
data_csv = pd.read_csv(r'Fed-traffic/PeMS.csv')
Normalization = preprocessing.MinMaxScaler()
Norm_TS = Normalization.fit_transform(data_csv.iloc[: ,:23])
all_simu,all_real=[],[]
for s in range(pre_len):
    all_simu.append(Normalization.inverse_transform(Norm_pred[:, :, s]))
    all_real.append(Normalization.inverse_transform(testY.data.numpy()[:, :, s]))


#%%
all_simu=np.array(all_simu)
all_real=np.array(all_real)

#%%
All_metrics=[]
for s in range(23):
    all_simu = Normalization.inverse_transform(Norm_pred[:, :, s])
    all_real = Normalization.inverse_transform(testY[:, :, s].data.numpy())
    Metric = []
    for i in range(pre_len):
        MAE, RMSE, MAPE, R2 = evaluation(all_real[i, :].reshape(-1, 1), all_simu[i, :].reshape(-1, 1))
        Metric.append([MAE, RMSE, MAPE, R2])

    M = np.mean(np.array(Metric), axis=0)
    Deg_M = pd.DataFrame(Metric)

    All_metrics.append(M)

#%%
A=np.array(All_metrics)
