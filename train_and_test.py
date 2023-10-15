import numpy as np
import torch
import time



######## hyper parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len=288
pre_len=144
batch_size=144
train_size=10000
# set seed for reproductive 42 is the answer to the universe
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


trainX_len=train_size-seq_len-pre_len

def train_net(net,epoch,train_loader,optimizer,loss_func,scheduler):

    best_val_loss = float("inf")
    best_model = None

    train_loss_all = []
    # net.train()  # Turn on the train mode
    net.train()  # Turn on the train mode
    total_loss = 0.

    for epoch in range(epoch):
        train_loss = 0
        train_num = 0
        for step, (x, y, PI) in enumerate(train_loader):

            time_start = time.time()

            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            data = torch.cat((PI.to(device), x), dim=1)
            pre_y = net(data.permute(0, 2, 1))

            loss = loss_func(y, pre_y.unsqueeze(1))
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

    return best_model

def test_net(Net,test_PI,testX):

    best_model = Net.eval()  # 转换成测试模式
    data_test = torch.cat((test_PI.float().to(device), testX.float().to(device)), dim=1)
    pred = best_model(data_test.permute(0, 2, 1).to(device))  #

    Norm_pred = pred.squeeze().data.cpu().numpy()

    return Norm_pred