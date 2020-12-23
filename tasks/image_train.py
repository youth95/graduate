import os, sys

os.chdir(os.path.dirname(__file__))
sys.path.append("..")
from torch import nn
from env.base import *
from models.layers import ModelDriver
from data_loaders.tmp_sat import fetch_data_set, MockDataSet
from torch.utils.data import DataLoader
from tqdm import tqdm
import tasks.assessment as assessment
import utils.evaluate_function as ef

lr = 1e-3
epoches = 100
weight_decays = 1e-3
batch_sizes = 2

criterion = nn.MSELoss()

data_set = fetch_data_set([2010])
all_length = len(data_set)
train_size = int(all_length * 0.8)
test_size = all_length - train_size
train_data, test_data = torch.utils.data.random_split(data_set, [train_size, test_size])

train_dataloader = DataLoader(dataset=train_data, batch_size=batch_sizes, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_sizes, shuffle=True)

model = ModelDriver().to(device)


def init_weights(m):
    for name, parm in m.named_parameters():
        nn.init.uniform_(parm.data, -0.08, 0.08)


# model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decays)

if __name__ == "__main__":
    train_loss, test_loss = [], []
    for epoch in range(epoches):
        model = model.train()
        total_train_loss, total_test_loss = 0, 0
        train_count, test_count = 0, 0
        mae, mse, rmse, lat_r2, lon_r2, E1 = 0, 0, 0, 0, 0, 0
        train_set = tqdm(train_dataloader, leave=False, total=len(train_dataloader))
        model.train()
        for batch_idx, data in enumerate(train_set):
            train_count += 1
            x, y = data
            x = x.to(device).type(torch.float32)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss_aver = loss.item()
            train_set.set_postfix({
                'trainloss': '{:.8f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
            loss.backward()
            optimizer.step()
            total_train_loss += loss_aver
        avg_train_loss = total_train_loss / train_count
        train_loss.append(avg_train_loss)
        # 评估
        test_set = tqdm(test_dataloader, leave=False, total=len(test_dataloader))

        model.eval()
        for batch_idx, data in enumerate(test_set):
            test_count += 1
            x, y = data
            x = x.to(device).type(torch.float32)
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            loss_aver = loss.item()
            test_set.set_postfix({
                'testloss': '{:.8f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
            total_test_loss += loss_aver
            # 评估其他指标
        #     avg_mae, avg_mse, each_rmse, each_lat_r2, each_lon_r2, each_E1 = assessment.evaluate(output, y)
        #     mae += avg_mse
        #     mse += avg_mse
        #     rmse += each_rmse
        #     lat_r2 += each_lat_r2
        #     lon_r2 += each_lon_r2
        #     E1 += each_E1
        avg_test_loss = total_test_loss / test_count
        test_loss.append(avg_test_loss)
        # print('Epoch：{},训练集loss:{},测试集loss:{}\t测试集的mae:{},mse:{},rmse:{},APE:{},lat_r2:{},lon_r2:{}'.format(epoch,
        #                                                                                                    avg_train_loss,
        #                                                                                                    avg_test_loss,
        #                                                                                                    mae / test_count,
        #                                                                                                    mse / test_count,
        #                                                                                                    rmse / test_count,
        #                                                                                                    E1 / test_count,
        #                                                                                                    lat_r2 / test_count,
        print('Epoch：{},训练集loss:{},测试集loss:{}'.format(epoch, avg_train_loss, avg_test_loss))
    assessment.loss_show(train_loss, test_loss, epoches)
    torch.save(model, '48_24_MAR_ConvLSTM.pkl')
