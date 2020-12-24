import torch
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

# 选GPU
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_sizes, shuffle=False)

model = ModelDriver().to(device)

checkpoint = torch.load('../runs/main_1/model_files/checkpoint_22_0.000000.pth.tar')
# checkpoint = torch.load('filename.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

# 评估
test_set = tqdm(test_dataloader, leave=False, total=len(test_dataloader))

# model.eval()
total_test_loss = 0
test_count = 0
mae, rmse, mse, E1, lat_r2, lon_r2 = 0, 0, 0, 0, 0, 0
test_loss = []
for batch_idx, data in enumerate(test_set):
    if test_count == 3:
        break
    test_count += 1
    x, y = data
    x = x.to(device).type(torch.float32)
    y = y.to(device)
    output = model(x)
    loss = criterion(output, y)
    loss_aver = loss.item()
    test_set.set_postfix({
        'testloss': '{:.8f}'.format(loss_aver),
    })
    total_test_loss += loss_aver
    # 评估其他指标
    avg_mae, avg_mse, each_rmse, each_lat_r2, each_lon_r2, each_E1 = assessment.evaluate(output, y)
    mae += avg_mse
    mse += avg_mse
    rmse += each_rmse
    lat_r2 += each_lat_r2
    lon_r2 += each_lon_r2
    E1 += each_E1
avg_test_loss = total_test_loss / test_count
test_loss.append(avg_test_loss)
print('测试集 loss:{},测试集的mae:{},mse:{},rmse:{},APE:{},lat_r2:{},lon_r2:{}'.format(
    avg_test_loss, mae / test_count, mse / test_count, rmse / test_count, E1 / test_count, lat_r2 / test_count,
                   lon_r2 / test_count
))
