import os, sys
from tensorboardX import SummaryWriter
import torchvision
import torch
os.chdir(os.path.dirname(__file__))
sys.path.append("..")
# 选GPU
import os
# tensorboard --logdir ./runs/main/logs
torch.cuda.set_device(3)
from tasks.utils import create_workspace
from torch import nn
from env.base import *
from models.layers import ModelDriver
from data_loaders.tmp_sat import fetch_data_set
from torch.utils.data import DataLoader
from tqdm import tqdm
from tasks.earlystopping import EarlyStopping
import numpy as np
import random
import tasks.assessment as assessment

# 创建工作环境 建议每次跑之前都新建一个工作环境
task_name, workspace, log_dir, model_files_dir = create_workspace("main_1")

print("tensorboard --logdir ./runs/{}/logs".format(task_name))

tb = SummaryWriter(
    log_dir=log_dir
)
print('训练')
lr = 1e-3
epoches = 1000
weight_decays = 1e-3
batch_sizes = 2

criterion = nn.MSELoss()

data_set = fetch_data_set([2010])
# data_set, _ = torch.utils.data.random_split(data_set, [2, len(data_set) - 2])  # 测试
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
early_stopping = EarlyStopping(patience=20, verbose=True)

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
        # pred_imgs = []
        # real_imgs = []
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
        #     pred_imgs.append(output)
        #     real_imgs.append(y)
        avg_test_loss = total_test_loss / test_count
        test_loss.append(avg_test_loss)
        print('Epoch：{},训练集loss:{},测试集loss:{}'.format(epoch, avg_train_loss, avg_test_loss))
        # real = torch.cat(real_imgs).view(-1, 1, 256, 256)
        # pred = torch.cat(pred_imgs).view(-1, 1, 256, 256)
        model_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        # order = np.random.choice(range(len(real)))
        # sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=order)
        # real = random.sample(real, 64)
        # pred = random.sample(pred, 64)
        # tb.add_image("Real_imgs", torchvision.utils.make_grid(real, normalize=True, nrow=8, ), epoch)
        # tb.add_image("Pred_imgs", torchvision.utils.make_grid(pred, normalize=True, nrow=8, ), epoch)
        early_stopping(avg_test_loss, model_dict, epoch, model_files_dir, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    assessment.loss_show(train_loss, test_loss, epoches)
    # torch.save(model, '48_24_MAR_ConvLSTM.pkl')
