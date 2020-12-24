import os, sys
from tensorboardX import SummaryWriter
import torchvision
import torch

torch.set_default_dtype(torch.float64)
os.chdir(os.path.dirname(__file__))
sys.path.append("..")
# 选GPU
import os

# tensorboard --logdir ./runs/main/logs
torch.cuda.set_device(3)
from tasks.utils import create_workspace, ssim
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
task_name, workspace, log_dir, model_files_dir = create_workspace("main_9")

image_save_head = 20

print("tensorboard --logdir ./runs/{}/logs".format(task_name))

tb = SummaryWriter(
    log_dir=log_dir
)
print('训练')
lr = 1e-3
epoches = 1000
weight_decays = 1e-3
batch_sizes = 2

# criterion = ssim
criterion = nn.MSELoss().to(device)

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

DEBUG = True
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
            x = x.to(device)
            input_images = x.reshape(-1, 1, 256, 256)
            if batch_idx == 1 and DEBUG:
                tb.add_image("input_imgs",
                             torchvision.utils.make_grid(input_images,
                                                         normalize=True, nrow=12, ),
                             epoch * len(train_set) + batch_idx)
                tb.add_image("input_label_imgs",
                             torchvision.utils.make_grid(x[:, :, 5:, :, :].reshape(-1, 1, 256, 256),
                                                         normalize=True, nrow=8, ),
                             epoch * len(train_set) + batch_idx)
                tb.add_image("input_遥感_imgs",
                             torchvision.utils.make_grid(x[:, :, 4:5, :, :].reshape(-1, 1, 256, 256),
                                                         normalize=True, nrow=8, ),
                             epoch * len(train_set) + batch_idx)
                tb.add_image("input_遥感*label_imgs",
                             torchvision.utils.make_grid(
                                 torch.mul(x[:, :, 5:, :, :] + 1, x[:, :, 4:5, :, :]).reshape(-1, 1, 256, 256),
                                 normalize=True, nrow=8, ),
                             epoch * len(train_set) + batch_idx)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            output, y = output.reshape(-1, 256, 256), y.reshape(-1, 256, 256)

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
        tb.add_scalar('TrainLoss', total_train_loss, epoch)
        train_loss.append(avg_train_loss)
        # 评估
        test_set = tqdm(test_dataloader, leave=False, total=len(test_dataloader))
        model.eval()
        i = 0
        for batch_idx, data in enumerate(test_set):
            test_count += 1
            x, y = data
            x = x.to(device).type(torch.float32)
            y = y.to(device)
            output = model(x)
            output, y = output.reshape(-1, 256, 256), y.reshape(-1, 256, 256)
            loss = criterion(output, y)
            loss_aver = loss.item()
            test_set.set_postfix({
                'testloss': '{:.8f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
            total_test_loss += loss_aver
            if i < image_save_head:
                pred = output.reshape(-1, 1, 256, 256)
                real = y.reshape(-1, 1, 256, 256)
                tb.add_image("Real_imgs", torchvision.utils.make_grid(real, normalize=True, nrow=8, ),
                             epoch * len(test_set) + batch_idx)
                tb.add_image("Pred_imgs", torchvision.utils.make_grid(pred, normalize=True, nrow=8, ),
                             epoch * len(test_set) + batch_idx)
                i += 1

        avg_test_loss = total_test_loss / test_count
        tb.add_scalar('TestLoss', total_test_loss, epoch)
        test_loss.append(avg_test_loss)
        print('Epoch：{},训练集loss:{},测试集loss:{}'.format(epoch, avg_train_loss, avg_test_loss))

        model_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        # order = np.random.choice(range(len(real)))
        # sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=order)
        # real = random.sample(real, 64)
        # pred = random.sample(pred, 64)

        early_stopping(total_test_loss, model_dict, epoch, model_files_dir, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    assessment.loss_show(train_loss, test_loss, epoches)
    # torch.save(model, '48_24_MAR_ConvLSTM.pkl')
