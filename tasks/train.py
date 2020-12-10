import warnings
import os, sys

os.chdir(os.path.dirname(__file__))
sys.path.append("..")
warnings.filterwarnings('ignore')
from models.Seq2Seq_att import DataSet_seq, Seq2Seq
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import time
import torch
import utils.evaluate_function as ef

start = time.time()

"""
可调整参数
"""

lr = 1e-4
epoches = 1000
criterion = nn.MSELoss()
weight_decays = 1e-3
batch_sizes = 64
in_n_features = 6
out_n_features = 2

in_seq_len = 16
out_seq_len = 8
num_layers = 3
embedding_dim = 64

"""
数据处理
"""
data = pd.read_csv('../data/48_predicted_24_supervised_data.csv').fillna(0.0)
data_new = data.iloc[16:, 1:]
data_train, data_test = train_test_split(data_new, test_size=0.2, shuffle=True)
train_data = DataSet_seq(data_train)
test_data = DataSet_seq(data_test)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_sizes, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_sizes, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Seq2Seq(
    in_seq_len=in_seq_len,
    in_n_features=in_n_features,
    out_n_features=out_n_features,
    num_layers=num_layers,
    embedding_dim=embedding_dim,
    out_seq_len=out_seq_len
).to(device)


def init_weights(m):
    for name, parm in m.named_parameters():
        #         print(name,parm)
        nn.init.uniform_(parm.data, -0.07, 0.07)


model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decays)

if __name__ == '__main__':
    test_loss = []
    train_loss = []
    for epoch in range(epoches):
        model = model.train()
        total_loss = 0
        t = tqdm(train_dataloader, leave=False, total=len(train_dataloader))
        for batch_idx, data in enumerate(t):
            x, y = data
            x = x.to(device).view(-1, in_seq_len, in_n_features)
            y = y.to(device).view(-1, out_seq_len, out_n_features)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.permute(1, 0, 2), y)
            loss_aver = loss.item()
            t.set_postfix({
                'trainloss': '{:.8f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().item()
        test_loss_one, eval_list = ef.evaulate(model, test_dataloader, criterion)
        print('epoch:{},tr_l:{},te_l:{},test_mae：{}, test_mse:{}, test_rmse:{},R2_lat:{},R2_lon:{},E1:{},E2:{}'.format(
            epoch + 1,
            total_loss / len(train_dataloader),
            test_loss_one,
            eval_list[0],
            eval_list[1],
            eval_list[2],
            eval_list[3],
            eval_list[4],
            eval_list[5],
            eval_list[6],
        ))
        test_loss.append(test_loss_one)
        train_loss.append(total_loss / len(train_dataloader))
    # 可视化
    ef.loss_show(test_loss, train_loss, epoches)
    ef.save_model(model)
end = time.time()
print('总时间time：', (end - start))
