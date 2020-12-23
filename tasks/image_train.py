import os,sys
os.chdir(os.path.dirname(__file__))
sys.path.append("..")
from torch import nn
from env.base import *
from models.layers import ModelDriver
from data_loaders.tmp_sat import fetch_data_set, MockDataSet
from torch.utils.data import DataLoader
from tqdm import tqdm
import assessment

lr = 1e-3
epoches = 100
weight_decays = 1e-3
batch_sizes =2

criterion = nn.MSELoss(size_average=False)

data_set = fetch_data_set([2010])
print(data_set)

train_dataloader = DataLoader(dataset=data_set, batch_size=batch_sizes, shuffle=True)
model = ModelDriver().to(device)


def init_weights(m):
    for name, parm in m.named_parameters():
        nn.init.uniform_(parm.data, -0.08, 0.08)


# model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decays)

if __name__ == "__main__":
    train_loss = []
    for epoch in range(epoches):
        model = model.train()
        total_loss = 0
        count = 0
        t = tqdm(train_dataloader, leave=False, total=len(train_dataloader))
        for batch_idx, data in enumerate(t):
            count += 1
            x, y = data
            x = x.to(device).type(torch.float32)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss_aver = loss.item()
            t.set_postfix({
                'trainloss': '{:.8f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
            loss.backward()
            optimizer.step()
            total_loss += loss_aver

            # lat,lon = assessment.graph_to_lon_lat(output)
            # print(lat,lon)
        # 每一个epoches的loss

        train_loss.append(total_loss / count)
        print('train_loss',train_loss)
        print('总共loss：',total_loss / count)

    assessment.loss_show(train_loss, epoches)
    torch.save(model,'48_24_MAR_ConvLSTM.pkl')
