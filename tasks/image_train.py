from torch import nn
from env.base import *
from models.layers import Seq2Seq
from data_loaders.tmp_sat import fetch_data_set
from torch.utils.data import DataLoader
from tqdm import tqdm

lr = 1e-3
epoches = 100
weight_decays = 1e-3
batch_sizes = 1

criterion = nn.MSELoss()
data_set = fetch_data_set([2010])
train_dataloader = DataLoader(dataset=data_set, batch_size=batch_sizes, shuffle=True)
model = Seq2Seq().to(device)


def init_weights(m):
    for name, parm in m.named_parameters():
        nn.init.uniform_(parm.data, -0.08, 0.08)


model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decays)

if __name__ == "__main__":
    for epoch in range(epoches):
        model = model.train()
        total_loss = 0
        t = tqdm(train_dataloader, leave=False, total=len(train_dataloader))
        for batch_idx, data in enumerate(t):
            x, y = data
            x = x.to(device)
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
