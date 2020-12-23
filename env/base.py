import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if torch.cuda.is_available():
#     torch.cuda.set_device(0)
# torch.cuda.set_device(3)
print("device is:", device)

# 学校服务器的数据地址
data_root = '/mnt/data1/ppdata/tmp_sat/{}'
# 家里服务器的数据地址
# data_root = "/srv/datasets/tmp_sat/{}"
