import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if torch.cuda.is_available():
#     torch.cuda.set_device(0)
    # torch.cuda.set_device(3)
print("device is:", device)
