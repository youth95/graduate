# 参数 + 评价指标
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

n_features = 6
seq_length = 16
label_length = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def unnormal(data):
    '''
    :param data: 归一化后的数据
    :return: 反归一化
    LAT_MAX = 179.96099999999998 LAT_MIN = 100.0
    LON_MAX = 62.1 LON_MAX = 0.6
    '''
    LAT_MAX = 179.96099999999998
    LAT_MIN = 100.0
    LON_MAX = 62.1
    LON_MIN = 0.6
    data[:, :, 0:1] = data[:, :, 0:1] * (LAT_MAX - LAT_MIN) + LAT_MAX
    data[:, :, 1:] = data[:, :, 1:] * (LON_MAX - LON_MIN) + LON_MIN
    return data


# if __name__ == '__main__':
#     a = torch.randn((1, 8, 2))
#     print(a)
#     data1 = unnormal(a)
#     print(data1.shape)
#     print(data1)


def mae_mse_rmse(target, prediction):
    '''
    :param target: (batch,label_len,features)
    :param prediction:
    :return:
    '''
    # print('target shape', target.shape)
    n = len(target) * 2
    # 反归一化后的数据
    # un_target = unnormal(target)
    # un_pred = unnormal(prediction)

    lat_real = np.array(target[:, :, 0:1].squeeze(-1).cpu())
    lon_real = np.array(target[:, :, 1:].squeeze(-1).cpu())
    lat_pred = np.array(prediction[:, :, 0:1].squeeze(-1).cpu())
    lon_pred = np.array(prediction[:, :, 1:].squeeze(-1).cpu())
    lat_mae = mean_absolute_error(lat_pred, lat_real)
    lon_mae = mean_absolute_error(lon_pred, lon_real)
    mae = (lat_mae + lon_mae) / 2
    lat_mse = mean_squared_error(lat_pred, lat_real)
    lon_mse = mean_squared_error(lon_pred, lon_real)
    mse = (lat_mse + lon_mse) / 2
    lat_r2 = r2_score(lat_pred, lat_real)
    lon_r2 = r2_score(lon_pred, lon_real)
    rmse = mse ** 0.5
    return mae, mse, rmse, lat_r2, lon_r2

    # err1_mae = torch.abs(target[:,:, 0:1] - prediction[:,:, 0:1])
    # err2_mae = torch.abs(target[:, :,1:] - prediction[:, :,1:])
    # mae = (torch.sum(err1_mae).item() + torch.sum(err2_mae).item()) / n
    # mse = (torch.sum(err1_mae * err1_mae).item() + torch.sum(err2_mae * err2_mae).item()) / n
    # rmse = math.sqrt(mse)
    # # 经度的R2，维度的R2
    # R2_lat = R(target[:,:, 0:1], prediction[:, :,0:1])
    # R2_lon = R(target[:,:, 1:], prediction[:,:, 1:])
    # return mae, mse, rmse, R2_lat, R2_lon


def evaulate(model, test_dataloader, loss_func):
    model.eval()
    # it = iter(test_dataloader)
    total_count = 0
    total_loss = 0
    total_mae = 0
    total_mse = 0
    total_rmse = 0
    total_R2_lat = 0
    total_R2_lon = 0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            x, y = data
            x = x.to(device).view(-1, seq_length, n_features)
            y = y.to(device).view(-1, label_length, 2)
            output_test = model(x)
            output_test = output_test.permute(1, 0, 2)
            with torch.no_grad():
                loss = loss_func(output_test, y)
                mae, mse, rmse, R2_lat, R2_lon = mae_mse_rmse(output_test, y)
                E1, E2 = distance(output_test, y)  # 运算经纬度误差 返回 E1 E2
                total_count += 1
                total_loss += loss.item()
                total_mae += mae
                total_mse += mse
                total_rmse += rmse
                total_R2_lat += R2_lat
                total_R2_lon += R2_lon
    model.train()
    return total_loss / total_count, [
        total_mae / total_count,
        total_mse / total_count,
        total_rmse / total_count,
        total_R2_lat / total_count,
        total_R2_lon / total_count,
        E1,
        E2,
    ]


def to_radians(x, *y):
    return math.radians(x)


def distance(prediction, target):
    '''
    :param pred:  shape (n,2)
    :param real:  shape (n,2)
    :return: (n,1)
    '''
    R = 6357
    # 反归一化后的数据
    real = unnormal(target)
    pred = unnormal(prediction)
    real = real.reshape((-1, 2))
    pred = pred.reshape((-1, 2))
    latPred = pred[:, 0].cpu()
    latPred.map_(latPred, to_radians)
    lonPred = pred[:, 1].cpu()
    lonPred.map_(lonPred, to_radians)
    latReal = real[:, 0].cpu()
    latReal.map_(latReal, to_radians)
    lonReal = real[:, 1].cpu()
    lonReal.map_(lonReal, to_radians)
    # print(latPred)
    # print(lonPred)
    # print(latReal)
    # print(lonReal)
    E1 = 2 * R * torch.asin(
        torch.sqrt(
            torch.sin(
                torch.pow((latPred - latReal) / 2, 2)
            )
            + torch.cos(latReal) * torch.cos(latPred) *
            torch.sin(torch.pow((lonPred - lonReal) / 2, 2))
        )
    )
    E2 = 2 * R * torch.asin(
        torch.sqrt(
            torch.pow(torch.sin((latPred - latReal) / 2), 2) + torch.cos(latReal) * torch.cos(latPred) * torch.pow(
                (lonPred - lonReal) / 2, 2)
        )
    )

    return np.average(E1), np.average(E2)


def loss_show(test_loss, train_loss, epoches):
    x_epoch = np.arange(1, epoches + 1)
    plt.figure()
    plt.plot(x_epoch, train_loss, label='train_loss')
    plt.plot(x_epoch, test_loss, label='test_loss')
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def save_model(model):
    torch.save(model, "./model.pt")

#
# if __name__ == '__main__':
#     pred = torch.randn((32, 8, 2))
#     real = torch.randn((32, 8, 2))
#     mae, mse, rmse, R2_lat, R2_lon = mae_mse_rmse(pred, real)
#     print(mae, mse, rmse, R2_lat, R2_lon)
