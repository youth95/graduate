'''
得到 测试集的经纬度进行评估
'''
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import torch
import matplotlib.pyplot as plt
import numpy as np

width = 256
height = 256

max_lon_diff = 179.96099999999998
min_lon_diff = 100.0
max_lat_diff = 62.1
min_lat_diff = 0.6


# 密度图转经纬度
def graph_to_lon_lat(graphs):
    """
    标签图转经纬度
    :param graphs: shape n,width,height
    :return:
    """
    graphs = graphs.reshape((-1, 256, 256))
    result = []
    for g in graphs:
        x, y = find(g)
        lat = x * (max_lat_diff - min_lat_diff) / width + min_lat_diff
        lon = y * (max_lon_diff - min_lon_diff) / height + min_lon_diff
        result.append([lat, lon])
    return np.array(torch.tensor(result)[:, 0:1]), np.array(torch.tensor(result)[:, 1:])


def find(g):
    max_val = torch.max(g)
    # print(max_val)
    for i in range(width):
        for j in range(height):
            if g[i][j] == max_val:
                return i, j


def evaluate(pred_map, real_map):
    pred_lat, pred_lon = graph_to_lon_lat(pred_map)
    real_lat, real_lon = graph_to_lon_lat(real_map)
    mae_lat = mean_absolute_error(pred_lat, real_lat)
    mae_lon = mean_absolute_error(pred_lon, real_lon)
    avg_mae = (mae_lat + mae_lon) / 2
    lat_mse = mean_squared_error(pred_lat, real_lat)
    lon_mse = mean_squared_error(pred_lon, real_lon)
    avg_mse = (lat_mse + lon_mse) / 2
    lat_r2 = r2_score(pred_lat, real_lat)
    lon_r2 = r2_score(pred_lon, real_lon)
    rmse = avg_mse ** 0.5
    E1 = distance(pred_lat, real_lat, pred_lon, real_lon)
    return avg_mae, avg_mse, rmse, lat_r2, lon_r2, E1


def to_radians(x, *y):
    return math.radians(x)


def distance(pred_lat, real_lat, pred_lon, real_lon):
    '''
    :param pred:  shape (n,2)
    :param real:  shape (n,2)
    :return: (n,1)
    '''
    R = 6357
    pred_lat, pred_lon, real_lat, real_lon = torch.tensor(pred_lat), torch.tensor(pred_lon), torch.tensor(
        real_lat), torch.tensor(real_lon)
    pred_lat.map_(pred_lat, to_radians)
    pred_lon.map_(pred_lon, to_radians)
    real_lat.map_(real_lat, to_radians)
    real_lon.map_(real_lon, to_radians)

    E1 = 2 * R * torch.asin(
        torch.sqrt(
            torch.sin(
                torch.pow((pred_lat - real_lat) / 2, 2)
            )
            + torch.cos(real_lat) * torch.cos(pred_lat) *
            torch.sin(torch.pow((pred_lon - real_lon) / 2, 2))
        )
    )
    E2 = 2 * R * torch.asin(
        torch.sqrt(
            torch.pow(torch.sin((pred_lat - real_lat) / 2), 2) + torch.cos(real_lat) * torch.cos(pred_lat) * torch.pow(
                (pred_lon - real_lon) / 2, 2)
        )
    )

    return torch.mean(E1)


# 画图loss曲线

def loss_show(train_loss, test_loss, epoches):
    x_epoch = np.arange(1, epoches + 1)
    plt.figure()
    plt.plot(x_epoch, train_loss, label='train_loss')
    plt.plot(x_epoch, test_loss, label='test_loss')
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    # plt.show()


if __name__ == '__main__':
    pred = torch.randn((2, 8, 256, 256))
    rea = torch.randn((2, 8, 256, 256))
    avg_mae, avg_mse, rmse, lat_r2, lon_r2, E1 = evaluate(pred, rea)
    print(avg_mae, avg_mse, rmse, lat_r2, lon_r2, E1)

# train_loss = [4.2976985, 1.59740962, 0.79139572, 0.79467306, 0.25570447, 9.18250898, 0.48530427,
#               0.30929214, 0.23451553, 0.18762622, 0.15461622, 0.12929283, 0.10989252, 0.09445145, 0.08273779,
#               0.07326682,
#               0.06594757, 0.05991374, 0.05598096, 1.79995963,
#               0.10041473,
#               0.0690101,
#               0.05612091,
#               0.04934649,
#               0.04563311,
#               0.05092497,
#               0.22149864,
#               0.13869008,
#               0.10182207,
#               0.07943768,
#               0.06396435,
#               0.05272042,
#               0.04473741,
#               0.03908781,
#               0.03703286,
#               0.05875691,
#               0.08732618,
#               0.6317684,
#               0.42508282,
#               0.19495695,
#               0.4668717,
#               0.46163893,
#               0.52295235,
#               0.47581402,
#               0.48344642,
#               0.52064368,
#               0.5099504,
#               0.50658289,
#               0.47282691,
#               0.48875708,
#               0.48618648,
#               0.47920963,
#               0.49014629,
#               0.46643559,
#               0.4800939,
#               0.47422153,
#               0.47053702,
#               0.49985517,
#               0.42944007,
#               0.43151148,
#               0.44345426,
#               0.41884436,
#               0.43365092,
#               0.49958709,
#               0.39675897,
#               0.41454186,
#               0.38744242,
#               0.38758743,
#               0.42794783,
#               0.34259683,
#               0.36644438,
#               0.35685789,
#               0.34819014,
#               0.34700917,
#               0.35563347,
#               0.42589085,
#               0.28289142,
#               0.37144885,
#               0.27490163,
#               0.12338977,
#               0.07698718,
#               0.06089537,
#               0.05133444,
#               0.04500092,
#               0.04068721,
#               0.03738235,
#               0.03519677,
#               0.03331551,
#               0.03201696,
#               0.03085543,
#               0.03004781,
#               0.02914126,
#               0.02853632,
#               0.02798069,
#               0.0273726,
#               0.02680045,
#               ]
# test_loss = [
#     2.41579645,
#     # 0.92802076,
#     0.44334536,
#     0.31284096,
#     0.21024751,
#     0.6916728,
#     0.35977958,
#     0.26234542,
#     0.2052125,
#     0.16761718,
#     0.13924727,
#     0.11746582,
#     0.10010065,
#     0.0868937,
#     0.07714164,
#     0.06831487,
#     0.06183963,
#     0.05878211,
#     0.05938463,
#     0.13207226,
#     0.0782635,
#     0.06046309,
#     0.05139655,
#     0.04690887,
#     0.04349537,
#     0.1034946,
#     0.30574966,
#     0.16488677,
#     0.11566536,
#     0.08828234,
#     0.06991858,
#     0.05719898,
#     0.04773599,
#     0.04130394,
#     0.03686316,
#     0.0363824,
#     0.0984065,
#     0.03231205,
#     0.45234263,
#     0.04981387,
#     0.1702927,
#     0.19114958,
#     0.24137152,
#     0.304631,
#     0.08487011,
#     0.07886303,
#     0.19006683,
#     0.20345536,
#     0.19020309,
#     0.12838739,
#     0.09170115,
#     0.06054076,
#     0.09459231,
#     0.08977903,
#     0.0534599,
#     0.07129896,
#     0.08653197,
#     0.10287989,
#     0.1633069,
#     0.07041362,
#     0.09435953,
#     0.06949195,
#     0.10988318,
#     0.08779566,
#     0.35075454,
#     0.1466323,
#     0.12422249,
#     0.17801935,
#     0.08927072,
#     0.15251585,
#     0.18720223,
#     0.13112686,
#     0.05693876,
#     0.07649459,
#     0.07438448,
#     0.14500681,
#     0.18921926,
#     0.14733326,
#     0.04919539,
#     0.09689003,
#     0.17582556,
#     0.088659,
#     0.06596394,
#     0.05465387,
#     0.04744258,
#     0.04199417,
#     0.03813134,
#     0.03560421,
#     0.03389771,
#     0.03234743,
#     0.03135133,
#     0.03026777,
#     0.02915575,
#     0.02836207,
#     0.02791821,
#     0.02743312,
#     0.02657909,
#     0.02643727, ]
