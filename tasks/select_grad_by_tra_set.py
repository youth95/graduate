import pandas as pd
from datetime import datetime
import pickle
import os
import netCDF4
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn import preprocessing
import warnings

warnings.filterwarnings('ignore')

min_max_scaler = preprocessing.MinMaxScaler()

min_lat_diff = 0.6
max_lat_diff = 62.1
min_lon_diff = 100.0
max_lon_diff = 179.96099999999998


def normalization(x):
    """"
    归一化到区间{0,1]
    返回副本
    """
    x = x.squeeze()
    t = torch.isnan(x)
    _max = x.max()
    _min = x.min()
    _range = _max - _min
    return (x - torch.min(x)) / _range


def lon_lat_to_graph(lat, lon, w, h, sigma=8):
    """
    经纬度转密度图
    :param y: shape n,2
    :return: n,w,h
    """
    g = torch.zeros((w, h), dtype=torch.float)
    x, y = (lat - min_lat_diff) / (max_lat_diff - min_lat_diff) * w, (lon - min_lon_diff) / (
            max_lon_diff - min_lon_diff) * h
    g[int(x)][int(y)] = torch.tensor(1., dtype=torch.float)
    g = gaussian_filter(g, sigma=sigma)
    return g


def save(data, filename):
    # 以二进制写模式打开目标文件
    f = open(filename, 'wb')
    # 将变量存储到目标文件中区
    pickle.dump(data, f)
    # 关闭文件
    f.close()


def read(filename):
    # 以二进制读模式打开目标文件
    f = open(filename, 'rb')
    # 将文件中的变量加载到当前工作区
    data = pickle.load(f)
    f.close()
    return data


def read_irwin_cdr_image(full_path):
    with netCDF4.Dataset(full_path) as nc_obj:
        data = nc_obj.variables['irwin_cdr'][:].data
        img = torch.tensor(data).view(2000, 5143)
        # torchvision.utils.save_image(img, save_path, nrow=1, normalize=True, )
        # print('avg:', np.average(img.numpy()))
        # print('min:', np.min(img.numpy()))
        # print('max:', np.max(img.numpy()))
        # print("lat", nc_obj.variables['lat'][:].data)
        # print("lat", nc_obj.variables['lat'][:].data.shape)
        # print("lon", nc_obj.variables['lon'][:].data)
        # print("lon", nc_obj.variables['lon'][:].data.shape)
    return img


def fill_nan(label, value):
    nan_count = pd.DataFrame(value.reshape((256, 256))).isna().sum().sum()
    if nan_count != 0:
        print("张量:{} 发现{} 个 nan !!!!!! 已替换".format(label, nan_count))
        value[np.isnan(value)] = 0


def read_met_sat(full_path, idx, size=256):
    with netCDF4.Dataset(full_path) as nc_obj:
        def hand_item(label):
            value = nc_obj.variables[label][:]
            value = value[idx:idx + 1, :, :]
            value = resize(value.reshape((261, 321)), size)
            value = min_max_scaler.fit_transform(value.reshape(size, size)).reshape((1, size, size))
            fill_nan(label, value)
            fill_nan(label, value)
            return value

        sst = hand_item("sst")
        sp = hand_item("sp")
        u10 = hand_item("u10")
        v10 = hand_item("v10")
        data = torch.tensor(np.vstack([sst, sp, u10, v10]))
        # data = torch.tensor(np.nan_to_num(data))
        # data = normalization(data)
        # data = min_max_scaler.fit_transform(data)
    return data


def crop(
        img,
        min_lon,
        min_lat,
        max_lon,
        max_lat,
):
    u = 0.07
    nimg = img[int((min_lat + 70) / u):int((max_lat + 70) / u), int((min_lon + 180) / u):int((max_lon + 180) / u), ]
    return nimg


def scale(img, x_step, y_step):
    return img[::x_step, ::y_step]


def resize(img, size=64):
    transform1 = transforms.Compose([
        transforms.Scale(size),
        transforms.CenterCrop((size, size)),
        transforms.ToTensor()
    ])
    return transform1(Image.fromarray(img, mode="F"))


def find(year, save_dir="/srv/datasets/tmp_sat"):
    print("fetch", year)
    df = pd.read_csv('../data/tra_sat.csv', usecols=["ISO_TIME", "LAT", "LON", "SID"])
    base_path = "/srv/datasets/grid_sat/%d" % year
    met_sat_base_path = "/srv/datasets/met_sat/%d" % year
    save_dir = os.path.join(save_dir, str(year))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for row in df.values:
        _time = row[1]
        t = datetime.strptime(_time, '%Y-%m-%d %H:%M:%S')
        if t.year == year:
            file_name = "GRIDSAT-B1.%d.%02d.%02d.%02d.v02r01.nc" % (t.year, t.month, t.day, t.hour)
            full_path = os.path.join(base_path, file_name)
            if not os.path.exists(full_path):
                print("no match: {}".format(full_path))
                continue
            else:
                item = dict()
                item["time"] = _time
                item["lat"] = row[3]
                item["lon"] = row[2]
                item["sid"] = row[0]
                item["grid_sat_file"] = full_path
                img = read_irwin_cdr_image(full_path)
                img = crop(img,
                           min_lon=100,
                           min_lat=0,
                           max_lon=180,
                           max_lat=65,
                           )
                img = scale(img, 3, 3)
                img = resize(img.numpy(), 256)
                img = normalization(img)
                item["grid_content"] = img
                met_sat_file_full_path = "{}/{}.nc".format(met_sat_base_path, t.strftime("%Y%m"))
                idx = int((t.day - 1) * 8 + t.hour / 3)
                print("idx day:{} hour:{} idx:{}".format(t.day, t.hour, idx))
                img2 = read_met_sat(met_sat_file_full_path, idx)
                item["met_content"] = img2
                item["label_content"] = lon_lat_to_graph(item["lat"], item["lon"], 256, 256)
                # min_lat_diff = 0.6
                # max_lat_diff = 62.1
                # min_lon_diff = 100.0
                # max_lon_diff = 179.96099999999998
                item["n_lat"] = (row[3] - min_lat_diff) / (max_lat_diff - min_lat_diff)
                item["n_lon"] = (row[2] - min_lon_diff) / (max_lon_diff - min_lon_diff)
                save_path = os.path.join(save_dir, t.strftime("%Y%m%d%H"))
                save(item, save_path)
                print("save to: {}".format(save_path))


if __name__ == "__main__":
    find(2011)
