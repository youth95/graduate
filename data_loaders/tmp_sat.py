from tasks.select_grad_by_tra_set import read
import os
from torch.utils.data import Dataset, ConcatDataset
import torch
import numpy as np
from env.base import data_root



class MockDataSet(Dataset):
    def __getitem__(self, index):
        x = torch.randn((16, 5, 256, 256))
        y = torch.randn((8, 1, 256, 256))
        return x, y

    def __len__(self):
        return 100


class TFDataSet(Dataset):

    def __init__(self, items, in_step, out_step):
        # TODO 可能需要排序
        self.data = []
        self.in_step = in_step
        self.out_step = out_step
        seq = in_step + out_step
        _len = len(items)
        for i in range(_len - in_step - out_step):
            self.data.append(items[i:seq + i])

    def __getitem__(self, index):
        result = self.data[index]
        x, y = result[:self.in_step], result[self.in_step:]
        x = [
            np.vstack(
                [np.array(item['met_content']).reshape((4, 256, 256)),
                 np.array(item['grid_content']).reshape((1, 256, 256))
                 ]
            ).reshape((1, 5, 256, 256)) for item in x]
        x = torch.tensor(np.vstack(x))
        y = [np.array(item["label_content"]).reshape((1, 1, 256, 256)) for item in y]
        y = torch.tensor(np.vstack(y))
        return x, y

    def __len__(self):
        return len(self.data)


def fetch_mapper(year):
    mapper = {}
    root_path = data_root.format(year)
    file_list = os.listdir(root_path)
    for file in file_list:
        item = read(os.path.join(root_path, file))
        if item["sid"] not in mapper:
            mapper[item["sid"]] = []
        mapper[item["sid"]].append(item)
    return mapper


def fetch_data_set(years):
    result = []
    for year in years:
        mapper = fetch_mapper(year)
        data_set = ConcatDataset([TFDataSet(
            items=item,
            in_step=16,
            out_step=8,
        ) for item in list(mapper.values())])
        result.append(data_set)
    return ConcatDataset(result)


if __name__ == "__main__":
    data_set = fetch_data_set([2010])
    print(len(data_set))
