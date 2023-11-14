import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, datas, labels):
        # if not isinstance(datas, torch.Tensor):
        #     raise TypeError("Data input should be tensor!")
        # refactor this later ,for now let it be np array
        self.datas = datas
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data_item = self.datas[index]
        label_item = self.labels[index]

        return data_item, label_item
