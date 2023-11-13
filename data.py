import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, eye_data, mouth_data, labels):
        if not isinstance(eye_data, torch.Tensor):
            raise TypeError("Dataset input type dismatch!")
        self.eye_data = eye_data
        self.mouth_data = mouth_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        eye_item = self.eye_data[index]
        mouth_item = self.mouth_data[index]
        label_item = self.labels[index]

        return {"eye": eye_item, "mouth": mouth_item}, label_item

