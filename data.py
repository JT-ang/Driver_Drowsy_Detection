import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, train_data_folder_path):
        self.train_data_folder = train_data_folder_path
        self.file_names = os.listdir(train_data_folder_path)

    def set_label_test(self, label_list):
        self.labels = label_list

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        image_path = os.path.join(self.train_data_folder, file_name)

        image = cv2.imread(image_path)
        image = cv2.resize(image, (640, 640))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))
        # cv preprocess
        # Return the image in tensor mode
        # image = torch.from_numpy(image)
        label = self.labels[index]
        return image, label
