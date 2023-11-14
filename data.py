import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, train_data_folder_path):
        self.train_data_folder = train_data_folder_path
        self.file_names = os.listdir(train_data_folder_path)
        self.labels = []

    def set_label_test(self, label_list):
        self.labels = label_list

    def __len__(self):
        if len(self.file_names) == len(self.labels):
            return len(self.file_names)
        else:
            print(f"data size:{len(self.file_names)}, labels:{len(self.labels)}")
            raise ValueError("Dataset data should match with labels")

    def __getitem__(self, index):
        # TODO 优化存储的结构，试着不要用list而使用tensor？
        file_name = self.file_names[index]
        image_path = os.path.join(self.train_data_folder, file_name)

        image = get_image_from_filepath(image_path, False)
        # cv preprocess
        # Return the image in tensor mode
        # image = torch.from_numpy(image)
        label = self.labels[index]
        return image, label


def get_image_from_filepath(img_path, need_batch=True):
    # (640,640,3) -> (1, 3, 640, 640)
    img = cv2.imread(img_path)
    or_img = cv2.resize(img, (640, 640))
    img = or_img[:, :, ::-1].transpose((2, 0, 1))
    img = img.astype(dtype=np.float32)
    img /= 255.0
    if need_batch:
        img = np.expand_dims(img, axis=0)

    return img
