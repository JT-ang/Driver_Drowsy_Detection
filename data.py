import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.train_data_folder = data_path
        self.labels = []
        self.file_paths = []

        normal_path = os.path.join(data_path, "normal")
        drowsy_path = os.path.join(data_path, "drowsy")

        self.file_paths.extend([os.path.join(normal_path, file_name) for file_name in os.listdir(normal_path)])
        self.file_paths.extend([os.path.join(drowsy_path, file_name) for file_name in os.listdir(drowsy_path)])

        self.labels.extend(make_label_tensors(normal_path))
        self.labels.extend(make_label_tensors(drowsy_path))

    def set_label_test(self, label_list):
        self.labels = label_list

    def __len__(self):
        if len(self.file_paths) == len(self.labels):
            return len(self.file_paths)
        else:
            print(f"data size: {len(self.file_paths)}, labels: {len(self.labels)}")
            raise ValueError("Dataset data should match with labels")

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        image = get_image_from_filepath(file_path, False)
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


def make_label_tensors(path):
    files = os.listdir(path)
    num = len(files) if len(files) else -1
    if num == -1:
        raise ValueError("Wrong Dataset init with folder wrong")
    if path.find("drowsy") != -1:
        # drowsy
        labels = [torch.tensor([0, 1.0]) for _ in range(num)]
        return labels
    else:
        # normal
        labels = [torch.tensor([1.0, 0]) for _ in range(num)]
        return labels


