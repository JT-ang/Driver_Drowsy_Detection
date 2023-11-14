import cv2
import torch
import numpy as np


def xywh2xyxy(x):
    # [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    # print(scores)
    keep = []
    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h

        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return keep


def make_tensor_back(img, crop_coor, debug=False):
    idx = crop_coor.shape[0]
    or_img = image_processor2(img)
    input_tensor = torch.Tensor(1, 3, 3, 227, 227)

    for i in range(idx):
        x1, y1, x2, y2, score, cls = crop_coor[i]
        cropped_image = or_img[int(y1):int(y2), int(x1):int(x2), :]
        resized_image = cv2.resize(cropped_image, (227, 227))
        if debug:
            cv2.imshow("1", resized_image)
            cv2.waitKey(0)

        image_data = resized_image.transpose((2, 0, 1))
        image_data = image_data / 255.0
        image_data = np.expand_dims(image_data, axis=0)

        image_tensor = torch.FloatTensor(image_data)
        input_tensor[:, int(cls), :, :, :] = image_tensor

    return input_tensor

def get_image_from_path(img_path):
    # (640,640,3) -> (1, 3, 640, 640)
    img = cv2.imread(img_path)
    or_img = cv2.resize(img, (640, 640))
    img = or_img[:, :, ::-1].transpose((2, 0, 1))
    img = img.astype(dtype=np.float32)
    img /= 255.0
    img = np.expand_dims(img, axis=0)

    return img

def image_processor2(img):
    # (1, 3, 640, 640) -> (640,640,3)
    img = img.squeeze()
    img = img.transpose((1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    or_img = (img * 255.0).astype(np.uint8)

    return or_img