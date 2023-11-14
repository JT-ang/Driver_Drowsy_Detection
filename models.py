import torch.nn as nn

import torch
import cv2
from torchvision import models
from run import Yolov5ONNX


class RegionDetector:
    def __init__(self):
        pass

    def work(self, images):
        return images


class DDnet(nn.Module):
    """
    预测模型：
    input: one frame
    output: a prob with 2 classes
    """
    def __init__(self):
        super(DDnet, self).__init__()
        # TODO change your onnx model file path here
        self.region_maker = Yolov5ONNX(onnx_path='./best.onnx')
        self.predictor = DDpredictor()

    def forward(self, in_frame):
        region_tensor = self.region_maker.get_area(in_frame)
        face, eye = region_tensor[:, 0, :, :, :], region_tensor[:, 1, :, :, :],
        return self.predictor(face, eye)


class DDpredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO change the pretrained = True
        self.eye_alex = models.alexnet(weights=None)
        self.eye_alex.classifier = nn.Sequential()

        self.mouth_alex = models.alexnet(weights=None)
        self.mouth_alex.classifier = nn.Sequential()

        self.predictor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6 * 2, 128),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, eye, mouth):
        eye_fea = self.eye_alex(eye)
        mouth_fea = self.mouth_alex(mouth)
        eye_fea = eye_fea.view(eye_fea.size(0), -1)
        mouth_fea = mouth_fea.view(mouth_fea.size(0), -1)
        fea = torch.cat([eye_fea, mouth_fea], dim=1)
        return self.predictor(fea)


if __name__ == '__main__':
    image = cv2.imread("D:\\Software\\spider\\Driver_Drowsy_Detection\\imgs\\001.bmp")
    model = DDnet()
    predict = model(image)
    print(predict)
