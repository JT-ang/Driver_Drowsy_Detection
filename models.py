import torch.nn as nn

import torch
from torchvision import models
from pytorchsummary import summary


class RegionDetector:
    def __init__(self):
        pass

    def work(self, images):
        return images


class DDnet(nn.Module):
    def __init__(self):
        super(DDnet, self).__init__()
        self.predictor = DDpredictor()

    def forward(self, X):
        eye_r, mouth_r = self.region_maker.work(X)
        return self.predictor(eye_r, mouth_r)


class DDpredictor(nn.Module):
    def __init__(self):
        super().__init__()

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
            nn.Linear(32, 3),
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
    eye = torch.randn(size=(1, 3, 227, 227))
    mouth = torch.randn(size=(1, 3, 227, 227))
    model = DDpredictor()
    predict = model(eye, mouth)
    print(predict)
