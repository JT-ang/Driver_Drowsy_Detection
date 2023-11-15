import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression

from utils.dataloader import get_image_from_path, image_processor2


class YOLODetector:
    def __init__(self, model_path, classes, device=torch.device('cpu')):
        self.classes = classes
        self.device = device
        self.model = attempt_load(model_path).to(self.device)
        self.model.requires_grad_(False)

    def detect(self, image_tensor):
        output = self.model(image_tensor)
        preds = non_max_suppression(output, 0.5, 0.5)
        return preds

    def draw(self, image_tensor, box_data):
        box_data = box_data.cpu().numpy()
        image = image_processor2(image_tensor)
        boxes = box_data[..., :4].astype(np.int32)
        scores = box_data[..., 4]
        classes = box_data[..., 5].astype(np.int32)
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = box
            print('class: {}, score: {}'.format(self.classes[cl], score))
            print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(self.classes[cl], score),
                        (top, left),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
        return image

    def process_batch(self, batch_images):
        face_tensor = torch.empty(0)
        eye_tensor = torch.empty(0)
        mouth_tensor = torch.empty(0)
        preds = self.detect(batch_images.to(self.device))

        for k, pred in enumerate(preds):
            # img = self.draw(batch_images[k], pred)
            # cv2.imshow("11", img)
            # cv2.waitKey(0)
            check_set = torch.unique(pred[:, 5])
            if len(pred) == 0:
                print('没有发现物体')
                exit(0)
                # TODO: 提醒操作者脸部存在遮挡
                return None
            if len(check_set) != 3:
                print("集合大小不等于3")
                exit(0)
                return None

            face_score = 0
            eye_score = 0
            mouth_score = 0
            for i in range(len(pred)):
                x1, y1, x2, y2, score, cls = pred[i]
                x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
                if cls == 0:
                    if score > face_score:
                        face_score = score
                        face_image = batch_images[k, :, y1:y2, x1:x2]
                        face_image = TF.resize(face_image, [227, 227]).unsqueeze(0)
                        face_tensor = torch.cat((face_tensor, face_image))
                elif cls == 1:
                    if score > eye_score:
                        eye_score = score
                        eye_image = batch_images[k, :, y1:y2, x1:x2]
                        eye_image = TF.resize(eye_image, [227, 227]).unsqueeze(0)
                        eye_tensor = torch.cat((eye_tensor, eye_image))
                else:
                    if score > mouth_score:
                        mouth_score = score
                        mouth_image = batch_images[k, :, y1:y2, x1:x2]
                        mouth_image = TF.resize(mouth_image, [227, 227]).unsqueeze(0)
                        mouth_tensor = torch.cat((mouth_tensor, mouth_image))

        return face_tensor, eye_tensor, mouth_tensor


class DDpredictor(nn.Module):
    """
    Classifier: Core part of the DDnet
    """

    def __init__(self):
        super().__init__()
        # TODO change the pretrained = True
        self.eye_alex = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.eye_alex.classifier = nn.Sequential()
        self.eye_alex.requires_grad_(False)

        self.mouth_alex = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.mouth_alex.classifier = nn.Sequential()
        self.mouth_alex.requires_grad_(False)

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


class DDnet(nn.Module):
    """
    预测模型：
    input: one frame
    output: a prob with 2 classes
    """

    def __init__(self, device=torch.device('cpu')):
        super(DDnet, self).__init__()
        # TODO change your onnx model file path here
        self.device = device
        model_path = "./weights/yolov5n_face.pt"
        CLASSES = ['face', 'eye', 'mouth']
        self.region_maker = YOLODetector(model_path, device=self.device, classes=CLASSES)
        self.predictor = DDpredictor()

    def forward(self, in_frame):
        face_tensor, eye_tensor, mouth_tensor = self.region_maker.process_batch(in_frame)
        return self.predictor(eye_tensor, mouth_tensor)


if __name__ == '__main__':
    image_path = "./images/001.jpg"
    device = torch.device('cpu')
    image = get_image_from_path(image_path)
    image = torch.from_numpy(image)
    model = DDnet(device)
    predict = model(image)
    print(predict)

