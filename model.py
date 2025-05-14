import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression

from utils.dataloader import get_image_from_path, image_processor2, change_img_color_to_show


class YOLODetector:
    def __init__(self, model_path, classes, o_device):
        self.device = o_device
        self.classes = classes
        self.model = attempt_load(model_path).to(self.device)
        self.model.requires_grad_(False)

    def detect(self, image_tensor):
        output = self.model(image_tensor)
        preds = non_max_suppression(output, 0.4, 0.4)
        return preds

    def draw(self, image_tensor, box_data):
        """
        return an image with boxes
        """
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

    def process_batch(self, batch_images, is_trained=True):
        face_tensor = torch.empty(0).to(self.device)
        eye_tensor = torch.empty(0).to(self.device)
        mouth_tensor = torch.empty(0).to(self.device)
        batch_images = batch_images.to(self.device)
        preds = self.detect(batch_images)
        bs, chl = batch_images.shape[0], batch_images.shape[1]

        for k, pred in enumerate(preds):
            check_set = torch.unique(pred[:, 5])
            if len(pred) == 0 or len(check_set) != 3:
                if is_trained:
                    img = self.draw(batch_images[k], pred)
                    cv2.imshow("检测效果图", img)
                    cv2.waitKey(0)
                    print("detect fail")
                    exit(0)
                    # TODO: 提醒操作者脸部存在遮挡
                else:
                    print('[Detect] Jump Invalid Frames')
                    face_tensor = torch.zeros(size=(bs, chl, 224, 224), device=self.device)
                    eye_tensor = torch.zeros(size=(bs, chl, 224, 224), device=self.device)
                    mouth_tensor = torch.zeros(size=(bs, chl, 224, 224), device=self.device)
                    return face_tensor, eye_tensor, mouth_tensor

            # draw the image if success
            pred = preds[0]
            if not (len(pred) == 0 or len(check_set) != 3):
                img = self.draw(batch_images[0], pred)
                img = change_img_color_to_show(img)
                cv2.imshow("检测效果图", img)
                cv2.waitKey(10)

            face_score = 0
            eye_score = 0
            mouth_score = 0
            face_image = torch.empty(0).to(self.device)
            eye_image = torch.empty(0).to(self.device)
            mouth_image = torch.empty(0).to(self.device)
            for i in range(len(pred)):
                x1, y1, x2, y2, score, cls = pred[i]
                x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), max(0, int(x2)), max(0, int(y2))
                if cls == 0:
                    if score > face_score:
                        face_score = score
                        face_image = batch_images[k, :, y1:y2, x1:x2]
                        face_image = TF.resize(face_image, [224, 224]).unsqueeze(0)
                elif cls == 1:
                    if score > eye_score:
                        eye_score = score
                        eye_image = batch_images[k, :, y1:y2, x1:x2]
                        eye_image = TF.resize(eye_image, [224, 224]).unsqueeze(0)
                else:
                    if score > mouth_score:
                        mouth_score = score
                        mouth_image = batch_images[k, :, y1:y2, x1:x2]
                        mouth_image = TF.resize(mouth_image, [224, 224]).unsqueeze(0)

            face_tensor = torch.cat((face_tensor, face_image))
            eye_tensor = torch.cat((eye_tensor, eye_image))
            mouth_tensor = torch.cat((mouth_tensor, mouth_image))

        return face_tensor, eye_tensor, mouth_tensor


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DDpredictor(nn.Module):
    """
    Classifier: Core part of the DDnet
    """

    def __init__(self):
        super(DDpredictor, self).__init__()
        self.eye_mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.eye_mobilenet.classifier = nn.Sequential()
        self.eye_mobilenet.requires_grad_(False)

        self.mouth_mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.mouth_mobilenet.classifier = nn.Sequential()
        self.mouth_mobilenet.requires_grad_(False)

        self.eye_se = SELayer(576)
        self.mouth_se = SELayer(576)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(576 * 2, 128),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )

    def forward(self, eye, mouth):
        eye_fea = self.eye_mobilenet.features(eye)
        mouth_fea = self.mouth_mobilenet.features(mouth)

        eye_fea = self.eye_se(eye_fea)
        mouth_fea = self.mouth_se(mouth_fea)

        eye_fea = eye_fea.mean([2, 3])  # Global average pooling
        mouth_fea = mouth_fea.mean([2, 3])  # Global average pooling

        fea = torch.cat([eye_fea, mouth_fea], dim=1)
        return self.classifier(fea)


class DDnet(nn.Module):
    """
    预测模型：
    input: one frame
    output: a prob with 2 classes
    """

    def __init__(self, o_device, y_path, is_train=True):
        super(DDnet, self).__init__()
        self.device = o_device
        self.yolo_path = y_path
        self.is_train = is_train
        classes = ['face', 'eye', 'mouth']
        # --- init the torch module ---
        self.region_maker = YOLODetector(self.yolo_path, o_device=self.device, classes=classes)
        self.predictor = DDpredictor().to(device=self.device)
        # --- set grad requirements ---
        self.region_maker.model.requires_grad_(False)
        self.predictor.classifier.requires_grad_(True)

    def forward(self, in_frame):
        face_tensor, eye_tensor, mouth_tensor = self.region_maker.process_batch(in_frame, self.is_train)
        print(face_tensor, eye_tensor, mouth_tensor)
        return self.predictor(eye_tensor, face_tensor)


if __name__ == '__main__':
    image_path = "./images/test.jpg"
    device = torch.device('cuda')
    image = get_image_from_path(image_path)
    image = torch.from_numpy(image)
    model = DDnet(device, "./weights/yolov5n_best.pt")
    predict = model(image)
    print(predict)
