import torch.nn as nn
import torch
from torchvision import models
from utils import *

import cv2
import numpy as np
import onnx
import onnxruntime as ort


class Yolov5ONNX(object):
    """
    PreTrained Model
    """

    def __init__(self, onnx_path):
        """检查onnx模型并初始化onnx"""
        onnx_model = onnx.load(onnx_path)
        try:
            onnx.checker.check_model(onnx_model)
        except Exception:
            print("Model incorrect")
        else:
            print("Model correct")

        options = ort.SessionOptions()
        options.enable_profiling = True
        # self.onnx_session = ort.InferenceSession(onnx_path, sess_options=options,
        #                                          providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.onnx_session = ort.InferenceSession(onnx_path)
        self.input_name = self.get_input_name()  # ['images']
        self.output_name = self.get_output_name()  # ['output0']

    def get_input_name(self):
        """获取输入节点名称"""
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)

        return input_name

    def get_output_name(self):
        """获取输出节点名称"""
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)

        return output_name

    def get_input_feed(self, image_numpy):
        """获取输入numpy"""
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = image_numpy

        return input_feed

    def inference(self, img):
        """
        onnx_session 推理
        """
        input_feed = self.get_input_feed(img)  # dict:{ input_name: input_value }
        pred = self.onnx_session.run(None, input_feed)[0]  # <class 'numpy.ndarray'>(1, 25200, 9)

        return pred


class ObjectDetector:
    """
    RegionMaker: Wrapper of the YOLO
    """

    def __init__(self, onnx_path, classes, conf_thres=0.5, iou_thres=0.5):
        self.classes = classes
        self.onnx_path = onnx_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model = self.load_model()

    def load_model(self):
        model = Yolov5ONNX(self.onnx_path)
        return model

    def filter_box(self, org_box):
        org_box = np.squeeze(org_box)

        conf = org_box[..., 4] > self.conf_thres
        box = org_box[conf == True]
        # print(box.shape)

        cls_cinf = box[..., 5:]
        cls = []
        for i in range(len(cls_cinf)):
            cls.append(int(np.argmax(cls_cinf[i])))
        all_cls = list(set(cls))

        output = []
        for i in range(len(all_cls)):
            curr_cls = all_cls[i]
            curr_cls_box = []
            curr_out_box = []

            for j in range(len(cls)):
                if cls[j] == curr_cls:
                    box[j][5] = curr_cls
                    curr_cls_box.append(box[j][:6])

            curr_cls_box = np.array(curr_cls_box)
            curr_cls_box = xywh2xyxy(curr_cls_box)
            curr_out_box = nms(curr_cls_box, self.iou_thres)

            for k in curr_out_box:
                output.append(curr_cls_box[k])
        output = np.array(output)
        return output

    def draw(self, image, box_data):
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

    def detect_img(self, img):
        output = self.model.inference(img)
        outbox = self.filter_box(output)

        check_set = set(outbox[:, 5])
        if len(outbox) == 0:
            print('没有发现物体')
            # TODO: 提醒操作者脸部存在遮挡
            return None
        if len(check_set) != 3:
            print("集合大小不等于3")
            return None

        img_tensor = make_tensor_back(img, outbox)
        # 返回检测结果
        return img_tensor


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

    def __init__(self,):
        super(DDnet, self).__init__()
        # TODO change your onnx model file path here
        model_path = 'weights/yolov5n.onnx'
        self.region_maker = ObjectDetector(model_path, classes=['face', 'eye', 'mouth'], conf_thres=0.5, iou_thres=0.5)
        self.predictor = DDpredictor()

    def forward(self, in_frame):
        region_tensor = self.region_maker.detect_img(in_frame)
        face, eye = region_tensor[:, 0, :, :, :], region_tensor[:, 1, :, :, :],
        return self.predictor(face, eye)


if __name__ == '__main__':
    image_path = "D:\\Software\\spider\\Driver_Drowsy_Detection\\images\\154.jpg"
    image = get_image_from_path(image_path)
    model = DDnet()
    predict = model(image)
    print(predict)
