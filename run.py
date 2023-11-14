import os

import onnx
import onnxruntime as ort
import cv2
import numpy as np
import torch

CLASSES = ['face', 'eye', 'mouth']  # coco80类别


class Yolov5ONNX(object):
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

    def inference(self, image_in):
        """ 1.拿到输入图像image_in
        2.图像转BGR2RGB和HWC2CHW(因为yolov5的onnx模型输入为 RGB：1 × 3 × 640 × 640)
        3.图像归一化
        4.图像增加维度
        5.onnx_session 推理 """
        if image_in.shape != (640, 640, 3):
            print("input size is not pre-changed")
        image_in = cv2.resize(image_in, (1,640, 640,3))  # resize后的原图 (640, 640, 3)
        img = image_in[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB和HWC2CHW
        img = img.astype(dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)  # [3, 640, 640]扩展为[1, 3, 640, 640]
        # TODO: 这部分应该放到数据集的处理数据输入,在写predict函数时可以这样做
        # img尺寸(1, 3, 640, 640)
        input_feed = self.get_input_feed(img)  # dict:{ input_name: input_value }
        pred = self.onnx_session.run(None, input_feed)[0]  # <class 'numpy.ndarray'>(1, 25200, 9)

        return pred, image_in

    def get_area(self, frame):
        """
        :param frame:输入一个numpy数组，表示一张图像
        :return: 返回一个tensor
        :details 默认该函数为训练过程中使用，因此不会出现frame中拿不到要求数量的区域
        """
        # TODO: 或许改成tensor?
        if not isinstance(frame, np.ndarray):
            if isinstance(frame, torch.Tensor):
                frame = frame.numpy()
            else:
                raise TypeError("Should be np array")

        pred, ori_frame = self.inference(frame)
        filter_res = filter_box(pred, 0.5, 0.5)
        res_tensor = make_tensor_back(ori_frame, filter_res)
        print(f"Res Tensor shape:{res_tensor.shape}")
        return res_tensor


# dets:  array [x,6] 6个值分别为x1,y1,x2,y2,score,class
# thresh: 阈值
def nms(dets, thresh):
    # dets:x1 y1 x2 y2 score class
    # x[:,n]就是取所有集合的第n个数据
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # -------------------------------------------------------
    #  计算框的面积
    #  置信度从大到小排序
    # -------------------------------------------------------
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]  # np.argsort()对某维度从小到大排序
    # [::-1] 从最后一个元素到第一个元素复制一遍。倒序从而从大到小排序

    while index.size > 0:
        i = index[0]
        keep.append(i)
        # -------------------------------------------------------
        #  计算相交面积
        #  1.相交
        #  2.不相交
        # -------------------------------------------------------
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h
        # -------------------------------------------------------
        #  计算该框与其它框的IOU，去除掉重复的框，即IOU值大的框
        #  IOU小于thresh的框保留下来
        # -------------------------------------------------------
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return keep


def xywh2xyxy(x):
    # [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def filter_box(org_box, conf_thres, iou_thres):  # 过滤掉无用的框
    # -------------------------------------------------------
    #  删除为1的维度
    #  删除置信度小于conf_thres的BOX
    # -------------------------------------------------------
    org_box = np.squeeze(org_box)  # 删除数组形状中单维度条目(shape中为1的维度)
    # (25200, 9)
    # […,4]：代表了取最里边一层的所有第4号元素，…代表了对:,:,:,等所有的的省略。此处生成：25200个第四号元素组成的数组
    conf = org_box[..., 4] > conf_thres  # 0 1 2 3 4 4是置信度，只要置信度 > conf_thres 的
    box = org_box[conf == True]  # 根据objectness score生成(n, 9)，只留下符合要求的框

    # -------------------------------------------------------
    #   通过argmax获取置信度最大的类别
    # -------------------------------------------------------
    cls_cinf = box[..., 5:]  # 左闭右开（5 6 7 8），就只剩下了每个grid cell中各类别的概率
    cls = []
    for i in range(len(cls_cinf)):
        cls.append(int(np.argmax(cls_cinf[i])))  # 剩下的objecctness score比较大的grid cell，分别对应的预测类别列表
    all_cls = list(set(cls))  # 去重，找出图中都有哪些类别
    # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
    # -------------------------------------------------------
    #  分别对每个类别进行过滤
    #  1.将第6列元素替换为类别下标
    #  2.xywh2xyxy 坐标转换
    #  3.经过非极大抑制后输出的BOX下标
    #  4.利用下标取出非极大抑制后的BOX
    # -------------------------------------------------------
    output = []
    for i in range(len(all_cls)):
        curr_cls = all_cls[i]
        curr_cls_box = []

        for j in range(len(cls)):
            if cls[j] == curr_cls:
                box[j][5] = curr_cls
                curr_cls_box.append(box[j][:6])  # 左闭右开，0 1 2 3 4 5

        curr_cls_box = np.array(curr_cls_box)  # 0 1 2 3 4 5 分别是 x y w h score class
        curr_cls_box = xywh2xyxy(curr_cls_box)  # 0 1 2 3 4 5 分别是 x1 y1 x2 y2 score class
        curr_out_box = nms(curr_cls_box, iou_thres)  # 获得nms后，剩下的类别在curr_cls_box中的下标

        for k in curr_out_box:
            output.append(curr_cls_box[k])
    output = np.array(output)
    return output


def draw(image, box_data):
    # -------------------------------------------------------
    # 取整，方便画框
    # -------------------------------------------------------
    boxes = box_data[..., :4].astype(np.int32)  # x1 x2 y1 y2
    scores = box_data[..., 4]
    classes = box_data[..., 5].astype(np.int32)
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
    return image


def make_tensor_back(ori_image, crop_coor):
    idx = crop_coor.shape[0]
    ori_image = cv2.resize(ori_image, (640, 640))
    # 只返回脸和眼睛
    input_tensor = torch.Tensor(1, 2, 3, 227, 227)

    for i in range(idx):
        x1, y1, x2, y2, score, cls = crop_coor[i]
        if cls != 0 or cls != 1:
            continue
        # 裁剪目标区域
        cropped_image = ori_image[int(y1):int(y2), int(x1):int(x2), :]

        # 调整图像尺寸为模型所需的输入尺寸
        resized_image = cv2.resize(cropped_image, (227, 227))

        # 将图像从HWC格式转换为CHW格式
        image_data = resized_image.transpose(2, 0, 1)
        # 将像素值归一化到0-1范围
        image_data = image_data / 255.0
        # 添加批处理维度
        image_data = np.expand_dims(image_data, axis=0)
        # 将图像转换为浮点型张量
        image_tensor = torch.FloatTensor(image_data)

        input_tensor[:, int(cls), :, :, :] = image_tensor

    return input_tensor


if __name__ == "__main__":
    onnx_path = './best.onnx'
    img_folder = './imgs'
    model = Yolov5ONNX(onnx_path)

    for filename in os.listdir(img_folder):
        if filename.endswith('.bmp') or filename.endswith('.jpg') or filename.endswith('.png') \
                or filename.endswith('.jpeg'):
            img_path = os.path.join(img_folder, filename)
            input_image = cv2.imread(img_path)
            output, or_img = model.inference(input_image)

            outbox = filter_box(output, 0.5, 0.5)  # 最终剩下的Anchors：0 1 2 3 4 5 分别是 x1 y1 x2 y2 score class
            print('outbox( x1 y1 x2 y2 score class):')
            check_set = set(outbox[:, 5])
            print(outbox)
            if len(outbox) == 0:
                print('没有发现物体')
                # TODO: 提醒操作者脸部存在遮挡
                continue
            if len(check_set) != 3:
                print("集合大小不等于3")
                continue
            tensor = make_tensor_back(input_image, outbox)

            # 保存结果图像
            or_img = draw(or_img, outbox)
            # result_path = os.path.join('./results', filename)
            # cv2.imwrite(result_path, or_img)
            cv2.imshow("final", or_img)
            cv2.waitKey(0)
