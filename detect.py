import time
import torch
from model import DDnet
from utils.logger import Logger

from utils.camera import Camera
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the input to a tensor
    transforms.Normalize(0.5, 0.5)
])

if __name__ == '__main__':
    # --- init the model ---
    r_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    yolo_path = './weights/yolov5n_best.pt'
    predictor_path = './weights/DDnet.pth'
    model = DDnet(r_device, yolo_path, False)
    # --- init the logger ---
    record_file_path = "./records.txt"
    recorder = Logger(record_file_path)
    # --- init the camera ---
    camera = Camera(0)
    frame_interval = 3
    batch_size = 5
    # --- START ---
    while True:
        st = time.time()
        frames = camera.get_frames(frame_interval, batch_size, False)
        t_frames = torch.stack([transform(frame) for frame in frames])
        ed = time.time()
        print(f"time cost:{ed - st}")



