import time
import torch
from model import DDnet
from utils.logger import Logger

from utils.camera import Camera
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the input to a tensor
])

if __name__ == '__main__':
    # --- init the logger ---
    record_file_path = "./records.txt"
    recorder = Logger(record_file_path)
    recorder.log_cli('Init The Model')
    # --- init the model ---
    r_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    yolo_path = './weights/yolov5n_best.pt'
    predictor_path = './weights/DDnet.pth'
    model = DDnet(r_device, yolo_path, True)
    model.load_state_dict(torch.load('weights/DDnet.pth'))
    # model.requires_grad_(False)
    recorder.log_cli('Init The Camera')
    # --- init the camera ---
    camera = Camera(recorder)
    frame_interval = 2
    batch_size = 5
    recorder.log_cli('START')
    # --- START ---
    while True:
        st = time.time()
        frames = camera.get_frames(frame_interval, batch_size, False)
        t_frames = torch.stack([transform(frame) for frame in frames])
        res = model(t_frames)
        res = torch.argmax(res, dim=1)
        ed = time.time()
        print(res)
        recorder.log_cli(f"Time Cost: {ed - st:.3f}s/Batch")
        break