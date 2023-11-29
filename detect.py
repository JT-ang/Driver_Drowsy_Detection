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
    model = DDnet(r_device, yolo_path, False)
    model.load_state_dict(torch.load('weights/DDnet.pth'))
    model.eval()
    # model.requires_grad_(False)
    recorder.log_cli('Init The Camera')
    # --- init the camera ---
    camera = Camera(recorder)
    frame_interval = 1
    batch_size = 10
    # --- camera heat ---
    camera.init_video()
    recorder.log_cli('START')
    # --- START ---
    while True:
        fir = time.time()
        frames = camera.get_frames(frame_interval, batch_size)
        t_frames = torch.stack([transform(frame) for frame in frames])
        sec = time.time()
        res = model(t_frames)
        res = torch.argmax(res, dim=1).sum()
        thr = time.time()
        if res >= 6:
            recorder.log_cli('Warning!')
        else:
            recorder.log_cli('Normal!')
        recorder.log_cli(f"CV Cost: {sec - fir:.3f}s/Batch")
        recorder.log_cli(f"Model Cost: {thr - sec:.3f}s/Batch")