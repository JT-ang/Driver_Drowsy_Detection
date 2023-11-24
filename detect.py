import time
import torch
from model import DDnet
from utils.logger import Logger

from utils.camera import Camera
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),   # Convert the input to a tensor
    transforms.Normalize(0.5, 0.5)
])

if __name__ == '__main__':
    # init a camera object & a logger
    record_file_path = "./records.txt"
    camera = Camera(0)
    recorder = Logger(record_file_path)
    frame_interval = 3
    batch_size = 10
    # get frames_tensor
    danger_counter = 0
    # while True:
    frames = camera.get_frames(frame_interval, batch_size)
    frames_tensor = torch.stack([transform(frame) for frame in frames])
    # define model
    r_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = DDnet(r_device, "./weights/yolov5n_best.pt")
    # model.load_state_dict(torch.load("TrainedWeights.pth"))
    # predict
    res = model(frames_tensor)
    ans = torch.argmax(res, dim=1)
    drowsy_bool = True if (ans == 1).sum() > 0 else False
    if drowsy_bool:
        recorder.log_info("Drowsy")
    recorder.flush_buffer()
    recorder.close()
    print("Normal")
    # print(res)