import numpy as np
import torch
from model import DDnet

from utils.camera import Camera
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),   # Convert the input to a tensor
    transforms.Normalize(0.5, 0.5)
])

if __name__ == '__main__':
    # init a camera object
    camera = Camera(0)
    frame_interval = 10
    batch_size = 2
    # get frames_tensor
    frames = camera.get_frames(frame_interval, batch_size)
    frames_tensor = torch.stack([transform(frame) for frame in frames])
    # define model
    r_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = DDnet(r_device, "./weights/yolov5n_best.pt")
    # model.load_state_dict(torch.load("TrainedWeights.pth"))
    # predict
    res = model(frames_tensor)
    print(res)
