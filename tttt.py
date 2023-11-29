import torch
from torchvision import transforms
from PIL import Image

# 加载模型和权重
from Driver_Drowsy_Detection.model import DDnet

model = DDnet(o_device=torch.device('cuda'), y_path="./weights/yolov5n_best.pt", is_train=False)
model.load_state_dict(torch.load("./weights/DDnet.pth"))
model.to(torch.device('cuda'))
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # 调整图像大小为模型所需的输入大小
    transforms.ToTensor(),  # 转换为张量
])

# 加载图像
image_path = "./images/5.jpg"  # 替换为你的图像路径
image = Image.open(image_path).convert("RGB")  # 打开图像并转换为RGB模式
image_tensor = transform(image).unsqueeze(0).to(torch.device('cuda'))  # 预处理图像并添加批次维度

# 使用模型进行预测
with torch.no_grad():
    output = model(image_tensor)

# 打印预测结果
predicted_class = torch.argmax(output, dim=1).item()
print("Predicted class:", predicted_class)