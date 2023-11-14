import torch.utils.data
import torch.optim
from torch.utils.data import Dataset, DataLoader
from models import DDnet, DDpredictor
from data import CustomDataset
import cv2


def train(model, data_set, epoch_num, device=torch.device('cpu'), lr=0.03, batch_size=1, reorder=True):
    model.to(device)
    model.train()
    paras = [para for para in model.predictor.parameters()]
    loss_list = []
    numer_workers = 4
    # only update the weights of the classifier
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(paras, lr=lr)
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=reorder, num_workers=numer_workers)
    for epoch in range(0, epoch_num):
        epoch_loss = 0.0
        for idx, data in enumerate(data_loader):
            image, label = data
            image = image.to(device)
            label = label.to(device)
            # 清零
            optimizer.zero_grad()
            # 训练开始
            with torch.set_grad_enabled(True):
                pred_prob = model(image)
                # ans = torch.max(pred_prob, dim=1)
                exit(0)
                loss_val = loss_func(pred_prob, label)
                loss_val.backward()
                optimizer.step()
                epoch_loss += loss_val.item()
            print(f"{idx:03d} / {len(data_loader):03d}")
        print(f"[Epoch]: {epoch}, [Loss]: {epoch_loss:.4f}")
        print(f"Epoch {epoch:02d}/{epoch_num:02d}")


if __name__ == '__main__':
    model = DDnet()
    r_device = torch.device('cpu')
    labels = []
    train_data_folder = "D:\\Software\\spider\\Driver_Drowsy_Detection\\imgs"
    dataset = CustomDataset(train_data_folder)
    dataset.set_label_test(labels)
    labels.append(torch.tensor(([0, 1.0])))
    train(model, dataset, 30, device=r_device, lr=0.003)
