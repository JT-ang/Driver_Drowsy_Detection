import torch
from torch.utils.data import DataLoader

from Driver_Drowsy_Detection.data import CustomDataset
from Driver_Drowsy_Detection.model import DDnet


def testacc(model, data_set, batch_size, device):
    print("[TEST START]")
    model.to(device)
    model.eval()
    numer_workers = 4
    # loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    loss_func = torch.nn.BCELoss(reduction="mean")
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, num_workers=numer_workers)
    total_loss = 0
    right_ans = 0

    with torch.no_grad():  # Disable gradient computation
        for idx, data in enumerate(data_loader):
            image, label = data
            image = image.to(device)
            label = label.to(device)

            pred_prob = model(image)
            loss_val = loss_func(pred_prob, label)

            total_loss += loss_val.item()
            correct_num = (torch.argmax(label, dim=1) == torch.argmax(pred_prob, dim=1)).sum()
            right_ans += correct_num

            if idx % 20 == 0:
                print(f"{idx * batch_size:03d} / {len(data_set):03d}")

    accuracy = right_ans / len(data_set)
    average_loss = total_loss / len(data_set)

    print(f"Right:{right_ans}/{len(data_set)}, ACC:{accuracy * 100:.4f}%")
    print(f"Average Loss: {average_loss:.4f}")


if __name__ == '__main__':
    r_device = torch.device('cuda')
    save_mode = True
    model = DDnet(o_device=r_device, y_path="./weights/yolov5n_best.pt", is_train=False)
    model.load_state_dict(torch.load("./weights/DDnet.pth"))
    train_data_folder = "C:\\Users\\admin\\Desktop\\my"
    dataset = CustomDataset(train_data_folder)

    testacc(model, dataset, 2, device=r_device)

    print("[Train Finished!]")
