import torch.optim
from model import DDnet
import torch.utils.data
from data import CustomDataset
from torch.utils.data import DataLoader


def train(model, data_set, epoch_num, device, lr=0.01, batch_size=2, reorder=True):
    print("[TRAIN START]")
    model.to(device)
    model.train()
    paras = [para for para in model.predictor.classifier.parameters()]
    loss_list = []
    grad_sum_list = []
    num_workers = 4

    loss_func = torch.nn.BCELoss(reduction="mean")
    # loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.SGD(paras, lr=lr)

    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=reorder, num_workers=num_workers)

    for epoch in range(0, epoch_num):
        epoch_loss = 0.0

        for idx, data in enumerate(data_loader):
            image, label = data
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            # train
            with torch.set_grad_enabled(True):
                pred_prob = model(image)
                # ans = torch.max(pred_prob, dim=1)
                loss_val = loss_func(pred_prob, label)
                loss_val.backward()
                optimizer.step()
                epoch_loss += loss_val.item()
            if idx % 10 == 0:
                print(f"{idx * batch_size:03d} / {len(data_set):03d}")
                grad_sum_list.append(check_sum(paras))
        print(f"[Epoch]: {epoch+1}, [Loss]: {epoch_loss / len(data_set):.4f}")
        print(f"Epoch {epoch+1:02d}/{epoch_num:02d}")
        loss_list.append(epoch_loss / len(data_set))
    print(f"loss list{loss_list}")
    print(f"weight list{grad_sum_list}")


def check_sum(paras):
    res = 0
    for para in paras:
        res += torch.sum(para.data)
    return res


def testacc(model, data_set, batch_size, device):
    print("[TEST START]")
    model.to(device)
    model.eval()
    num_workers = 4
    loss_func = torch.nn.BCELoss(reduction="mean")
    # loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, num_workers=num_workers)
    total_loss = 0
    right_ans = 0

    with torch.no_grad():
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
    save_mode = False
    train_data_folder = "C:\\Users\\admin\\Desktop\\trian_400"
    test_data_folder = "C:\\Users\\admin\\Desktop\\val"
    model_path = "./weights/yolov5n_best.pt"

    model = DDnet(device=r_device, yolo_path=model_path)
    dataset_train = CustomDataset(train_data_folder)
    dataset_test = CustomDataset(test_data_folder)
    # TODO: make the labels read from file, the device shouldn't change here
    train(model, dataset_train, epoch_num=10, lr=0.03, batch_size=4, device=r_device)
    testacc(model, dataset_test, 4, device=r_device)
    if save_mode:
        torch.save(model.state_dict(), "weights/DDnet.pth")
    print("[Train Finished!]")
