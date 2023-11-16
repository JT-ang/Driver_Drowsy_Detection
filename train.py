import torch.utils.data
import torch.optim
from torch.utils.data import Dataset, DataLoader
from model import DDnet, DDpredictor
from data import CustomDataset


def train(model, data_set, epoch_num, device, lr=0.03, batch_size=2, reorder=True):
    model.to(device)
    model.train()
    paras = [para for para in model.predictor.classifier.parameters()]
    loss_list = []
    grad_sum_list = []
    numer_workers = 4
    # only update the weights of the classifier
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.SGD(paras, lr=lr)
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
                loss_val = loss_func(pred_prob, label)
                loss_val.backward()
                optimizer.step()
                epoch_loss += loss_val.item()
            print("done one")
            if idx % 20 == 0:
                print(f"{idx:03d} / {len(data_loader):03d}")
                grad_sum_list.append(check_sum(paras))
        print(f"[Epoch]: {epoch}, [Loss]: {epoch_loss / len(data_set):.4f}")
        print(f"Epoch {epoch:02d}/{epoch_num:02d}")
        loss_list.append(epoch_loss / len(data_set))
    print(f"loss list{loss_list}")
    print(f"weight list{grad_sum_list}")


def check_sum(paras):
    res = 0
    for para in paras:
        res += torch.sum(para.data)
    return res


def testacc(model, data_set, batch_size, device):
    model.to(device)
    model.eval()
    numer_workers = 4
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
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

            print("done one")
            if idx % 20 == 0:
                print(f"{idx:03d} / {len(data_set):03d}")

    accuracy = right_ans / len(data_set)
    average_loss = total_loss / len(data_set)

    print(f"Right:{right_ans}/{len(data_set)}, ACC:{accuracy * 100:.4f}%")
    print(f"Average Loss: {average_loss:.4f}")


if __name__ == '__main__':
    r_device = torch.device('cuda')
    model = DDnet(device=r_device)
    train_data_folder = "E:\\dataset\\train_set"
    dataset = CustomDataset(train_data_folder)
    # TODO: make the labels read from file, the device shouldn't change here
    # train(model, dataset, 2, lr=0.01, device=r_device)
    testacc(model, dataset, 2, device=r_device)
    print("[Train Finished!]")
