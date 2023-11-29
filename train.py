import torch.utils.data
import torch.optim
from torch.utils.data import Dataset, DataLoader
from model import DDnet, DDpredictor
from data import CustomDataset


def train(model, dataset_train, dataset_test, epoch_num, device, lr=0.03, batch_size=4, reorder=True):
    print("[TRAIN START]")
    model.to(device)
    model.train()
    paras = [para for para in model.predictor.parameters()]
    loss_list = []
    numer_workers = 4
    acc_max = 0
    # only update the weights of the classifier
    loss_func = torch.nn.BCELoss(reduction="mean")
    optimizer = torch.optim.SGD(paras, lr=lr)
    data_loader1 = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=reorder, num_workers=numer_workers)
    for epoch in range(0, epoch_num):
        epoch_loss = 0.0

        for idx, data in enumerate(data_loader1):
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
            if idx % 10 == 0:
                print(f"{idx * batch_size:03d} / {len(dataset_train):03d}")

        print("[TEST START]")
        model.to(device)
        model.eval()
        loss_func = torch.nn.BCELoss(reduction="mean")
        data_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, num_workers=numer_workers)
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
                    print(f"{idx * batch_size:03d} / {len(dataset_test):03d}")

        accuracy = right_ans / len(dataset_test)
        average_loss = total_loss / len(dataset_test)

        print(f"Right:{right_ans}/{len(dataset_test)}, ACC:{accuracy * 100:.4f}%")
        print(f"Average Loss: {average_loss:.4f}")

        if accuracy > acc_max:
            acc_max = accuracy
            print(f'----------------BETTER : {acc_max}---------------------')
            torch.save(model.state_dict(), "weights/DDnet.pth")

            print(f"[Epoch]: {epoch}, [Loss]: {epoch_loss / len(dataset_train):.4f}")
            print(f"Epoch {epoch:02d}/{epoch_num:02d}")
            loss_list.append(epoch_loss / len(dataset_train))
        print(f"loss list{loss_list}")




if __name__ == '__main__':
    r_device = torch.device('cuda')
    save_mode = True
    model = DDnet(o_device=r_device, y_path="./weights/yolov5n_best.pt")
    train_data_folder = "C:\\Users\\admin\\Desktop\\trian_400"
    test_data_folder = "C:\\Users\\admin\\Desktop\\val"
    dataset_train = CustomDataset(train_data_folder)
    dataset_test = CustomDataset(test_data_folder)

    # TODO: make the labels read from file, the device shouldn't change here
    train(model, dataset_train, dataset_test, 40, lr=0.03, device=r_device)
    print("[Train Finished!]")
