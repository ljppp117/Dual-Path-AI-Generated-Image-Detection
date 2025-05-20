import torch
import torch.optim as optim
import torch.nn as nn
from clipforfakedetection.clipfordetectiondata.datasets import TestDataset, TrainDataset
from clipforfakedetection.train_model.earlystop import EarlyStopping
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, average_precision_score
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
from clipforfakedetection.models.clipnet import OpenClipLinear
from tqdm import tqdm

def evaluate_model(model, dataloader, device, writer, prefix, epoch):
    model.eval()
    predictions = []
    labels = []
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Evaluating Epoch {epoch+1}'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            loss = nn.BCEWithLogitsLoss()(outputs, targets.float())
            total_loss += loss.item()

            predicted = (outputs > 0.5).float()
            predictions.extend(predicted.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    ac = accuracy_score(labels, predictions)
    ap = average_precision_score(labels, predictions)
    loss = total_loss / len(dataloader)

    writer.add_scalar(f'{prefix}/loss', loss, epoch)
    writer.add_scalar(f'{prefix}/accuracy', ac, epoch)
    writer.add_scalar(f'{prefix}/average_precision', ap, epoch)

    return ac, ap, -ac

def train_model(model, train_dataloader, test_dataloader, epochs, device, save_path):
    model = nn.DataParallel(model)  # 包装模型以使用 DataParallel
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=0)

    best_ac = 0
    best_model_state = None
    early_stopping = EarlyStopping(patience=5, verbose=True)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(save_path, f"train_log{current_time}")
    eval_log_dir = os.path.join(save_path, f"eval_log{current_time}")
    train_writer = SummaryWriter(train_log_dir)
    eval_writer = SummaryWriter(eval_log_dir)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_dataloader)

        for batch_idx, (data, target) in tqdm(enumerate(train_dataloader), total=total_batches, desc=f'Epoch {epoch + 1}/{epochs}'):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / total_batches
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}')

        train_writer.add_scalar('training_loss', epoch_loss, epoch)

        ac, ap, val_loss = evaluate_model(model, test_dataloader, device, eval_writer, 'validation', epoch)
        print(f'Epoch {epoch + 1}, Test AC: {ac:.4f}, Test AP: {ap:.4f}')

        if ac > best_ac:
            best_ac = ac
            best_model_state = model.state_dict()

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    print('Finished Training')
    train_writer.close()
    eval_writer.close()

    if best_model_state:
        model.load_state_dict(best_model_state)
        model_save_path = f"{save_path}/best_model_{current_time}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f'Best Test AC: {best_ac:.4f}')
    else:
        print('No better model found.')

# 创建数据集实例
train_dataset = TrainDataset(is_train=True, args={'data_path': '/home/pc/code/ljp/clipdetectiondataset/train'})
test_dataset = TestDataset(is_train=False, args={'data_path': '/home/pc/code/ljp/clipdetectiondataset/val', 'eval_data_path': '/home/pc/code/ljp/clipdetectiondataset/val'})

# 创建DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda:0")  # 设置默认设备为第一个GPU

pretrained_weights_path = '../weights/open_clip_pytorch_model.bin'
model = OpenClipLinear(normalize=True, next_to_last=True, pretrained_model_path=pretrained_weights_path)
print(f"Loaded CLIP model")

save_path = '../weights/model_save'
train_model(model, train_dataloader, test_dataloader, epochs=3, device=device, save_path=save_path)