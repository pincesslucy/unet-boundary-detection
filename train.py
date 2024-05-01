import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from bsdsdataset import CustomDataset
from model import UNet
from torch.utils.tensorboard import SummaryWriter
import os
import time

# load data
data_dir = './BSDS500/'

# transform
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))])

train_dataset = CustomDataset(data_dir=data_dir, type='train', transform=transform)
val_dataset = CustomDataset(data_dir=data_dir, type='val', transform=transform)


# DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model
model = UNet().to(device)

# loss, optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# tensorboard
now = time.localtime()
train_dir = f'./logs/train/{now.tm_year}_{now.tm_mon}_{now.tm_mday}_{now.tm_hour}_{now.tm_min}_{now.tm_sec}'
val_dir = f'./logs/val/{now.tm_year}_{now.tm_mon}_{now.tm_mday}_{now.tm_hour}_{now.tm_min}_{now.tm_sec}'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
writer_train = SummaryWriter(log_dir=train_dir)
writer_val = SummaryWriter(log_dir=val_dir)

# 학습
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    loss_arr = []
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_arr += [loss.item()]   

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {np.mean(loss_arr):.4f}')
        # tensorboard
        label = labels.cpu().detach().numpy().transpose(0, 2, 3, 1)
        input = images.cpu().detach().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5
        output = outputs.cpu().detach().numpy().transpose(0, 2, 3, 1)
        writer_train.add_images('label', label, epoch * len(train_loader) + i, dataformats='NHWC')
        writer_train.add_images('input', input, epoch * len(train_loader) + i, dataformats='NHWC')
        writer_train.add_images('output', output, epoch * len(train_loader) + i, dataformats='NHWC')
    writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

    with torch.no_grad():
        model.eval()
        loss_arr = []
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss_arr += [loss.item()]
            
            # tensorboard
            label = labels.cpu().detach().numpy().transpose(0, 2, 3, 1)
            input = images.cpu().detach().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5
            output = outputs.cpu().detach().numpy().transpose(0, 2, 3, 1)
            output = np.where(output>0.5, 1, 0)
            writer_val.add_images('label', label, epoch * len(val_loader) + i, dataformats='NHWC')
            writer_val.add_images('input', input, epoch * len(val_loader) + i, dataformats='NHWC')
            writer_val.add_images('output', output, epoch * len(val_loader) + i, dataformats='NHWC')
        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {np.mean(loss_arr):.4f}')
    # save model
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), f'./save/model_epoch_{epoch+1}.pth')

    writer_train.close()
    writer_val.close()