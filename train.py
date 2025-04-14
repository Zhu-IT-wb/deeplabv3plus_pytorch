import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from tqdm import tqdm
from deeplabv3plus import DeeplabV3Plus
from config import Config



# 训练
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc="Train", leave=False):
        imgs, masks = imgs.to(Config.DEVICE), masks.squeeze(1).long().to(Config.DEVICE)
        masks[masks == 255] = -1
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 验证
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Val", leave=False):
            imgs, masks = imgs.to(Config.DEVICE), masks.squeeze(1).long().to(Config.DEVICE)
            masks[masks == 255] = -1
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(loader)

# 主函数
def main():
    train_loader, val_loader = get_dataloaders()

    model = DeeplabV3Plus()
    model.to(Config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch [{epoch+1}/{Config.EPOCHS}]")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "deeplabv3_voc2012.pth")
    print("模型已保存为 deeplabv3_voc2012.pth")

if __name__ == '__main__':
    main()
