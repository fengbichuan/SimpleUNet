# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

# 导入你本地的文件
from ReSimpleUNet import SimpleUNet
from dataload import get_loaders
from metrics import calculate_metrics_and_loss

# --- 1. 配置参数 ---
# !! 修改为你自己的路径 !!
DATA_PATH = r"D:\对照试验模型\dataset\9-isic2018"

# 超参数
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 300
NUM_WORKERS = 4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
NUM_CLASSES = 2  # 0: 背景, 1: 病灶
SAVE_PATH = "[16,32,64,128,256]SimpleUnet.pth"


def train_fn(loader, model, optimizer, loss_fn, device):
    """
    训练一个 Epoch
    """
    loop = tqdm(loader, desc="Training")
    running_loss = 0.0

    model.train()  # 确保在训练模式

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)  # targets 形状 [B, H, W]

        # 1. 前向传播
        predictions = model(data)  # predictions 形状 [B, C, H, W]

        # 2. 计算Loss
        loss = loss_fn(predictions, targets)

        # 3. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 4. 更新进度条
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(loader)
    print(f"Train Epoch Loss: {avg_loss:.4f}")


def main():
    print(f"Using device: {DEVICE}")

    # --- 2. Dataloaders ---
    train_loader, val_loader = get_loaders(
        DATA_PATH,
        BATCH_SIZE,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        NUM_WORKERS,
        PIN_MEMORY
    )

    # --- 3. 初始化模型、Loss、优化器 ---
    # 使用你模型中测试的参数
    model = SimpleUNet(
        in_channels=3,
        num_cls=NUM_CLASSES,
        stage_channels=[16, 32, 64, 128, 256],
        num_blocks=[1, 1, 1, 1, 1],
        short_rate=0.5,
        #adw=True
    ).to(DEVICE)

    # Loss 函数
    loss_fn = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # (可选) 学习率调度器
    # 我们根据 mIoU 来调整学习率，mIoU 越高越好，所以 mode='max'
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=3, factor=0.1, verbose=True
    )

    # --- 4. 训练循环 ---
    best_miou = -1.0  # 用于保存最佳模型

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        # 训练
        train_fn(train_loader, model, optimizer, loss_fn, DEVICE)

        # 验证 (计算 Loss, mIoU, mDice)
        val_loss, mIoU, mDice, iou_cls, dice_cls = calculate_metrics_and_loss(
            val_loader, model, loss_fn, DEVICE, num_cls=NUM_CLASSES
        )

        print(f"\nValidation Results:")
        print(f"  Avg Loss: {val_loss:.4f}")
        print(f"  mIoU:     {mIoU:.4f}")
        print(f"  mDice:    {mDice:.4f}")
        print(f"  IoU (Class 0, 1): {iou_cls[0]:.4f}, {iou_cls[1]:.4f}")

        # 更新学习率 (如果使用)
        scheduler.step(mIoU)

        # 保存最佳模型
        if mIoU > best_miou:
            best_miou = mIoU
            print(f"==> New best model found! mIoU: {mIoU:.4f}")
            torch.save(model.state_dict(), SAVE_PATH)

    print("\nTraining finished.")
    print(f"Best validation mIoU: {best_miou:.4f}")
    print(f"Best model saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()