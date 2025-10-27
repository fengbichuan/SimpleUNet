import torch
import torch.nn as nn
import torch.optim as optim
import os

from thop import profile
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
SAVE_PATH = "Wavelet-isic-2018-[16,16,16,16,16]"
early_stop_patience = 10  # 你要求的10轮
early_stop_counter = 0    # 计数器
stage_channels = [16,16,16,16,16]


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
        stage_channels=stage_channels,
        num_blocks=[1, 1, 1, 1, 1],
        short_rate=0.5,
        #adw=True
    ).to(DEVICE)

    # <--- 2. 在这里插入 Params 和 FLOPs 计算 ---
    print("\n" + "---" * 15)
    print("Calculating model parameters and FLOPs...")

    # 2.1 计算可训练参数量 (单位: M, 兆)
    # 1e6 = 1,000,000 (M)
    trainable_params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total Trainable Params (M): {trainable_params_m:.2f}M")

    # 2.2 计算 FLOPs (单位: G, 吉)
    # 创建一个符合模型输入的虚拟张量
    # 形状: (Batch_size=1, Channels, Height, Width)
    # 我们使用 Batch_size=1 来计算单个样本的 FLOPs
    dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH).to(DEVICE)

    # 使用 thop.profile
    # G-FLOPs (Giga Floating Point Operations)
    # 1e9 = 1,000,000,000 (G)
    # verbose=False 可以禁止 thop 打印每一层的详细信息
    flops, params_thop = profile(model, inputs=(dummy_input,), verbose=False)
    g_flops = flops / 1e9

    print(f"FLOPs (G): {g_flops:.2f}G")
    # thop 也会返回参数量，可以用来交叉验证
    # print(f"(thop calculated Params (M): {params_thop / 1e6:.2f}M)")
    print("---" * 15 + "\n")
    # ==================================================================

    # Loss 函数
    loss_fn = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # (可选) 学习率调度器
    # 我们根据 前景IoU 来调整学习率，前景IoU 越高越好，所以 mode='max'
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=3, factor=0.1, verbose=True
    )

    # --- 4. 训练循环 ---
    best_iou_fg = -1.0  # 用于保存最佳模型
    best_dice_fg = -1.0
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        # 训练
        train_fn(train_loader, model, optimizer, loss_fn, DEVICE)


        val_loss, IoU_foreground, Dice_foreground, _, _ = calculate_metrics_and_loss(
            val_loader, model, loss_fn, DEVICE, num_cls=NUM_CLASSES
        )

        print(f"\nValidation Results:")
        print(f"  Avg Loss: {val_loss:.4f}")
        print(f"  IoU (Foreground):     {IoU_foreground:.4f}")
        print(f"  Dice (Foreground):    {Dice_foreground:.4f}")


        scheduler.step(IoU_foreground)

        if IoU_foreground > best_iou_fg:
            best_iou_fg = IoU_foreground
            print(f"==> New best model found! Foreground IoU: {IoU_foreground:.4f}")
            torch.save(model.state_dict(), SAVE_PATH)

            # 2. 检查 Dice 并执行早停逻辑
            # (这替换了你原来 'if Dice_foreground > best_dice_fg:' 的简单判断)
            if Dice_foreground > best_dice_fg:
                best_dice_fg = Dice_foreground
                print(f"==> New best Dice found! Foreground Dice: {Dice_foreground:.4f}")
                # Dice 提升了，重置早停计数器
                early_stop_counter = 0
            else:
                # Dice 没有提升，计数器+1
                early_stop_counter += 1
                print(f"Early stopping counter: {early_stop_counter} / {early_stop_patience}")

            # 3. 检查是否触发早停
            if early_stop_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered: Dice score did not improve for {early_stop_patience} epochs.")
                break  # 中断训练循环

    print("\nTraining finished.")
    print(f"Best validation Foreground IoU: {best_iou_fg:.4f}")
    print(f"Best validation Foreground Dice: {best_dice_fg:.4f}")
    print(f"Best model saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()