import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv  # [新增] 导入 CSV 库

from thop import profile
from tqdm import tqdm

# 导入你本地的文件
# 假设 ReSimpleUNet.py 包含了我们上次修改的、集成了 MANO 的 SimpleUNet
from ReSimpleUNet import SimpleUNet
from dataload import get_loaders
from metrics import calculate_metrics_and_loss

# --- 1. 配置参数 ---
# !! 修改为你自己的路径 !!
DATA_PATH = r"D:\对照试验模型\dataset\9-isic2018"

# 超参数
LEARNING_RATE = 0.0003
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 300
NUM_WORKERS = 4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
NUM_CLASSES = 1
SAVE_PATH = "MANO+Converse2D[64,128,256,512,1024]"
early_stop_patience = 20
early_stop_counter = 0
stage_channels = [64, 128, 256, 512, 1024]

# [!!! NEW !!!]
# ----------------------------------------------------
# MANO Bottleneck Config (与 SimpleUNet __init__ 匹配)
# ----------------------------------------------------
USE_MANO_BOTTLENECK = True  # <-- 设为 True 来启用 MANO
MANO_DIM = 256  # MANO 内部维度 (可以调整, 比如 128, 256)
MANO_DEPTH = 4  # MANO Block 堆叠层数
MANO_HEADS = 4  # MANO 注意力头数
MANO_DIM_HEAD = 64  # MANO 每个头的维度 (MANO_HEADS * MANO_DIM_HEAD = 256, 应该匹配 MANO_DIM)
MANO_ATT_SAMPLING = "conv"  # 下采样方式 ('conv' 或 'avg_pool')
MANO_ATT_SAMPLING_RATE = 4  # MANO 多尺度下采样率 (例如 2 或 4)
MANO_LOCAL_SPAN = 2  # 局部注意力的窗口大小
MANO_LOCAL_STRIDE = 1  # 局部注意力的步幅


# ----------------------------------------------------


def train_fn(loader, model, optimizer, loss_fn, device):
    """
    训练一个 Epoch
    """
    loop = tqdm(loader, desc="Training")
    running_loss = 0.0

    model.train()  # 确保在训练模式

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # 1. 前向传播
        predictions = model(data)

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

    return avg_loss


def save_results_to_csv(filepath, header, data_rows):
    """
    将结果列表保存到 CSV 文件
    """
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_rows)
        print(f"Metrics successfully saved to {filepath}")
    except Exception as e:
        print(f"Error saving CSV to {filepath}: {e}")


def main():
    global early_stop_counter
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

    # [!!! MODIFIED !!!]
    # 在此处实例化 SimpleUNet 时，传入 MANO 所需的新参数
    # ----------------------------------------------------
    model = SimpleUNet(
        in_channels=3,
        num_cls=NUM_CLASSES,

        # (!!! 关键修改 !!!)
        image_size_hw=IMAGE_HEIGHT,  # 传入原始图像尺寸
        device=DEVICE,  # 传入 DEVICE

        # (U-Net 原始参数)
        stage_channels=stage_channels,
        num_blocks=[1, 1, 1, 1, 1],
        short_rate=0.5,
        ks=3,  # 确保 ks=3 (或您需要的值) 被传递

        # (!!! 关键修改 !!!) (传入 MANO 参数)
        use_mano_bottleneck=USE_MANO_BOTTLENECK,
        mano_dim=MANO_DIM,
        mano_depth=MANO_DEPTH,
        mano_heads=MANO_HEADS,
        mano_dim_head=MANO_DIM_HEAD,
        mano_att_sampling=MANO_ATT_SAMPLING,
        mano_att_sampling_rate=MANO_ATT_SAMPLING_RATE,
        mano_local_span=MANO_LOCAL_SPAN,
        mano_local_stride=MANO_LOCAL_STRIDE

    ).to(DEVICE)
    # ----------------------------------------------------

    # <--- 2. Params 和 FLOPs 计算 (这部分不受影响) ---
    print("\n" + "---" * 15)
    print("Calculating model parameters and FLOPs...")
    trainable_params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total Trainable Params (M): {trainable_params_m:.2f}M")
    dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH).to(DEVICE)

    # (注意: 如果 thop 无法处理 MANO 中的某些操作，可能会报错或不准)
    try:
        flops, params_thop = profile(model, inputs=(dummy_input,), verbose=False)
        g_flops = flops / 1e9
        print(f"FLOPs (G) (via thop): {g_flops:.2f}G")
    except Exception as e:
        print(f"Could not calculate FLOPs with thop. Error: {e}")
        print("Continuing without FLOPs calculation...")

    print("---" * 15 + "\n")
    # ==================================================================

    # Loss 函数
    loss_fn = nn.BCEWithLogitsLoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # (可选) 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=3, factor=0.1, verbose=True
    )

    # --- 4. 训练循环 ---
    best_iou_fg = -1.0
    best_dice_fg = -1.0

    results_list = []
    csv_header = ["Epoch", "Train Loss", "Val Loss", "Val IoU (FG)", "Val Dice (FG)"]
    csv_save_path = f"{SAVE_PATH}.csv"

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        train_loss = train_fn(train_loader, model, optimizer, loss_fn, DEVICE)

        val_loss, IoU_foreground, Dice_foreground = calculate_metrics_and_loss(
            val_loader, model, loss_fn, DEVICE
        )

        print(f"\nValidation Results:")
        print(f"  Avg Loss: {val_loss:.4f}")
        print(f"  IoU (Foreground):     {IoU_foreground:.4f}")
        print(f"  Dice (Foreground):    {Dice_foreground:.4f}")

        epoch_data = [epoch + 1, train_loss, val_loss, IoU_foreground, Dice_foreground]
        results_list.append(epoch_data)

        scheduler.step(Dice_foreground)

        if IoU_foreground > best_iou_fg:
            best_iou_fg = IoU_foreground
            print(f"==> New best Foreground IoU: {IoU_foreground:.4f}")

        if Dice_foreground > best_dice_fg:
            best_dice_fg = Dice_foreground
            print(f"==> New best Dice found! Saving model... Foreground Dice: {Dice_foreground:.4f}")
            torch.save(model.state_dict(), SAVE_PATH)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter} / {early_stop_patience}")

        if early_stop_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered: Dice score did not improve for {early_stop_patience} epochs.")
            break

    print("\nTraining finished.")
    print(f"Best validation Foreground IoU: {best_iou_fg:.4f}")
    print(f"Best validation Foreground Dice: {best_dice_fg:.4f}")
    print(f"Best model saved to {SAVE_PATH}")

    save_results_to_csv(csv_save_path, csv_header, results_list)


if __name__ == "__main__":
    main()