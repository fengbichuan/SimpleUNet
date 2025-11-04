import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import matplotlib.pyplot as plt  # [新增] 导入 matplotlib

from thop import profile
from tqdm import tqdm

# 导入你本地的文件
from ReSimpleUNet import SimpleUNet
from dataload import get_loaders
from metrics import calculate_metrics_and_loss, CombinedLoss

# --- 1. 配置参数 ---
# !! 修改为你自己的路径 !!
DATA_PATH = r"D:\对照试验模型\dataset\9-isic2018"

# 超参数
LEARNING_RATE = 0.0003
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 300
NUM_WORKERS = 0
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
NUM_CLASSES = 1  # <-- 修改点: 二分类 (BCE) 模式下, 输出通道为 1
SAVE_PATH = "MBRC+Converse2D[16,16,16,16,16]3"
early_stop_patience = 300
early_stop_counter = 0
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
        targets = targets.to(device=device)  # targets 形状现在是 [B, 1, H, W], float

        # 1. 前向传播
        predictions = model(data)  # predictions 形状 [B, 1, H, W]

        # 2. 计算Loss
        # BCEWithLogitsLoss 需要 (N, C, H, W) 和 (N, C, H, W)
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

    return avg_loss  # [修改] 返回平均训练损失


# [新增] 保存结果到 CSV 文件的函数 (代码不变)
def save_results_to_csv(filepath, header, data_rows):
    """
    将结果列表保存到 CSV 文件
    :param filepath: CSV 文件的保存路径
    :param header: CSV 文件的表头 (list of strings)
    :param data_rows: 包含指标数据的列表 (list of lists)
    """
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(header)
            # 写入所有数据行
            writer.writerows(data_rows)
        print(f"Metrics successfully saved to {filepath}")
    except Exception as e:
        print(f"Error saving CSV to {filepath}: {e}")


# [新增] 保存训练曲线图的函数
def save_plots(results_list, save_path_prefix):
    """
    根据 results_list 绘制 Loss, IoU, Dice 曲线并保存为一张图片。
    :param results_list: 包含指标数据的列表 (list of lists)
    :param save_path_prefix: 图片的保存路径前缀 (e.g., "model_name")
    """
    try:
        # 1. 从 results_list 中解压数据
        # 格式: [epoch + 1, train_loss, val_loss, IoU_foreground, Dice_foreground]
        epochs = [row[0] for row in results_list]
        train_losses = [row[1] for row in results_list]
        val_losses = [row[2] for row in results_list]
        val_ious = [row[3] for row in results_list]
        val_dices = [row[4] for row in results_list]

        # 2. 创建一个包含两个子图的画布 (2行1列)
        # fig 是整个画布, (ax1, ax2) 是两个子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        fig.suptitle('Training Metrics', fontsize=16)

        # 3. 绘制第一个子图：Loss 曲线
        ax1.plot(epochs, train_losses, 'b-o', label='Train Loss')
        ax1.plot(epochs, val_losses, 'r-o', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # 4. 绘制第二个子图：Metrics (IoU & Dice) 曲线
        ax2.plot(epochs, val_ious, 'g-s', label='Validation IoU (FG)')
        ax2.plot(epochs, val_dices, 'm-^', label='Validation Dice (FG)')
        ax2.set_title('Validation Metrics (IoU & Dice)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        # (可选) 设置 Y 轴范围，使曲线更清晰
        # ax2.set_ylim([0.5, 1.0])
        ax2.legend()
        ax2.grid(True)

        # 5. 调整布局并保存图片
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局，为大标题留出空间

        # 定义保存路径
        plot_save_path = f"{save_path_prefix}_metrics_plot.png"

        plt.savefig(plot_save_path)
        plt.close(fig)  # 关闭画布，释放内存
        print(f"Metrics plot successfully saved to {plot_save_path}")

    except Exception as e:
        print(f"Error saving plots: {e}")


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
    model = SimpleUNet(
        in_channels=3,
        num_cls=NUM_CLASSES,  # <-- 修改点: 传入 num_cls=1
        stage_channels=stage_channels,
        num_blocks=[1, 1, 1, 1, 1],
        short_rate=0.5,
        ks_psf=13
        # adw=True
    ).to(DEVICE)

    # <--- 2. Params 和 FLOPs 计算 (这部分不受影响) ---
    print("\n" + "---" * 15)
    print("Calculating model parameters and FLOPs...")
    trainable_params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total Trainable Params (M): {trainable_params_m:.2f}M")
    dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH).to(DEVICE)
    flops, params_thop = profile(model, inputs=(dummy_input,), verbose=False)
    g_flops = flops / 1e9
    print(f"FLOPs (G): {g_flops:.2f}G")
    print("---" * 15 + "\n")
    # ==================================================================

    # Loss 函数
    loss_fn = CombinedLoss(weight_bce=0.5, weight_dice=0.5)  # <-- 修改点: 更换损失函数

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # (可选) 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=5, factor=0.1, verbose=True
    )

    # --- 4. 训练循环 ---
    best_iou_fg = -1.0
    best_dice_fg = -1.0

    # [新增] 用于存储 CSV 结果的列表和表头 (代码不变)
    results_list = []
    csv_header = ["Epoch", "Train Loss", "Val Loss", "Val IoU (FG)", "Val Dice (FG)"]
    # [新增] 定义 CSV 文件的保存路径 (基于模型保存路径) (代码不变)
    csv_save_path = f"{SAVE_PATH}.csv"

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        # 训练 [修改] 接收 train_loss
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, DEVICE)

        # <-- 修改点: 新的评估函数不再需要 num_cls，且返回的直接是前景指标
        # [修改] 调用评估函数，并确保将返回值转换为 Python float
        # 这一步是关键，确保后续的 plotting 和 CSV 保存都能正常进行
        val_loss_tensor, iou_fg_tensor, dice_fg_tensor = calculate_metrics_and_loss(
            val_loader, model, loss_fn, DEVICE
        )
        # 将可能的 Tensor 转换为 Python float。如果已经是 float，.item() 会报错，
        # 所以我们做个判断。但更安全的做法是直接在 metrics.py 中处理。
        # 这里为了兼容性，我们直接假设它们可能是 Tensor，并尝试转成float。
        val_loss = val_loss_tensor.item() if isinstance(val_loss_tensor, torch.Tensor) else float(val_loss_tensor)
        IoU_foreground = iou_fg_tensor.item() if isinstance(iou_fg_tensor, torch.Tensor) else float(iou_fg_tensor)
        Dice_foreground = dice_fg_tensor.item() if isinstance(dice_fg_tensor, torch.Tensor) else float(dice_fg_tensor)

        print(f"\nValidation Results:")
        print(f"  Avg Loss: {val_loss:.4f}")
        print(f"  IoU (Foreground):     {IoU_foreground:.4f}")
        print(f"  Dice (Foreground):    {Dice_foreground:.4f}")

        # [新增] 将本轮次的结果添加到列表中 (代码不变)
        # [修改] 将本轮次的结果添加到列表中
        # 使用 .item() 将 Tensor 转换为 Python 标量 (float)，以供 matplotlib 绘图
        epoch_data = [
            epoch + 1,
            train_loss,
            val_loss,  # <-- 去掉 .item()
            IoU_foreground,  # <-- 去掉 .item()
            Dice_foreground  # <-- 去掉 .item()
        ]
        results_list.append(epoch_data)
        # 1. (可选) 更新学习率调度器 (基于 Dice)
        scheduler.step(Dice_foreground)

        # 2. (可选) 追踪最佳 IoU (仅用于最后打印)
        if IoU_foreground > best_iou_fg:
            best_iou_fg = IoU_foreground
            print(f"==> New best Foreground IoU: {IoU_foreground:.4f}")

        # 3. 修正后的早停和模型保存逻辑 (基于 Dice)
        if Dice_foreground > best_dice_fg:
            best_dice_fg = Dice_foreground
            print(f"==> New best Dice found! Saving model... Foreground Dice: {Dice_foreground:.4f}")
            torch.save(model.state_dict(), SAVE_PATH)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter} / {early_stop_patience}")

        # 4. 检查是否触发早停
        if early_stop_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered: Dice score did not improve for {early_stop_patience} epochs.")
            break

    print("\nTraining finished.")
    print(f"Best validation Foreground IoU: {best_iou_fg:.4f}")
    print(f"Best validation Foreground Dice: {best_dice_fg:.4f}")
    print(f"Best model saved to {SAVE_PATH}")

    # [新增] 在训练结束后，调用函数保存 CSV
    save_results_to_csv(csv_save_path, csv_header, results_list)

    # [新增] 在训练结束后，调用函数保存绘图
    # 我们使用 SAVE_PATH ("MBRC+Converse2D[...]") 作为图片文件名的前缀
    if results_list:  # 确保列表不为空
        save_plots(results_list, SAVE_PATH)
    else:
        print("No results to plot.")


if __name__ == "__main__":
    main()