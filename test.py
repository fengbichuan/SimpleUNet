import torch
import torch.nn as nn
from tqdm import tqdm

# 导入你本地的文件
from ReSimpleUNet import SimpleUNet
from dataload import get_loaders
from metrics import calculate_metrics_and_loss

# --- 1. 配置参数 ---
# !! 确保这些参数与你 train.py 中的设置完全一致 !!

# 数据路径
DATA_PATH = r"D:\对照试验模型\dataset\9-isic2018"
# 模型权重路径
MODEL_PATH = ".\Experience-isic2018\VSA+Converse2D[64,128,256,512,1024]"

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 评估时 Batch Size 可以适当调大 (如果显存允许)
BATCH_SIZE = 8
NUM_WORKERS = 4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
stage_channels = [64, 128, 256, 512, 1024]
# [!! 关键修改 !!]
# 既然 metrics 是 BCE 专用版，模型输出通道必须是 1
NUM_CLASSES = 1


def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading model weights from: {MODEL_PATH}")

    # --- 2. Dataloader ---
    # 我们只需要验证集加载器 (val_loader)
    try:
        _, val_loader = get_loaders(
            DATA_PATH,
            BATCH_SIZE,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            NUM_WORKERS,
            PIN_MEMORY
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        print("请确保 'dataload.py' 和数据路径 'DATA_PATH' 配置正确。")
        # 警告：确保 dataload.py 返回的 mask 是 [B, 1, H, W] 且类型为 float
        print("警告: 你的 metrics.py 是BCE版, 请确保 dataload.py 返回的 mask 是 [B, 1, H, W] 且类型为 float。")
        return

    print("Validation data loaded.")

    # --- 3. 初始化模型 ---
    # [!! 关键修改 !!] num_cls 必须与 BCE 逻辑匹配，改为 1
    model = SimpleUNet(
        in_channels=3,
        num_cls=NUM_CLASSES,  # <--- 必须是 1
        stage_channels=stage_channels,
        num_blocks=[1, 1, 1, 1, 1],
        short_rate=0.5,
        #adw=True  # <--- 确保这里与 train.py 一致
    ).to(DEVICE)

    # --- 警告：关于模型加载 ---
    # 你使用了 strict=False，这可能会隐藏模型架构不匹配的问题。
    # 请100%确保你这里定义的 SimpleUNet 参数 (特别是 adw=True)
    # 与你训练并保存 MODEL_PATH 时用的参数 *完全一致*。
    print("Warning: Using strict=False to load model. Ensure architecture matches *exactly*.")

    # --- 4. 加载训练好的权重 ---
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)

    # --- 5. 定义 Loss (metrics 函数需要它) ---
    # [!! 关键修改 !!] 必须使用BCE Loss来匹配 metrics.py 的逻辑
    loss_fn = nn.BCEWithLogitsLoss()

    # --- 6. 运行评估 ---
    print("\nRunning final evaluation on the validation set...")

    # [!! 关键修改 !!]
    # 1. 移除 num_cls 参数
    # 2. 只接收 3 个返回值
    val_loss, IoU_foreground, Dice_foreground = calculate_metrics_and_loss(
        val_loader, model, loss_fn, DEVICE
    )

    # --- 7. 打印最终结果 ---
    # (这部分无需修改，因为它只打印了前景指标)
    print("\n--- Final Test Results ---")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Validation Loss: {val_loss:.4f}")
    print("-----------------------------")

    print(f"  Foreground IoU (Class 1 - 病灶):  {IoU_foreground:.4f}")
    print(f"  Foreground Dice (Class 1 - 病灶): {Dice_foreground:.4f}")


if __name__ == "__main__":
    main()