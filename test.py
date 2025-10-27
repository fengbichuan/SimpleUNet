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
MODEL_PATH = "isic-2018-[64,128,256,512,1024]"

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 评估时 Batch Size 可以适当调大 (如果显存允许)
BATCH_SIZE = 16
NUM_WORKERS = 4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
NUM_CLASSES = 2  # 0: 背景, 1: 病灶


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
        return

    print("Validation data loaded.")

    # --- 3. 初始化模型 ---
    # [!! 修改点 !!] 必须使用与训练时完全相同的模型参数
    # train.py 中 adw=True 是注释掉的，这里也必须注释掉
    model = SimpleUNet(
        in_channels=3,
        num_cls=NUM_CLASSES,
        stage_channels=[64, 128, 256, 512, 1024],
        num_blocks=[1, 1, 1, 1, 1],
        short_rate=0.5,
        #adw=True  # <--- 确保这里与 train.py 一致
    ).to(DEVICE)

    # --- 4. 加载训练好的权重 ---

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE),strict=False)


    # --- 5. 定义 Loss (metrics 函数需要它) ---
    loss_fn = nn.CrossEntropyLoss()

    # --- 6. 运行评估 ---
    print("\nRunning final evaluation on the validation set...")

    # [!! 修改点 !!] 更改接收的变量名
    # 调用 metrics.py 中的函数
    val_loss, IoU_foreground, Dice_foreground, iou_cls, dice_cls = calculate_metrics_and_loss(
        val_loader, model, loss_fn, DEVICE, num_cls=NUM_CLASSES
    )

    # --- 7. 打印最终结果 ---
    # [!! 修改点 !!] 更新打印的指标
    print("\n--- Final Test Results ---")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Validation Loss: {val_loss:.4f}")
    print("-----------------------------")

    print(f"  Foreground IoU (Class 1 - 病灶):  {IoU_foreground:.4f}")
    print(f"  Foreground Dice (Class 1 - 病灶): {Dice_foreground:.4f}")


if __name__ == "__main__":
    main()

#baseline
# # test.py
#
# import torch
# import torch.nn as nn
# from tqdm import tqdm
#
# # 导入你本地的文件
# from SimpleUNet import SimpleUNet
# from dataload import get_loaders
# from metrics import calculate_metrics_and_loss
#
# # --- 1. 配置参数 ---
# # !! 确保这些参数与你 train.py 中的设置完全一致 !!
#
# # 数据路径
# DATA_PATH = r"D:\对照试验模型\dataset\9-isic2018"
# # 模型权重路径
# MODEL_PATH = "baseline.pth"
#
# # 设备
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # 评估时 Batch Size 可以适当调大 (如果显存允许)
# BATCH_SIZE = 16
# NUM_WORKERS = 4
# IMAGE_HEIGHT = 256
# IMAGE_WIDTH = 256
# PIN_MEMORY = True
# NUM_CLASSES = 2  # 0: 背景, 1: 病灶
#
#
# def main():
#     print(f"Using device: {DEVICE}")
#     print(f"Loading model weights from: {MODEL_PATH}")
#
#     # --- 2. Dataloader ---
#     # 我们只需要验证集加载器 (val_loader)
#     # get_loaders 会返回 (train_loader, val_loader)
#     # 我们用 _ 来忽略 train_loader
#     try:
#         _, val_loader = get_loaders(
#             DATA_PATH,
#             BATCH_SIZE,
#             IMAGE_HEIGHT,
#             IMAGE_WIDTH,
#             NUM_WORKERS,
#             PIN_MEMORY
#         )
#     except Exception as e:
#         print(f"Error loading data: {e}")
#         print("请确保 'dataload.py' 和数据路径 'DATA_PATH' 配置正确。")
#         return
#
#     print("Validation data loaded.")
#
#     # --- 3. 初始化模型 ---
#     # 必须使用与训练时完全相同的模型参数
#     model = SimpleUNet(
#         in_channels=3,
#         num_cls=NUM_CLASSES,
#         stage_channels=[16, 16, 16, 16, 16],
#         num_blocks=[1, 1, 1, 1, 1],
#         short_rate=0.5,
#         adw=True
#     ).to(DEVICE)
#
#     # --- 4. 加载训练好的权重 ---
#     try:
#         model.load_state_dict(
#             torch.load(MODEL_PATH, map_location=DEVICE)
#         )
#     except FileNotFoundError:
#         print(f"错误: 找不到模型权重文件 '{MODEL_PATH}'")
#         print("请先运行 train.py 来生成 '[16,32,64,128,256]SimpleUnet.pth' 文件。")
#         return
#     except Exception as e:
#         print(f"加载模型权重时出错: {e}")
#         print("请确保 'SimpleUNet' 中的模型结构与 '[16,32,64,128,256]SimpleUnet.pth' 匹配。")
#         return
#
#     # --- 5. 定义 Loss (metrics 函数需要它) ---
#     loss_fn = nn.CrossEntropyLoss()
#
#     # --- 6. 运行评估 ---
#     print("\nRunning final evaluation on the validation set...")
#
#     # 调用 metrics.py 中的函数
#     # 它会自动设置 model.eval() 并返回指标
#     val_loss, mIoU, mDice, iou_cls, dice_cls = calculate_metrics_and_loss(
#         val_loader, model, loss_fn, DEVICE, num_cls=NUM_CLASSES
#     )
#
#     # --- 7. 打印最终结果 ---
#     print("\n--- Final Test Results ---")
#     print(f"  Model: {MODEL_PATH}")
#     print(f"  Validation Loss: {val_loss:.4f}")
#     print("-----------------------------")
#     print(f"  Mean IoU (mIoU): {mIoU:.4f}")
#     print(f"  Mean Dice (mDice): {mDice:.4f}")
#     print("-----------------------------")
#     print(f"  IoU - Class 0 (背景):     {iou_cls[0]:.4f}")
#     print(f"  IoU - Class 1 (病灶): {iou_cls[1]:.4f}")
#     print("-----------------------------")
#     print(f"  Dice - Class 0 (背景):    {dice_cls[0]:.4f}")
#     print(f"  Dice - Class 1 (病灶):{dice_cls[1]:.4f}")
#     print("-----------------------------")
#
#
# if __name__ == "__main__":
#     main()
