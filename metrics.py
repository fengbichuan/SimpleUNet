# metrics.py

import torch
import torch.nn.functional as F
from tqdm import tqdm


def calculate_metrics_and_loss(loader, model, loss_fn, device, num_cls):
    """
    在验证集上评估模型，计算 Loss, mIoU 和 mDice
    """
    model.eval()  # 切换到评估模式
    loop = tqdm(loader, desc="Validation", leave=False)

    running_loss = 0.0
    smooth = 1e-6  # 防止除以0

    # 累加整个数据集的指标
    total_intersection = torch.zeros(num_cls).to(device)
    total_union = torch.zeros(num_cls).to(device)
    total_dice_sum = torch.zeros(num_cls).to(device)  # Dice 的分母 (P + T)

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=device)
            targets = targets.to(device=device)  # [B, H, W]

            # 1. 前向传播
            predictions = model(data)  # [B, C, H, W]

            # 2. 计算Loss
            loss = loss_fn(predictions, targets)
            running_loss += loss.item()

            # 3. 计算指标
            # 将模型的 logits 输出转换为预测类别 [B, H, W]
            preds_labels = torch.argmax(predictions, dim=1)

            # 转换为 One-Hot 编码 [B, C, H, W]
            preds_one_hot = F.one_hot(preds_labels, num_cls).permute(0, 3, 1, 2).float()
            targets_one_hot = F.one_hot(targets, num_cls).permute(0, 3, 1, 2).float()

            # --- 计算 mIoU ---
            # 批量计算交集 (Intersection)
            intersection = (preds_one_hot * targets_one_hot).sum(dim=[0, 2, 3])  # [C]
            # 批量计算并集 (Union)
            p_sum = preds_one_hot.sum(dim=[0, 2, 3])  # [C]
            t_sum = targets_one_hot.sum(dim=[0, 2, 3])  # [C]
            union = (p_sum + t_sum) - intersection

            total_intersection += intersection
            total_union += union

            # --- 计算 mDice ---
            # Dice 的分母是 p_sum + t_sum
            total_dice_sum += (p_sum + t_sum)

    # --- 最终计算 ---

    # mIoU (Mean Intersection over Union)
    iou_per_class = (total_intersection + smooth) / (total_union + smooth)
    mIoU = iou_per_class.mean()

    # mDice (Mean Dice Coefficient)
    dice_per_class = (2. * total_intersection + smooth) / (total_dice_sum + smooth)
    mDice = dice_per_class.mean()

    # Avg Loss
    avg_loss = running_loss / len(loader)

    # 切换回训练模式
    model.train()

    return avg_loss, mIoU, mDice, iou_per_class, dice_per_class