import torch
import torch.nn.functional as F
from tqdm import tqdm


def calculate_metrics_and_loss(loader, model, loss_fn, device):  # <-- 修改点: 签名移除 num_cls
    """
    在验证集上评估模型，计算 Loss, 前景IoU 和 前景Dice
    (BCE 专用版本)
    """
    model.eval()  # 切换到评估模式
    loop = tqdm(loader, desc="Validation", leave=False)

    running_loss = 0.0
    smooth = 1e-6  # 防止除以0

    # <-- 修改点: 我们只累加前景指标，不再需要按类别
    total_intersection = 0.0
    total_union = 0.0
    total_dice_sum = 0.0  # (Dice的分母 P_sum + T_sum)

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=device)
            targets = targets.to(device=device)  # [B, 1, H, W], float

            # 1. 前向传播
            predictions = model(data)  # [B, 1, H, W], logits

            # 2. 计算Loss
            loss = loss_fn(predictions, targets)
            running_loss += loss.item()

            # 3. 计算指标
            # <-- 修改点: BCE 的指标计算逻辑
            # (1) Sigmoid: 将 logits -> 概率 (0~1)
            preds_probs = torch.sigmoid(predictions)
            # (2) Threshold: 将 概率 -> 标签 (0.0或1.0)
            preds_labels = (preds_probs > 0.5).float()

            # (3) 计算交集、并集 (targets 已经是 [B, 1, H, W], float)
            # .sum() 会在所有维度上求和
            intersection = (preds_labels * targets).sum()
            p_sum = preds_labels.sum()
            t_sum = targets.sum()

            union = (p_sum + t_sum) - intersection
            dice_sum = p_sum + t_sum

            # (4) 累加
            total_intersection += intersection
            total_union += union
            total_dice_sum += dice_sum

    # --- 最终计算 ---

    # IoU (Foreground)
    iou_foreground = (total_intersection + smooth) / (total_union + smooth)

    # Dice (Foreground)
    dice_foreground = (2. * total_intersection + smooth) / (total_dice_sum + smooth)

    # Avg Loss
    avg_loss = running_loss / len(loader)

    # 切换回训练模式
    model.train()

    # <-- 修改点: 返回值
    return avg_loss, iou_foreground, dice_foreground