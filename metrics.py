import torch
import torch.nn.functional as F
from tqdm import tqdm


def calculate_metrics_and_loss(loader, model, loss_fn, device, num_cls):
    """
    在验证集上评估模型，计算 Loss, IoU 和 Dice (特指前景类)
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
            # 你的代码假设 num_cls >= 2 (例如: 0=背景, 1=前景)
            # 并且使用 CrossEntropyLoss 类型的损失
            preds_labels = torch.argmax(predictions, dim=1)

            # 转换为 One-Hot 编码 [B, C, H, W]
            preds_one_hot = F.one_hot(preds_labels, num_cls).permute(0, 3, 1, 2).float()
            targets_one_hot = F.one_hot(targets, num_cls).permute(0, 3, 1, 2).float()

            # --- 计算 IoU ---
            # 批量计算交集 (Intersection)
            intersection = (preds_one_hot * targets_one_hot).sum(dim=[0, 2, 3])  # [C]
            # 批量计算并集 (Union)
            p_sum = preds_one_hot.sum(dim=[0, 2, 3])  # [C]
            t_sum = targets_one_hot.sum(dim=[0, 2, 3])  # [C]
            union = (p_sum + t_sum) - intersection

            total_intersection += intersection
            total_union += union

            # --- 计算 Dice ---
            # Dice 的分母是 p_sum + t_sum
            total_dice_sum += (p_sum + t_sum)

    # --- 最终计算 ---

    # IoU (Intersection over Union) - 每个类别
    iou_per_class = (total_intersection + smooth) / (total_union + smooth)

    # Dice (Dice Coefficient) - 每个类别
    dice_per_class = (2. * total_intersection + smooth) / (total_dice_sum + smooth)

    # --- [!! 修改点 !!] ---
    # 假设: Class 0 是背景, Class 1 是前景
    # 我们只返回前景的指标，这对应论文中的 IoU^i 和 DSC^i

    # 检查 num_cls 是否至少为2，否则索引[1]会出错
    if num_cls > 1:
        IoU_foreground = iou_per_class[1]
        Dice_foreground = dice_per_class[1]
    else:
        # 如果 num_cls=1 (不推荐用于此脚本)，则默认返回索引0
        IoU_foreground = iou_per_class[0]
        Dice_foreground = dice_per_class[0]

    # Avg Loss
    avg_loss = running_loss / len(loader)

    # 切换回训练模式
    model.train()

    # 返回前景的 IoU 和 Dice，同时也返回所有类的列表供调试
    return avg_loss, IoU_foreground, Dice_foreground, iou_per_class, dice_per_class