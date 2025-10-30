import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveCoordAtt(nn.Module):
    """
    Adaptive Coordinate Attention
    - 沿 H、W 两方向自适应池化得到 (H×1) 与 (1×W) 描述；拼接后经共享 MLP，再拆分为 H/W 两支并映射回 C；
    - 生成 a_h, a_w 的逐通道权重后相加，对输入做逐通道加权；
    - alpha 用于调节注意力强度（进入映射前的缩放因子）。
    Inputs : x ∈ (B, C, H, W)
    Outputs: y ∈ (B, C, H, W)
    """
    def __init__(self, in_channels: int, reduction: int = 16, alpha: float = 0.9):
        super().__init__()
        self.in_channels = in_channels
        self.reduction   = reduction
        self.mid_channels = max(8, in_channels // reduction)  # 瓶颈通道
        self.alpha = alpha

        # 共享 MLP（1×1 Conv + BN + ReLU）
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True)
        )
        # 分别映射回 C
        self.conv_h = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()

        # 沿两个方向的全局平均池化
        x_h = F.adaptive_avg_pool2d(x, (h, 1))   # (B,C,H,1)
        x_w = F.adaptive_avg_pool2d(x, (1, w))   # (B,C,1,W)
        x_w = x_w.permute(0, 1, 3, 2)            # (B,C,W,1)

        # 拼接后经共享 MLP
        y = torch.cat([x_h, x_w], dim=2)         # (B,C,H+W,1)
        y = self.shared_conv(y)                   # (B,mid,H+W,1)

        # 再拆为 H/W 两支
        y_h, y_w = torch.split(y, [h, w], dim=2) # y_h:(B,mid,H,1), y_w:(B,mid,W,1)
        y_w = y_w.permute(0, 1, 3, 2)            # (B,mid,1,W)

        # 生成注意力并缩放到 (0,1)
        a_h = self.conv_h(y_h * self.alpha).sigmoid()  # (B,C,H,1)
        a_w = self.conv_w(y_w * self.alpha).sigmoid()  # (B,C,1,W)

        # 融合并加权
        out = x * (a_h + a_w)
        return out


if __name__ == "__main__":
    # ===== 示例（统一格式）=====
    # 1) 配置
    B, H, W, C = 2, 64, 64, 32
    reduction = 16
    alpha = 0.9

    # 2) 构造输入：形状 (B, C, H, W)
    x = torch.randn(B, C, H, W)

    # 3) 实例化模块（统一命名为 block）
    block = AdaptiveCoordAtt(in_channels=C, reduction=reduction, alpha=alpha)

    # 4) 前向计算（不计算梯度）
    with torch.no_grad():
        y = block(x)

    # 5) 打印结构与张量形状
    print(block)
    print("x.shape =", x.shape)  # [B, C, H, W]
    print("y.shape =", y.shape)  # [B, C, H, W]
