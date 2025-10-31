import torch
from torch import nn
import torch.nn.functional as F

class SPR_SA(nn.Module):
    """
    Spatial-Perceptive Reweighting (SPR_SA)
    - DW(3×3, groups=C) + PW(1×1) 提升到 hidden_dim；
    - GAP→Softmax 生成通道注意力（逐通道，空间 1×1），再对特征重标定；
    - GELU + 1×1 回落到原通道数。
    Inputs : x ∈ (B, C, H, W)
    Outputs: y ∈ (B, C, H, W)
    """
    def __init__(self, dim: int, growth_rate: float = 2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        # 先分组卷积(≈深度卷积)，再逐点卷积到 hidden_dim
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_0(x)                         # (B, hidden, H, W)
        x1 = F.adaptive_avg_pool2d(x, (1, 1))      # (B, hidden, 1, 1)
        x1 = F.softmax(x1, dim=1)                  # 通道注意力
        x  = x * x1                                # 重标定
        x  = self.act(x)
        x  = self.conv_1(x)                        # 回落到 dim
        return x


if __name__ == "__main__":
    # ===== 示例（统一格式）=====
    # 1) 配置
    B, H, W, C = 1, 64, 64, 32
    growth_rate = 2.0

    # 2) 构造输入：形状 (B, C, H, W)
    x = torch.randn(B, C, H, W)

    # 3) 实例化模块（统一命名为 block）
    block = SPR_SA(dim=C, growth_rate=growth_rate)

    # 4) 前向计算（不计算梯度）
    with torch.no_grad():
        y = block(x)

    # 5) 打印结构与张量形状
    print(block)
    print("x.shape =", x.shape)  # [B, C, H, W]
    print("y.shape =", y.shape)  # [B, C, H, W]
