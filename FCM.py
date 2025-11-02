import torch
import torch.nn as nn



def calculate_padding(kernel_size, padding=None, dilation=1):
    """Same padding 计算（含 dilation 支持）"""
    if dilation > 1:
        if isinstance(kernel_size, int):
            kernel_size = dilation * (kernel_size - 1) + 1
        else:
            kernel_size = [dilation * (x - 1) + 1 for x in kernel_size]
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]
    return padding


class ConvolutionLayer(nn.Module):
    """Conv-BN-Act 基础层（可选融合 BN）"""
    default_activation = nn.SiLU()

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None,
                 groups=1, dilation=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            calculate_padding(kernel_size, padding, dilation),
            groups=groups, dilation=dilation, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = self.default_activation if activation is True else \
            (activation if isinstance(activation, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv(x)))

    def forward_fused(self, x):
        return self.activation(self.conv(x))


class ChannelAttention(nn.Module):
    """通道注意力：DW-Conv → GAP → Sigmoid，输出 (B,C,1,1)"""
    def __init__(self, channels):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.global_pooling(x)
        return self.sigmoid(x)


class SpatialAttention(nn.Module):
    """空间注意力：1×1 Conv → BN → Sigmoid，输出 (B,1,H,W)"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return self.sigmoid(x)


class FCM(nn.Module):
    """
    特征互补映射模块
    - 通道按 3/4:1/4 划分为主/子分支；主分支提语义，子分支保细节；双向注意力后相加融合。
    Inputs:  x ∈ (B, C, H, W)
    Outputs: y ∈ (B, C, H, W)
    """
    def __init__(self, channels):
        super().__init__()
        self.main_channels = channels - channels // 4  # 3/4 C
        self.sub_channels  = channels // 4             # 1/4 C

        # 主分支（语义）：3×3 → 3×3 → 1×1
        self.main_branch_conv1 = ConvolutionLayer(self.main_channels, self.main_channels, kernel_size=3, stride=1, padding=1)
        self.main_branch_conv2 = ConvolutionLayer(self.main_channels, self.main_channels, kernel_size=3, stride=1, padding=1)
        self.main_branch_conv3 = ConvolutionLayer(self.main_channels, channels,         kernel_size=1, stride=1)

        # 子分支（空间/细节）：1×1
        self.sub_branch_conv   = ConvolutionLayer(self.sub_channels,  channels,         kernel_size=1, stride=1)

        # 注意力
        self.spatial_attention = SpatialAttention(channels)  # (B,1,H,W)
        self.channel_attention = ChannelAttention(channels)  # (B,C,1,1)

    def forward(self, x):
        # 通道划分
        main_feat, sub_feat = torch.split(x, [self.main_channels, self.sub_channels], dim=1)

        # 主分支（语义）
        m = self.main_branch_conv1(main_feat)
        m = self.main_branch_conv2(m)
        m = self.main_branch_conv3(m)  # (B,C,H,W)

        # 子分支（空间）
        s = self.sub_branch_conv(sub_feat)  # (B,C,H,W)

        # 双向引导融合
        fused_spatial = self.spatial_attention(s) * m   # (B,1,H,W) ⊙ (B,C,H,W)
        fused_channel = self.channel_attention(m) * s   # (B,C,1,1) ⊙ (B,C,H,W)
        return fused_spatial + fused_channel


if __name__ == "__main__":
    # ===== 示例（统一格式）=====
    # 1) 配置
    B, H, W, C = 1, 50, 50, 32

    # 2) 构造输入：形状 (B, C, H, W)
    x = torch.randn(B, C, H, W)

    # 3) 实例化模块（统一命名为 block）
    block = FCM(channels=C)

    # 4) 前向计算（不计算梯度）
    with torch.no_grad():
        y = block(x)

    # 5) 打印结构与张量形状
    print(block)
    print("x.shape =", x.shape)  # [B, C, H, W]
    print("y.shape =", y.shape)  # [B, C, H, W]
