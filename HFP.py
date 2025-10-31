import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as DCT

"""
High-Frequency Perception (HFP)
- 目的：在频域抑制低频背景、突出小目标的边缘/纹理；再以通道/空间路径生成注意力并融合。
- 关键点：DCT 高通掩码 → iDCT 回到空域；通道路径做逐通道统计，空间路径做逐位置掩码。
- 形状：输入/输出均为 (B, C, H, W)。
"""

# ---------------- Spatial path with DCT ----------------
class DctSpatialInteraction(nn.Module):
    def __init__(self, in_channels: int, ratio=(0.25, 0.25), isdct: bool = True):
        super().__init__()
        self.ratio = ratio
        self.isdct = isdct
        if not isdct:
            self.spatial1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

    def _compute_weight(self, h, w, ratio):
        """构造高通权重：左上 (低频区) 置 0，其余为 1。"""
        h0, w0 = int(h * ratio[0]), int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))

        x_dct = DCT.dct_2d(x, norm='ortho')                        # (B,C,H,W)
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight = weight.view(1, 1, h, w).expand_as(x_dct)          # 修复：补齐通道维再广播
        x_dct = x_dct * weight                                     # 高通
        mask = DCT.idct_2d(x_dct, norm='ortho')                    # (B,C,H,W)
        return x * mask


# ---------------- Channel path with DCT ----------------
class DctChannelInteraction(nn.Module):
    def __init__(self, in_channels: int, patch=(8, 8), ratio=(0.25, 0.25), isdct: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.h, self.w = patch
        self.ratio = ratio
        self.isdct = isdct

        # 逐通道 1×1（groups=32 可按需改为 gcd(in_channels, 32)）
        g = 32
        self.channel1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=g)
        self.channel2x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=g)
        self.relu = nn.ReLU()

    def _compute_weight(self, h, w, ratio):
        h0, w0 = int(h * ratio[0]), int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        if not self.isdct:
            amaxp = F.adaptive_max_pool2d(x, (1, 1))
            aavgp = F.adaptive_avg_pool2d(x, (1, 1))
            ch = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))
            return x * torch.sigmoid(self.channel2x1(ch))

        x_dct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight = weight.view(1, 1, h, w).expand_as(x_dct)          # 修复：补齐通道维再广播
        x_hp = DCT.idct_2d(x_dct * weight, norm='ortho')           # 高频增强后的空域特征

        # 逐通道统计（patch 汇聚再求和）
        amaxp = F.adaptive_max_pool2d(x_hp, (self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(x_hp, (self.h, self.w))
        amaxp = torch.sum(self.relu(amaxp), dim=(2, 3), keepdim=True)  # (B,C,1,1)
        aavgp = torch.sum(self.relu(aavgp), dim=(2, 3), keepdim=True)  # (B,C,1,1)

        ch = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        return x * torch.sigmoid(self.channel2x1(ch))


# ---------------- HFP wrapper ----------------
class High_Frequency_Perception_Module(nn.Module):
    """
    高频感知模块：空间(位置) × 频域高通 + 通道(类别) × 频域统计，最后 3×3 + GN 输出。
    """
    def __init__(self, in_channels: int, ratio=(0.25, 0.25), patch=(8, 8), isdct: bool = True):
        super().__init__()
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct=isdct)
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct=isdct)
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.spatial(x)
        c = self.channel(x)
        return self.out(s + c)


if __name__ == "__main__":
    # ===== 示例（统一格式）=====
    # 1) 配置
    B, H, W, C = 1, 50, 50, 32
    ratio = (0.25, 0.25)
    patch = (8, 8)
    isdct = True

    # 2) 构造输入：形状 (B, C, H, W)
    x = torch.randn(B, C, H, W)

    # 3) 实例化模块（统一命名为 block）
    block = High_Frequency_Perception_Module(in_channels=C, ratio=ratio, patch=patch, isdct=isdct)

    # 4) 前向计算（不计算梯度）
    with torch.no_grad():
        y = block(x)

    # 5) 打印结构与张量形状
    print(block)
    print("x.shape =", x.shape)  # [B, C, H, W]
    print("y.shape =", y.shape)  # [B, C, H, W]
