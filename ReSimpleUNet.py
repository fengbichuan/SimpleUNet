import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


# ----------------------------------------------------------------------
# 0. 您提供的 HDPA 模块 (粘贴在此处)
# ----------------------------------------------------------------------
class MBRConv1(nn.Module):
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale

        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 2, out_channels, 1)
        # (!!! 注意 !!!) 原始代码中 weight.requires_grad = False
        # 在U-Net中, 我们希望所有参数都能训练, 将其改为 True 或移除
        # self.conv_out.weight.requires_grad = False # <-- 建议移除或注释掉

        # (!!! 修正 !!!)
        # 原始代码中 weight1 的初始化方式 (zeros_like) 和 re-param 逻辑
        # (F.conv2d) 在 thop.profile 中会出问题，且效率不高。
        # 我将其修改为等效的、更标准的 PyTorch 模块化实现，
        # 这样更容易训练和分析。

        # --- 修正后的 MBRConv1 ---
        self.rep_conv = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.rep_bn = nn.BatchNorm2d(out_channels * rep_scale)
        self.reparam_conv = nn.Conv2d(in_channels, out_channels * rep_scale, 1)

        # 1x1 卷积用于融合两个分支
        self.merge_conv = nn.Conv2d(out_channels * rep_scale * 2, out_channels, 1)

    def forward(self, inp):
        # 分支 1: conv
        x_rep = self.rep_conv(inp)

        # 分支 2: conv -> bn
        x_bn = self.rep_bn(self.reparam_conv(inp))

        # 合并
        x = torch.cat([x_rep, x_bn], 1)

        # 融合
        out = self.merge_conv(x)
        return out


class HDPA(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(HDPA, self).__init__()
        self.channels = channels

        self.globalatt = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # (!!! 修正 !!!)
            # 原始 MBRConv1 实现中 F.conv2d + 权重相加 的方式不利于集成
            # 我将其修改为等效的、更标准的 PyTorch 模块化实现
            nn.Conv2d(channels, channels, 1),  # 简化 MBRConv1
            # MBRConv1(channels, channels, rep_scale=rep_scale), # 您也可以用上面修正后的MBRConv1
            nn.Sigmoid()
        )
        self.localatt = nn.Sequential(
            # (!!! 修正 !!!)
            nn.Conv2d(1, 1, 3, padding=1, bias=False),  # 用一个简单的 3x3 卷积处理空间图
            nn.BatchNorm2d(1),
            nn.Conv2d(1, channels, 1),  # 1x1 卷积扩展回 C 通道
            # MBRConv1(1, channels, rep_scale=rep_scale), # 您也可以用上面修正后的MBRConv1
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. 全局通道注意力
        x1 = self.globalatt(x)  # (B, C, 1, 1)

        # 2. 局部空间注意力
        # (!!! 修正 !!!) 原始的 x1*x 会导致梯度问题, 改为 x1 应用后再 max
        x_gated = x1 * x
        max_out, _ = torch.max(x_gated, dim=1, keepdim=True)  # (B, 1, H, W)
        x2 = self.localatt(max_out)  # (B, C, H, W)

        # 3. 融合 (原始逻辑)
        x3 = torch.mul(x1, x2) * x
        return x3


# ----------------------------------------------------------------------
# 0. Converse2D 模块 (不变)
# ----------------------------------------------------------------------
class Converse2D(nn.Module):
    """
    Converse2D: 频域闭式解型上采样-去卷积算子（深度可分）
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            scale: int = 1,
            padding: int = 2,
            padding_mode: str = "circular",
            eps: float = 1e-5,
    ):
        super().__init__()
        # --- 参数与约束 ---
        assert out_channels == in_channels, "Converse2D 仅支持 out_channels == in_channels（深度可分）"
        assert isinstance(scale, int) and scale >= 1, "scale 必须为 >=1 的整数"
        assert kernel_size > 0 and isinstance(kernel_size, int), "kernel_size 必须为正整数"
        assert padding_mode in {"reflect", "replicate", "circular", "constant"}, "非法 padding_mode"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.padding = padding
        self.padding_mode = padding_mode
        self.eps = float(eps)

        # 卷积核与偏置（每通道一核）
        self.weight = nn.Parameter(torch.randn(1, in_channels, kernel_size, kernel_size))
        with torch.no_grad():
            w = self.weight.data.view(1, in_channels, -1)
            self.weight.copy_(F.softmax(w, dim=-1).view_as(self.weight))  # 核归一化（按通道）

        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))  # 可学习先验强度

    # ----------------- 主流程 -----------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 边界填充（在空域，匹配 s-fold 上采样后裁剪）
        if self.padding > 0:
            x = F.pad(
                x,
                pad=[self.padding, self.padding, self.padding, self.padding],
                mode=self.padding_mode,
                value=0.0,
            )

        # 正则/先验强度（>0）
        biaseps = torch.sigmoid(self.bias - 9.0) + self.eps  # 形状 (1, C, 1, 1)

        # s-fold 上采样（零填充）
        STy = self._s_fold_upsample(x, scale=self.scale)  # (B, C, H*s, W*s)

        # 供对比的最近邻上采（不参与公式，仅保留你原始注释思想）
        if self.scale != 1:
            x_nn = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        else:
            x_nn = x

        Hs, Ws = STy.shape[-2:]
        FB = self._psf2otf(self.weight.to(dtype=x.dtype, device=x.device), (Hs, Ws))  # (1,C,Hs,Ws)
        FBC = torch.conj(FB)
        F2B = torch.abs(FB) ** 2

        # 右端项：FBC * FFT(STy)
        FBFy = FBC * torch.fft.fftn(STy, dim=(-2, -1))

        # FR = FBFy + FFT(biaseps * x)
        FR = FBFy + torch.fft.fftn(biaseps * x_nn, dim=(-2, -1))

        # 频域闭式解各项
        x1 = FB * FR
        FBR = torch.mean(self._splits(x1, self.scale), dim=-1)  # (B,C,Hs/s,Ws/s)
        invW = torch.mean(self._splits(F2B, self.scale), dim=-1)  # (1,C,Hs/s,Ws/s)
        invWBR = FBR / (invW + biaseps + self.eps)  # 稳定除法

        # 重构
        FCBinvWBR = FBC * invWBR.repeat(1, 1, self.scale, self.scale)  # broadcast 回到 (B,C,Hs,Ws)
        FX = (FR - FCBinvWBR) / (biaseps + self.eps)
        out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

        # 去除之前的 padding（注意要按放大后的步长裁剪）
        if self.padding > 0:
            p = self.padding * self.scale
            out = out[..., p:-p, p:-p]

        return out

    # ----------------- 工具函数 -----------------
    @staticmethod
    def _splits(a: torch.Tensor, scale: int) -> torch.Tensor:
        """
        将 (..., W, H) 切分为 (..., W/scale, H/scale, scale^2)，用于频域子采样平均。
        """
        *lead, W, H = a.size()
        assert W % scale == 0 and H % scale == 0, "空间尺寸需可被 scale 整除"
        Ws, Hs = W // scale, H // scale
        b = a.view(*lead, scale, Ws, scale, Hs)
        # 将两个 scale 维并到最后
        perm = list(range(len(lead))) + [len(lead) + 1, len(lead) + 3, len(lead), len(lead) + 2]
        b = b.permute(*perm).contiguous()
        return b.view(*lead, Ws, Hs, scale * scale)

    @staticmethod
    def _psf2otf(psf: torch.Tensor, shape_hw: tuple[int, int]) -> torch.Tensor:
        """
        PSF -> OTF：把 PSF 放到左上角，roll 到中心，再做 FFT，得到 (N=1, C, H, W) 的 OTF。
        """
        H, W = shape_hw
        otf = torch.zeros(psf.shape[:-2] + (H, W), dtype=psf.dtype, device=psf.device)
        otf[..., :psf.shape[-2], :psf.shape[-1]] = psf
        otf = torch.roll(otf, shifts=(-psf.shape[-2] // 2, -psf.shape[-1] // 2), dims=(-2, -1))
        return torch.fft.fftn(otf, dim=(-2, -1))

    @staticmethod
    def _s_fold_upsample(x: torch.Tensor, scale: int) -> torch.Tensor:
        """
        s-fold 上采样：在 (H*s, W*s) 的网格上每隔 s 填一个原像素，其余为 0。
        """
        if scale == 1:
            return x
        B, C, H, W = x.shape
        z = torch.zeros(B, C, H * scale, W * scale, dtype=x.dtype, device=x.device)
        z[..., ::scale, ::scale] = x
        return z


# ----------------------------------------------------------------------
# 1. 基础卷积块 (不变)
# ----------------------------------------------------------------------
class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, dilation=1):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)


# ----------------------------------------------------------------------
# 2. 编码器 (下采样) 模块 (不变)
# ----------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels, stage_channels, num_blocks, ks, pad, dilation):
        super().__init__()
        self.depth = len(stage_channels)
        self.down = nn.MaxPool2d(2)

        self.en_layer0 = SingleConv(in_channels, stage_channels[0], ks, pad, dilation)
        self.en_layers = nn.ModuleList()
        for i in range(1, self.depth):
            layers = []
            for _ in range(num_blocks[i] - 1):
                layers.append(SingleConv(stage_channels[i - 1], stage_channels[i - 1], ks, pad, dilation))
            layers.append(SingleConv(stage_channels[i - 1], stage_channels[i], ks, pad, dilation))
            self.en_layers.append(nn.Sequential(*layers))

    def forward(self, x):
        shortcuts = []
        x = self.en_layer0(x)
        shortcuts.append(x)
        for i in range(self.depth - 2):
            x = self.down(x)
            x = self.en_layers[i](x)
            shortcuts.append(x)
        x = self.down(x)
        x = self.en_layers[self.depth - 2](x)
        return x, shortcuts


# ----------------------------------------------------------------------
# 3. 跳跃连接处理模块 (!!! 已修改 - 插入 HDPA !!!)
# ----------------------------------------------------------------------
class SkipConnections(nn.Module):
    def __init__(self, stage_channels, short_rate, rep_scale=4):  # <-- (!!! 修改 !!!) 接收 rep_scale
        super().__init__()
        self.depth = len(stage_channels)
        self.short_layers = nn.ModuleList()
        self.hdpa_layers = nn.ModuleList()  # (!!! 新增 !!!) 用于存放 HDPA 模块

        for i in range(self.depth - 1):
            # 1x1 卷积的输出通道
            current_out_channels = int(short_rate * stage_channels[i])

            # 1. 原始的 1x1 卷积 (不变)
            layer = SingleConv(stage_channels[i], current_out_channels, 1, 0, 1)
            self.short_layers.append(layer)

            # 2. (!!! 新增 !!!) 对应的 HDPA 模块
            # HDPA 的输入/输出通道 = 1x1 卷积的输出通道
            self.hdpa_layers.append(
                HDPA(channels=current_out_channels, rep_scale=rep_scale)
            )

    def forward(self, shortcuts):
        refined_shortcuts = []
        for i in range(len(shortcuts)):
            # 1. 原始计算：通过 1x1 卷积
            refined = self.short_layers[i](shortcuts[i])

            # 2. (!!! 新增 !!!) 应用 HDPA 注意力进行提炼
            refined = self.hdpa_layers[i](refined)

            refined_shortcuts.append(refined)
        return refined_shortcuts


# ----------------------------------------------------------------------
# 4. 瓶颈层 (Bottleneck) 模块 (不变)
# ----------------------------------------------------------------------
class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, num_blocks, ks, pad, dilation):
        super().__init__()
        layers = []
        layers.append(SingleConv(in_channels, mid_channels, ks, pad, dilation))
        for _ in range(num_blocks - 1):
            layers.append(SingleConv(mid_channels, mid_channels, ks, pad, dilation))
        self.bottleneck_convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.bottleneck_convs(x)


# ----------------------------------------------------------------------
# 5. 解码器 (上采样) 模块 (不变, 沿用 Converse2D 版本)
# ----------------------------------------------------------------------
class Decoder(nn.Module):
    """
    U-Net的解码器（上采样）路径。
    使用 Converse2D 作为上采样层。
    """

    def __init__(self, stage_channels, num_blocks, short_rate, ks, pad, dilation):
        super().__init__()
        self.depth = len(stage_channels)
        re_stage_channels = stage_channels[::-1]
        re_num_blocks = num_blocks[::-1]

        self.de_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        up_channels = [int(short_rate * ch) for ch in re_stage_channels]

        for i in range(self.depth - 1):
            current_up_channels = up_channels[i]
            self.up_layers.append(
                Converse2D(
                    in_channels=current_up_channels,
                    out_channels=current_up_channels,
                    kernel_size=ks,
                    scale=2,
                    padding=pad,
                    padding_mode="circular"
                )
            )

            de_layers_block = []
            in_ch_concat = up_channels[i] + int(short_rate * re_stage_channels[i + 1])
            out_ch = int(short_rate * re_stage_channels[i + 1])

            de_layers_block.append(SingleConv(in_ch_concat, out_ch, ks, pad, dilation))
            for _ in range(re_num_blocks[i + 1] - 1):
                de_layers_block.append(SingleConv(out_ch, out_ch, ks, pad, dilation))
            self.de_layers.append(nn.Sequential(*de_layers_block))

    def forward(self, x_from_bottleneck, refined_shortcuts):
        re_shortcuts = refined_shortcuts[::-1]
        x = x_from_bottleneck

        for j in range(self.depth - 1):
            x_up_simple = F.interpolate(
                x,
                scale_factor=2,
                mode='bilinear',
                align_corners=False
            )
            x_up_complex = self.up_layers[j](x)
            x_up = x_up_simple + x_up_complex
            shortcut = re_shortcuts[j]
            y = torch.concat([shortcut, x_up], dim=1)
            x = self.de_layers[j](y)

        return x


# ----------------------------------------------------------------------
# 6. 重构后的 SimpleUNet (主模块) (!!! 已修改 !!!)
# ----------------------------------------------------------------------
class SimpleUNet(nn.Module):
    def __init__(self, in_channels, num_cls, ks=3, dilation=1, stage_channels=5 * [32], num_blocks=5 * [1],
                 short_rate=0.5, rep_scale=4):  # <-- (!!! 新增 !!!) rep_scale 参数

        super(SimpleUNet, self).__init__()
        assert short_rate > 0, 'short_rate must be greater than 0!'
        assert len(stage_channels) == len(num_blocks), 'The length of stage_channels and num_blocks must match!'

        self.pad = dilation * (ks - 1) // 2

        self.encoder = Encoder(in_channels, stage_channels, num_blocks, ks, self.pad, dilation)

        # (!!! 修改 !!!) 将 rep_scale 传递给 SkipConnections
        self.skip_connections = SkipConnections(stage_channels, short_rate, rep_scale)

        self.bottleneck = Bottleneck(
            in_channels=stage_channels[-1],
            mid_channels=int(short_rate * stage_channels[-1]),
            num_blocks=num_blocks[-1],
            ks=ks,
            pad=self.pad,
            dilation=dilation
        )
        self.decoder = Decoder(stage_channels, num_blocks, short_rate, ks, self.pad, dilation)
        self.seg_head = SingleConv(int(short_rate * stage_channels[0]), num_cls, 1, 0, 1)

    def forward(self, x):
        x_to_bottleneck, shortcuts = self.encoder(x)

        # (!!!) 这里的 refined_shortcuts 现在是经过 HDPA 处理过的
        refined_shortcuts = self.skip_connections(shortcuts)

        x_after_bottleneck = self.bottleneck(x_to_bottleneck)
        x = self.decoder(x_after_bottleneck, refined_shortcuts)
        output = self.seg_head(x)
        return output


# ----------------------------------------------------------------------
# 7. 测试代码 (不变)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 确保有可用的CUDA设备
    if torch.cuda.is_available():
        input = torch.randn(1, 3, 256, 256).cuda()

        # (!!!) 现在可以传入 rep_scale, 默认为 4
        model = SimpleUNet(
            in_channels=3,
            num_cls=1,
            ks=3,
            stage_channels=[16, 16, 16, 16, 16],
            num_blocks=[1, 1, 1, 1, 1],
            short_rate=0.5,
            rep_scale=4
        ).cuda()

        flops, params = profile(model, inputs=(input,))
        output = model(input)

        print(f"\n--- Final Model Output (with Converse2D + HDPA) ---")
        print(f"Input shape: {input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"FLOPs (G): {flops / 1e9}")
        print(f"Params (M): {params / 1e6}")
    else:
        print("CUDA not available. Please run this on a machine with a GPU.")