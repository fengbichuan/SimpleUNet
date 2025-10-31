import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from thop import profile


# ----------------------------------------------------------------------
# 0A. 您新增的 HalfConv 模块 (已添加)
# ----------------------------------------------------------------------
class HalfConv(nn.Module):
    """
    HalfConv
    - 将通道按 1/n_div : (1-1/n_div) 划分，仅对前一部分做 3×3 卷积，剩余通道保持不变；
    - 计算更省、保留部分原始特征。
    Inputs : x ∈ (B, C, H, W)
    Outputs: y ∈ (B, C, H, W)
    """

    def __init__(self, dim: int, n_div: int = 2):
        super().__init__()
        # (!!! 注意 !!!)
        # 增加一个n_div=1的保险，此时它等同于一个标准Conv2d
        if n_div == 1:
            self.dim_conv3 = dim
            self.dim_untouched = 0
        else:
            assert dim % n_div == 0, f"dim ({dim}) 必须能被 n_div ({n_div}) 整除"
            self.dim_conv3 = dim // n_div
            self.dim_untouched = dim - self.dim_conv3

        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.dim_untouched == 0:
            return self.partial_conv3(x)

        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), dim=1)


# ----------------------------------------------------------------------
# 0B. 您提供的 Converse2D 模块 (不变)
# ----------------------------------------------------------------------
class Converse2D(nn.Module):
    """
    Converse2D: 频域闭式解型上采样-去卷积算子（深度可分）
    用途：图像复原/超分重构。先将输入做 s-fold 上采样，再在频域中解耦点扩散核 PSF 的影响，
          通过 OTF(=FFT(psf)) 与先验项（可学习偏置）得到闭式解近似。
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
# 1. 基础卷积块 (!!! 已修改 !!!)
#    - 添加 use_half_conv 参数
# ----------------------------------------------------------------------
class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, dilation=1, use_half_conv=False, n_div=2):
        super().__init__()

        conv_layer = None

        if use_half_conv:
            # HalfConv 的约束条件
            assert in_channels == out_channels, "HalfConv 要求 in_channels == out_channels"
            assert kernel_size == 3, "HalfConv 仅支持 3x3 卷积"
            assert dilation == 1, "HalfConv 不支持 dilation"
            assert pad == 1, "HalfConv 内部 padding=1"

            # 使用 HalfConv 替换标准 Conv2d
            conv_layer = HalfConv(dim=in_channels, n_div=n_div)
        else:
            # 保持原始的标准卷积
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, dilation=dilation,
                                   bias=False)

        self.single_conv = nn.Sequential(
            conv_layer,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)


# ----------------------------------------------------------------------
# 2. 编码器 (下采样) 模块 (!!! 已修改 !!!)
#    - 在 C->C 卷积中启用 use_half_conv=True
# ----------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels, stage_channels, num_blocks, ks, pad, dilation):
        super().__init__()
        self.depth = len(stage_channels)
        self.down = nn.MaxPool2d(2)

        # C_in -> C_0 (通道变化, 不使用 HalfConv)
        self.en_layer0 = SingleConv(in_channels, stage_channels[0], ks, pad, dilation, use_half_conv=False)

        self.en_layers = nn.ModuleList()
        for i in range(1, self.depth):
            layers = []

            # C_i-1 -> C_i-1 (通道不变, 使用 HalfConv)
            for _ in range(num_blocks[i] - 1):
                layers.append(
                    SingleConv(stage_channels[i - 1], stage_channels[i - 1], ks, pad, dilation, use_half_conv=True))

            # C_i-1 -> C_i (通道变化, 不使用 HalfConv)
            layers.append(SingleConv(stage_channels[i - 1], stage_channels[i], ks, pad, dilation, use_half_conv=False))
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
# 3. 跳跃连接处理模块 (不变)
# ----------------------------------------------------------------------
class SkipConnections(nn.Module):
    def __init__(self, stage_channels, short_rate):
        super().__init__()
        self.depth = len(stage_channels)
        self.short_layers = nn.ModuleList()
        for i in range(self.depth - 1):
            layer = SingleConv(stage_channels[i], int(short_rate * stage_channels[i]), 1, 0, 1)
            self.short_layers.append(layer)

    def forward(self, shortcuts):
        refined_shortcuts = []
        for i in range(len(shortcuts)):
            refined = self.short_layers[i](shortcuts[i])
            refined_shortcuts.append(refined)
        return refined_shortcuts


# ----------------------------------------------------------------------
# 4. 瓶颈层 (Bottleneck) 模块 (!!! 已修改 !!!)
#    - 在 C->C 卷积中启用 use_half_conv=True
# ----------------------------------------------------------------------
class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, num_blocks, ks, pad, dilation):
        super().__init__()
        layers = []

        # C_in -> C_mid (通道变化, 不使用 HalfConv)
        layers.append(SingleConv(in_channels, mid_channels, ks, pad, dilation, use_half_conv=False))

        # C_mid -> C_mid (通道不变, 使用 HalfConv)
        for _ in range(num_blocks - 1):
            layers.append(SingleConv(mid_channels, mid_channels, ks, pad, dilation, use_half_conv=True))

        self.bottleneck_convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.bottleneck_convs(x)


# ----------------------------------------------------------------------
# 5. 解码器 (上采样) 模块 (!!! 已修改 !!!)
#    - 在 C->C 卷积中启用 use_half_conv=True
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

        for i in range(self.depth - 1):  # 循环 (depth-1) 次

            # 1. 添加 Converse2D 上采样层
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

            # 2. 创建解码器卷积块
            de_layers_block = []

            in_ch_concat = up_channels[i] + int(short_rate * re_stage_channels[i + 1])
            out_ch = int(short_rate * re_stage_channels[i + 1])

            # C_concat -> C_out (通道变化, 不使用 HalfConv)
            de_layers_block.append(SingleConv(in_ch_concat, out_ch, ks, pad, dilation, use_half_conv=False))

            # C_out -> C_out (通道不变, 使用 HalfConv)
            for _ in range(re_num_blocks[i + 1] - 1):
                de_layers_block.append(SingleConv(out_ch, out_ch, ks, pad, dilation, use_half_conv=True))

            self.de_layers.append(nn.Sequential(*de_layers_block))

    def forward(self, x_from_bottleneck, refined_shortcuts):
        re_shortcuts = refined_shortcuts[::-1]
        x = x_from_bottleneck

        for j in range(self.depth - 1):
            x_up = self.up_layers[j](x)
            shortcut = re_shortcuts[j]
            y = torch.concat([shortcut, x_up], dim=1)
            x = self.de_layers[j](y)

        return x


# ----------------------------------------------------------------------
# 6. 重构后的 SimpleUNet (主模块) (不变)
# ----------------------------------------------------------------------
class SimpleUNet(nn.Module):
    def __init__(self, in_channels, num_cls, ks=3, dilation=1, stage_channels=5 * [32], num_blocks=5 * [1],
                 short_rate=0.5):
        super(SimpleUNet, self).__init__()
        assert short_rate > 0, 'short_rate must be greater than 0!'
        assert len(stage_channels) == len(num_blocks), 'The length of stage_channels and num_blocks must match!'

        self.pad = dilation * (ks - 1) // 2

        self.encoder = Encoder(in_channels, stage_channels, num_blocks, ks, self.pad, dilation)
        self.skip_connections = SkipConnections(stage_channels, short_rate)
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
        input_tensor = torch.randn(1, 3, 256, 256).cuda()

        print("--- Testing Model with num_blocks = 1 (HalfConv NOT used) ---")
        model_1 = SimpleUNet(
            in_channels=3,
            num_cls=1,
            ks=3,
            stage_channels=[16, 32, 64, 128, 256],
            num_blocks=[1, 1, 1, 1, 1],  # num_blocks=1, 不会触发 HalfConv
            short_rate=0.5
        ).cuda()

        flops_1, params_1 = profile(model_1, inputs=(input_tensor,))
        output_1 = model_1(input_tensor)

        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output_1.shape}")
        print(f"FLOPs (G): {flops_1 / 1e9:.4f}")
        print(f"Params (M): {params_1 / 1e6:.4f}")

        # ---

        print("\n--- Testing Model with num_blocks = 2 (HalfConv IS used) ---")
        # (确保通道数为偶数以满足 HalfConv n_div=2 的要求)
        model_2 = SimpleUNet(
            in_channels=3,
            num_cls=1,
            ks=3,
            stage_channels=[16, 32, 64, 128, 256],
            num_blocks=[2, 2, 2, 2, 2],  # num_blocks=2, 将触发 HalfConv
            short_rate=0.5
        ).cuda()

        flops_2, params_2 = profile(model_2, inputs=(input_tensor,))
        output_2 = model_2(input_tensor)

        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output_2.shape}")
        print(f"FLOPs (G): {flops_2 / 1e9:.4f}")
        print(f"Params (M): {params_2 / 1e6:.4f}")

        # 比较
        print("\n--- Comparison ---")
        print(f"Params (M) [blocks=2 vs blocks=1]: {params_2 / 1e6:.4f} vs {params_1 / 1e6:.4f}")
        print(f"FLOPs (G) [blocks=2 vs blocks=1]: {flops_2 / 1e9:.4f} vs {flops_1 / 1e9:.4f}")
        print("\n(FLOPs 和 Params 增加，因为层数从1增加到2，但 HalfConv 使得增幅小于使用标准 Conv)")

    else:
        print("CUDA not available. Please run this on a machine with a GPU.")