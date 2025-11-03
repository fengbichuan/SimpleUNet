import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
# (!!! 新增 !!!)
# 确保你已安装此库: pip install pytorch-wavelets
from pytorch_wavelets import DWTForward


# ----------------------------------------------------------------------
# 0. 您提供的 Converse2D 模块 (不变)
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
# (!!! 新增 !!!)
# 2. 您提供的 RHDWT 模块
# ----------------------------------------------------------------------
class Residual_Haar_Discrete_Wavelet_Transform(nn.Module):
    def __init__(self, in_channels, n=1):
        super(Residual_Haar_Discrete_Wavelet_Transform, self).__init__()
        # 残差路径卷积（stride=2 下采样，padding=1）
        self.identety = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * n,
            kernel_size=3,
            stride=2,
            padding=1
        )

        # Haar 小波变换（单层分解）
        self.DWT = DWTForward(J=1, wave='haar')

        # 小波特征编码
        self.dconv_encode = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * n, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def _transformer(self, DMT1_yl, DMT1_yh):
        """重组低频与三方向高频：输出 [N, 4*C, H/2, W/2]"""
        list_tensor = []
        a = DMT1_yh[0]  # J=1 仅一层
        list_tensor.append(DMT1_yl)
        for i in range(3):
            list_tensor.append(a[:, :, i, :, :])
        return torch.cat(list_tensor, 1)

    def forward(self, x):
        input = x
        # Haar 分解
        DMT1_yl, DMT1_yh = self.DWT(x)  # yl: [N,C,H/2,W/2], yh[0]: [N,C,3,H/2,W/2]
        # 重组 + 编码
        DMT = self._transformer(DMT1_yl, DMT1_yh)
        x = self.dconv_encode(DMT)
        # 残差下采样
        res = self.identety(input)
        # 融合
        out = torch.add(x, res)
        return out


# ----------------------------------------------------------------------
# 3. 编码器 (下采样) 模块 (!!! 已修改 !!!)
# ----------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels, stage_channels, num_blocks, ks, pad, dilation):
        super().__init__()
        self.depth = len(stage_channels)

        # (!!! 已修改 !!!)
        # 移除了 self.down = nn.MaxPool2d(2)
        # self.down = nn.MaxPool2d(2)

        # (!!! 新增 !!!)
        # 创建一个 ModuleList 来存储 (depth-1) 个 RHDWT 下采样层
        self.down_layers = nn.ModuleList()
        for i in range(self.depth - 1):
            # U-Net 的下采样层通常不改变通道数 (MaxPool)，
            # 随后的 en_layers[i] 期望的输入通道是 stage_channels[i]。
            # 因此，我们设置 n=1，使 RHDWT 的输出通道 = 输入通道。
            self.down_layers.append(
                Residual_Haar_Discrete_Wavelet_Transform(
                    in_channels=stage_channels[i],
                    n=1  # (!!!) 关键：保持通道数不变
                )
            )

        # (!!! 不变 !!!)
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

        # (!!! 已修改 !!!)
        # 循环 (depth-2) 次
        for i in range(self.depth - 2):
            # x = self.down(x)  # 替换
            x = self.down_layers[i](x)  # 使用第 i 个 RHDWT 下采样
            x = self.en_layers[i](x)
            shortcuts.append(x)

        # (!!! 已修改 !!!)
        # 最后第 (depth-1) 次下采样
        # x = self.down(x) # 替换
        x = self.down_layers[self.depth - 2](x)  # 使用最后一个 RHDWT 下采样

        # (!!! 不变 !!!)
        x = self.en_layers[self.depth - 2](x)
        return x, shortcuts


# ----------------------------------------------------------------------
# 4. 跳跃连接处理模块 (不变)
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
# 5. 瓶颈层 (Bottleneck) 模块 (不变)
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
# 6. 解码器 (上采样) 模块 (不变, 沿用你的 Converse2D)
# ----------------------------------------------------------------------
class Decoder(nn.Module):
    """
    U-Net的解码器（上采样）路径。
    使用 Converse2D 作为上采样层。
    """

    def __init__(self, stage_channels, num_blocks, short_rate, ks, pad, dilation):
        super().__init__()

        # 移除了 self.up = nn.Upsample(...)

        self.depth = len(stage_channels)
        re_stage_channels = stage_channels[::-1]
        re_num_blocks = num_blocks[::-1]

        # 1. 解码器层 (de_layer0, de_layer1, ...)
        self.de_layers = nn.ModuleList()

        # (!!! 新增 !!!)
        # 2. 创建 (depth-1) 个 Converse2D 上采样层
        # 每一层的通道数 C 必须与 x (来自上一层解码器或瓶颈层) 的通道数匹配
        # 因为 Converse2D 要求 in_channels == out_channels
        self.up_layers = nn.ModuleList()

        # 计算解码器路径中，输入到 *上采样层* 的特征图通道数
        # up_channels[0] = 瓶颈层输出通道数
        # up_channels[1] = 第1个解码块输出通道数
        # ...
        up_channels = [int(short_rate * ch) for ch in re_stage_channels]

        for i in range(self.depth - 1):  # 循环 (depth-1) 次, 创建 (depth-1) 个上采样层

            # (!!! 新增 !!!)
            # 添加 Converse2D 上采样层
            # 通道数 C = up_channels[i]
            # 我们使用与解码器卷积相同的 ks 和 pad
            current_up_channels = up_channels[i]
            self.up_layers.append(
                Converse2D(
                    in_channels=current_up_channels,
                    out_channels=current_up_channels,
                    kernel_size=ks,  # 复用传入的 ks
                    scale=2,  # U-Net 固定的2倍上采样
                    padding=pad,  # 复用传入的 pad
                    padding_mode="circular"  # 沿用 Converse2D 示例中的模式
                )
            )

            # (!!! 原有逻辑: 创建解码器卷积块 !!!)
            # 注意：i 在这里是从 0 开始的 (因为 range(self.depth - 1))
            # 但 re_stage_channels 和 re_num_blocks 的索引需要匹配原始逻辑 (从 1 开始)
            # 因此我们使用 i+1 作为 re_... 的索引, i 作为 up_channels 的索引
            de_layers_block = []

            # 拼接后的输入通道计算保持不变
            # [i]   -> re_stage_channels[i]   (上一层解码器的输出, 即 up_channels[i])
            # [i+1] -> re_stage_channels[i+1] (来自SkipConnection)
            in_ch_concat = up_channels[i] + int(short_rate * re_stage_channels[i + 1])
            out_ch = int(short_rate * re_stage_channels[i + 1])  # (即 up_channels[i+1])

            de_layers_block.append(SingleConv(in_ch_concat, out_ch, ks, pad, dilation))

            for _ in range(re_num_blocks[i + 1] - 1):  # 使用 re_num_blocks[i+1]
                de_layers_block.append(SingleConv(out_ch, out_ch, ks, pad, dilation))

            self.de_layers.append(nn.Sequential(*de_layers_block))

    def forward(self, x_from_bottleneck, refined_shortcuts):
        # 将跳跃连接反转，以便从深到浅使用
        re_shortcuts = refined_shortcuts[::-1]

        x = x_from_bottleneck  # 从瓶颈层的输出开始

        # 循环上采样
        for j in range(self.depth - 1):  # 循环 (depth-1) 次

            # (!!! 已修改 !!!)
            # 1. 使用 Converse2D 进行上采样
            x_up = self.up_layers[j](x)

            # 2. 获取对应的跳跃连接
            shortcut = re_shortcuts[j]

            # 3. 标准U-Net融合：直接拼接
            y = torch.concat([shortcut, x_up], dim=1)

            # 4. 通过解码器卷积块
            x = self.de_layers[j](y)

        return x  # 返回解码器最后一层的输出


# ----------------------------------------------------------------------
# 7. 重构后的 SimpleUNet (主模块) (不变)
# ----------------------------------------------------------------------
class SimpleUNet(nn.Module):
    def __init__(self, in_channels, num_cls, ks=3, dilation=1, stage_channels=5 * [32], num_blocks=5 * [1],
                 short_rate=0.5):
        super(SimpleUNet, self).__init__()
        assert short_rate > 0, 'short_rate must be greater than 0!'
        assert len(stage_channels) == len(num_blocks), 'The length of stage_channels and num_blocks must match!'

        self.pad = dilation * (ks - 1) // 2

        # (!!!) 这里的 Encoder 实例化会自动调用我们修改后的 Encoder
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
        # (!!!) 这里的 Decoder 实例化不变
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
# 8. 测试代码 (不变)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 确保有可用的CUDA设备
    if torch.cuda.is_available():
        input = torch.randn(1, 3, 256, 256).cuda()

        # 使用的参数 (ks=3, pad=1)
        model = SimpleUNet(
            in_channels=3,
            num_cls=1,
            ks=3,  # (!!!) 将被传递给 Converse2D
            stage_channels=[16, 16, 16, 16, 16],  # 编码器各阶段通道
            num_blocks=[1, 1, 1, 1, 1],
            short_rate=0.5
        ).cuda()

        flops, params = profile(model, inputs=(input,))
        output = model(input)

        print(f"\n--- Final Model Output (with RHDWT Downsampling + Converse2D Upsampling) ---")
        print(f"Input shape: {input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"FLOPs (G): {flops / 1e9}")
        print(f"Params (M): {params / 1e6}")
    else:
        print("CUDA not available. Please run this on a machine with a GPU.")

        # (!!!) CPU 测试（用于没有GPU的环境）
        print("--- Running on CPU (for verification) ---")
        input_cpu = torch.randn(1, 3, 256, 256)
        model_cpu = SimpleUNet(
            in_channels=3,
            num_cls=1,
            ks=3,
            stage_channels=[16, 16, 16, 16, 16],
            num_blocks=[1, 1, 1, 1, 1],
            short_rate=0.5
        )

        # 警告：Converse2D 在 CPU 上的 FFT 可能较慢
        with torch.no_grad():
            output_cpu = model_cpu(input_cpu)

        print(f"Input shape (CPU): {input_cpu.shape}")
        print(f"Output shape (CPU): {output_cpu.shape}")
        print("Model runs successfully on CPU.")