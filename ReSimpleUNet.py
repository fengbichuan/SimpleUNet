import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


# ----------------------------------------------------------------------
# 0A. 您提供的 Converse2D 模块 (不变)
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
# 0B. (!!! 新增 !!!) 您提供的 DynamicSpatialAttention 模块
# ----------------------------------------------------------------------
class DynamicSpatialAttention(nn.Module):
    """
    Dynamic Spatial Attention (DSA)
    - 思路：为每个样本生成一个专属的 2D 卷积核（由全局通道描述生成），
      用该核去卷积通道平均图，得到单通道空间注意力图，再对输入逐像素加权。
    """

    def __init__(self, in_channels: int, kernel_size: int = 3):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric padding"
        self.kernel_size = kernel_size

        # 共享的核生成器：先提全局通道描述（GAP），再两层 1×1 生成 k*k 个权重
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, 1, 1)
            nn.Conv2d(in_channels, in_channels, 1),  # (B, C, 1, 1)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, kernel_size ** 2, 1)  # (B, k*k, 1, 1)
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1) 生成每个样本的动态卷积核  (B, k*k, 1, 1) -> (B, 1, k, k)
        kernels = self.kernel_generator(x).view(B, 1, self.kernel_size, self.kernel_size)

        # 2) 对输入在通道维求平均，得到单通道图 (B, 1, H, W)
        x_mean = x.mean(dim=1, keepdim=True)

        # 3) 重排为 grouped conv 的输入格式
        #    输入:  (1, B, H, W), 卷积核: (B, 1, k, k), groups=B  => 每个样本仅与自己的核卷积
        x_mean_group = x_mean.view(1, B, H, W)

        # 4) 进行样本级动态卷积，得到注意力响应 (1, B, H, W) -> (B, 1, H, W)
        att = F.conv2d(
            x_mean_group,
            weight=kernels,
            padding=self.kernel_size // 2,
            groups=B
        ).view(B, 1, H, W)

        # 5) Sigmoid 归一化并施加到原特征
        att = self.act(att)
        return x * att


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
        # 注意：Encoder 的 shortcuts 列表包含 (depth-1) 个元素
        # 分别来自 stage_channels[0] 到 stage_channels[depth-2]
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
        # 创建 (depth-1) 个 1x1 卷积
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
# 5. 解码器 (上采样) 模块 (!!! 已修改 !!!)
# ----------------------------------------------------------------------
class Decoder(nn.Module):
    """
    U-Net的解码器（上采样）路径。
    使用 Converse2D 作为上采样层。
    (!!! 修改 !!!) 使用 DynamicSpatialAttention (DSA) 过滤跳跃连接。
    """

    def __init__(self, stage_channels, num_blocks, short_rate, ks, pad, dilation):
        super().__init__()

        self.depth = len(stage_channels)
        re_stage_channels = stage_channels[::-1]
        re_num_blocks = num_blocks[::-1]

        # 1. 解码器卷积层 (de_layer0, de_layer1, ...)
        self.de_layers = nn.ModuleList()

        # 2. Converse2D 上采样层
        self.up_layers = nn.ModuleList()

        # (!!! 新增 !!!)
        # 3. DSA 模块层，用于过滤跳跃连接
        self.dsa_skip_layers = nn.ModuleList()

        # 计算上采样层的输入通道 (来自上一层解码器或瓶颈层)
        up_channels = [int(short_rate * ch) for ch in re_stage_channels]

        # 计算跳跃连接的通道 (来自 SkipConnections 模块)
        # re_stage_channels = [sc[4], sc[3], sc[2], sc[1], sc[0]]
        # 我们需要的 skip 通道是 [sc[3]*r, sc[2]*r, sc[1]*r, sc[0]*r]
        # 对应 re_stage_channels[i+1] * short_rate
        skip_channels = [int(short_rate * re_stage_channels[i + 1]) for i in range(self.depth - 1)]

        for i in range(self.depth - 1):  # 循环 (depth-1) 次

            # (!!! A. 添加 Converse2D 上采样层 !!!)
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

            # (!!! B. 新增：添加 DSA 注意力层 !!!)
            # DSA 的 in_channels 必须匹配跳跃连接的通道数
            current_skip_channels = skip_channels[i]
            self.dsa_skip_layers.append(
                DynamicSpatialAttention(
                    in_channels=current_skip_channels,
                    kernel_size=ks  # 复用 ks=3
                )
            )

            # (!!! C. 添加解码器卷积块 !!!)
            de_layers_block = []

            # 拼接后的输入通道 = (来自上采样的) + (来自跳跃连接的)
            in_ch_concat = current_up_channels + current_skip_channels
            out_ch = int(short_rate * re_stage_channels[i + 1])  # 即 up_channels[i+1]

            de_layers_block.append(SingleConv(in_ch_concat, out_ch, ks, pad, dilation))

            for _ in range(re_num_blocks[i + 1] - 1):  # 使用 re_num_blocks[i+1]
                de_layers_block.append(SingleConv(out_ch, out_ch, ks, pad, dilation))

            self.de_layers.append(nn.Sequential(*de_layers_block))

    def forward(self, x_from_bottleneck, refined_shortcuts):
        # 将跳跃连接反转，以便从深到浅使用
        # 对应通道: [skip_channels[0], skip_channels[1], ...]
        re_shortcuts = refined_shortcuts[::-1]

        x = x_from_bottleneck  # 从瓶颈层的输出开始

        # 循环上采样
        for j in range(self.depth - 1):  # 循环 (depth-1) 次

            # 1. 使用 Converse2D 进行上采样
            x_up = self.up_layers[j](x)

            # 2. 获取对应的跳跃连接
            shortcut = re_shortcuts[j]

            # (!!! 新增 !!!)
            # 2.5 使用 DSA 模块“过滤”跳跃连接
            shortcut_attended = self.dsa_skip_layers[j](shortcut)

            # (!!! 修改 !!!)
            # 3. 标准U-Net融合：拼接 (使用过滤后的 shortcut)
            y = torch.concat([shortcut_attended, x_up], dim=1)

            # 4. 通过解码器卷积块
            x = self.de_layers[j](y)

        return x  # 返回解码器最后一层的输出


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
        # Decoder 的初始化调用不变
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
        input = torch.randn(1, 3, 256, 256).cuda()

        # 使用的参数 (ks=3, pad=1)
        model = SimpleUNet(
            in_channels=3,
            num_cls=1,
            ks=3,  # (!!!) 将被传递给 Converse2D 和 DynamicSpatialAttention
            stage_channels=[16, 16, 16, 16, 16],
            num_blocks=[1, 1, 1, 1, 1],
            short_rate=0.5
        ).cuda()

        flops, params = profile(model, inputs=(input,))
        output = model(input)

        print(f"\n--- Final Model Output (with Converse2D + DSA on Skips) ---")
        print(f"Input shape: {input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"FLOPs (G): {flops / 1e9}")
        print(f"Params (M): {params / 1e6}")
    else:
        print("CUDA not available. Please run this on a machine with a GPU.")