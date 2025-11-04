import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


# ----------------------------------------------------------------------
# + 新增: MBRConv5 模块 (结构重参数化) (!!! 按方案A修改 !!!)
# ----------------------------------------------------------------------
class MBRConv5(nn.Module):
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv5, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 5x5 卷积分支
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 5, 1, 2, bias=False)
        self.conv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )

        # 1x1 卷积分支
        self.conv1 = nn.Conv2d(in_channels, out_channels * rep_scale, 1, bias=False)
        self.conv1_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )

        # 3x3 卷积分支
        self.conv2 = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1, bias=False)
        self.conv2_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )

        # 3x1 卷积分支
        self.conv_crossh = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0), bias=False)
        self.conv_crossh_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )

        # 1x3 卷积分支
        self.conv_crossv = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1), bias=False)
        self.conv_crossv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )

        # 1x1 融合层
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 10, out_channels, 1)

        # 重参数化技巧：分离可学习权重和固定权重
        self.conv_out.weight.requires_grad = False
        self.weight1 = nn.Parameter(torch.zeros_like(self.conv_out.weight))
        nn.init.xavier_normal_(self.weight1)

        # (!!!) 方案A：新增 ReLU
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inp):
        x1 = self.conv(inp)
        x2 = self.conv1(inp)
        x3 = self.conv2(inp)
        x4 = self.conv_crossh(inp)
        x5 = self.conv_crossv(inp)

        x = torch.cat(
            [x1, x2, x3, x4, x5,
             self.conv_bn(x1),
             self.conv1_bn(x2),
             self.conv2_bn(x3),
             self.conv_crossh_bn(x4),
             self.conv_crossv_bn(x5)],
            1
        )

        final_weight = self.conv_out.weight + self.weight1
        out = F.conv2d(x, final_weight, self.conv_out.bias)

        # (!!!) 方案A：在末尾应用 ReLU
        return self.relu(out)

    # slim方法：仅在推理时使用
    def slim(self):
        # 辅助函数：融合 Conv 和 BN
        def fuse_conv_bn(conv, bn):
            k = 1 / (bn.running_var + bn.eps) ** .5
            k_unqueezed = k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            weight = conv.weight * k_unqueezed * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            b_bn = - bn.running_mean / (bn.running_var + bn.eps) ** .5
            bias = b_bn * bn.weight + bn.bias

            # 如果 conv 有 bias，需要加上
            if conv.bias is not None:
                bias = bias + conv.bias * k

            return weight, bias

        # 1. 融合所有 (Conv + BN) 分支
        conv_w, conv_b = fuse_conv_bn(self.conv, self.conv_bn[0])
        conv1_w, conv1_b = fuse_conv_bn(self.conv1, self.conv1_bn[0])
        conv2_w, conv2_b = fuse_conv_bn(self.conv2, self.conv2_bn[0])
        conv_crossh_w, conv_crossh_b = fuse_conv_bn(self.conv_crossh, self.conv_crossh_bn[0])
        conv_crossv_w, conv_crossv_b = fuse_conv_bn(self.conv_crossv, self.conv_crossv_bn[0])

        # 2. 融合所有 (仅 Conv) 分支 (模拟一个恒等BN)
        def get_conv_params(conv):
            weight = conv.weight
            bias = torch.zeros(conv.out_channels, device=conv.weight.device)
            return weight, bias

        conv_pre_w, conv_pre_b = get_conv_params(self.conv)
        conv1_pre_w, conv1_pre_b = get_conv_params(self.conv1)
        conv2_pre_w, conv2_pre_b = get_conv_params(self.conv2)
        conv_crossh_pre_w, conv_crossh_pre_b = get_conv_params(self.conv_crossh)
        conv_crossv_pre_w, conv_crossv_pre_b = get_conv_params(self.conv_crossv)

        # 3. 将所有非 5x5 核 Pad 到 5x5
        conv1_w = nn.functional.pad(conv1_w, (2, 2, 2, 2))
        conv2_w = nn.functional.pad(conv2_w, (1, 1, 1, 1))
        conv_crossv_w = nn.functional.pad(conv_crossv_w, (1, 1, 2, 2))
        conv_crossh_w = nn.functional.pad(conv_crossh_w, (2, 2, 1, 1))

        conv1_pre_w = nn.functional.pad(conv1_pre_w, (2, 2, 2, 2))
        conv2_pre_w = nn.functional.pad(conv2_pre_w, (1, 1, 1, 1))
        conv_crossv_pre_w = nn.functional.pad(conv_crossv_pre_w, (1, 1, 2, 2))
        conv_crossh_pre_w = nn.functional.pad(conv_crossh_pre_w, (2, 2, 1, 1))

        # 4. 准备 10 个分支的权重和偏置
        weight_cat = torch.cat(
            [conv_pre_w, conv1_pre_w, conv2_pre_w, conv_crossh_pre_w, conv_crossv_pre_w,
             conv_w, conv1_w, conv2_w, conv_crossh_w, conv_crossv_w],
            0
        )
        bias_cat = torch.cat(
            [conv_pre_b, conv1_pre_b, conv2_pre_b, conv_crossh_pre_b, conv_crossv_pre_b,
             conv_b, conv1_b, conv2_b, conv_crossh_b, conv_crossv_b],
            0
        )

        # 5. 融合 1x1 卷积 (conv_out)
        final_conv_weight = self.conv_out.weight + self.weight1
        final_conv_bias = self.conv_out.bias

        weight_cat_permuted = weight_cat.permute(1, 2, 3, 0)  # (C_in, 5, 5, 10*C_rep)
        final_conv_weight_squeezed = final_conv_weight.squeeze()  # (C_out, 10*C_rep)

        fused_weight = torch.matmul(weight_cat_permuted, final_conv_weight_squeezed.t())
        fused_weight = fused_weight.permute(3, 0, 1, 2)  # (C_out, C_in, 5, 5)

        fused_bias = torch.matmul(final_conv_weight_squeezed, bias_cat)
        if final_conv_bias is not None:
            fused_bias = fused_bias + final_conv_bias

        # (!!!) 方案A：构造一个融合后的新 Conv+ReLU 模块
        fused_conv = nn.Conv2d(self.in_channels, self.out_channels, 5, 1, 2, bias=True)
        fused_conv.weight.data.copy_(fused_weight)
        fused_conv.bias.data.copy_(fused_bias)

        # (!!!) 方案A：返回一个包含激活的序列，使其与 forward 路径等价
        return nn.Sequential(fused_conv, self.relu)


# ----------------------------------------------------------------------
# 0. Converse2D 模块 (不变)
# ----------------------------------------------------------------------
class Converse2D(nn.Module):
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
        self.weight = nn.Parameter(torch.randn(1, in_channels, kernel_size, kernel_size))
        with torch.no_grad():
            w = self.weight.data.view(1, in_channels, -1)
            self.weight.copy_(F.softmax(w, dim=-1).view_as(self.weight))
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.padding > 0:
            x = F.pad(
                x,
                pad=[self.padding, self.padding, self.padding, self.padding],
                mode=self.padding_mode,
                value=0.0,
            )
        biaseps = torch.sigmoid(self.bias - 9.0) + self.eps  # (1, C, 1, 1)
        STy = self._s_fold_upsample(x, scale=self.scale)  # (B, C, H*s, W*s)
        if self.scale != 1:
            x_nn = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        else:
            x_nn = x
        Hs, Ws = STy.shape[-2:]
        FB = self._psf2otf(self.weight.to(dtype=x.dtype, device=x.device), (Hs, Ws))  # (1,C,Hs,Ws)
        FBC = torch.conj(FB)
        F2B = torch.abs(FB) ** 2
        FBFy = FBC * torch.fft.fftn(STy, dim=(-2, -1))
        FR = FBFy + torch.fft.fftn(biaseps * x_nn, dim=(-2, -1))
        x1 = FB * FR
        FBR = torch.mean(self._splits(x1, self.scale), dim=-1)  # (B,C,Hs/s,Ws/s)
        invW = torch.mean(self._splits(F2B, self.scale), dim=-1)  # (1,C,Hs/s,Ws/s)
        invWBR = FBR / (invW + biaseps + self.eps)
        FCBinvWBR = FBC * invWBR.repeat(1, 1, self.scale, self.scale)
        FX = (FR - FCBinvWBR) / (biaseps + self.eps)
        out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))
        if self.padding > 0:
            p = self.padding * self.scale
            out = out[..., p:-p, p:-p]
        return out

    @staticmethod
    def _splits(a: torch.Tensor, scale: int) -> torch.Tensor:
        *lead, W, H = a.size()
        assert W % scale == 0 and H % scale == 0, "空间尺寸需可被 scale 整除"
        Ws, Hs = W // scale, H // scale
        b = a.view(*lead, scale, Ws, scale, Hs)
        perm = list(range(len(lead))) + [len(lead) + 1, len(lead) + 3, len(lead), len(lead) + 2]
        b = b.permute(*perm).contiguous()
        return b.view(*lead, Ws, Hs, scale * scale)

    @staticmethod
    def _psf2otf(psf: torch.Tensor, shape_hw: tuple[int, int]) -> torch.Tensor:
        H, W = shape_hw
        otf = torch.zeros(psf.shape[:-2] + (H, W), dtype=psf.dtype, device=psf.device)
        otf[..., :psf.shape[-2], :psf.shape[-1]] = psf
        otf = torch.roll(otf, shifts=(-psf.shape[-2] // 2, -psf.shape[-1] // 2), dims=(-2, -1))
        return torch.fft.fftn(otf, dim=(-2, -1))

    @staticmethod
    def _s_fold_upsample(x: torch.Tensor, scale: int) -> torch.Tensor:
        if scale == 1:
            return x
        B, C, H, W = x.shape
        z = torch.zeros(B, C, H * scale, W * scale, dtype=x.dtype, device=x.device)
        z[..., ::scale, ::scale] = x
        return z


# ----------------------------------------------------------------------
# 1. 基础卷积块 (!!! 按方案A修改 !!!)
# ----------------------------------------------------------------------
class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, dilation=1):
        super().__init__()

        # (!!!) 方案A 关键逻辑 (!!!)
        if kernel_size > 1:
            # MBRConv5 已经内置了 (Conv + 内部BNs + ReLU)
            # 所以我们直接使用它，不再需要额外的 BN 或 ReLU
            self.single_conv = MBRConv5(in_channels, out_channels, rep_scale=4)
        else:
            # 1x1 卷积保持原始的 Conv -> BN -> ReLU 结构
            self.single_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=False),
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
        self.down = nn.AvgPool2d(2)
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
# 5. 解码器 (上采样) 模块 (不变)
# ----------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, stage_channels, num_blocks, short_rate,
                 ks, pad, dilation,
                 ks_psf=13):
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
                    kernel_size=ks_psf,
                    scale=2,
                    padding=ks_psf // 2,
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
# 6. 重构后的 SimpleUNet (主模块) (不变)
# ----------------------------------------------------------------------
class SimpleUNet(nn.Module):
    def __init__(self, in_channels, num_cls, ks=3, dilation=1,
                 stage_channels=5 * [32], num_blocks=5 * [1], short_rate=0.5,
                 ks_psf=13):
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
        self.decoder = Decoder(stage_channels, num_blocks, short_rate,
                               ks, self.pad, dilation,
                               ks_psf=ks_psf)
        self.seg_head = SingleConv(int(short_rate * stage_channels[0]), num_cls, 1, 0, 1)

    def forward(self, x):
        x_to_bottleneck, shortcuts = self.encoder(x)
        refined_shortcuts = self.skip_connections(shortcuts)
        x_after_bottleneck = self.bottleneck(x_to_bottleneck)
        x = self.decoder(x_after_bottleneck, refined_shortcuts)
        output = self.seg_head(x)
        return output