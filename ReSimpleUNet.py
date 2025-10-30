import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from einops import rearrange
from einops.layers.torch import Rearrange
from thop import profile


# ----------------------------------------------------------------------
# 0. 您提供的 MANO 模块 (粘贴在此处)
# ----------------------------------------------------------------------
class FeedForward(nn.Module):
    """Pre-LN + MLP(GELU, Dropout)。输入输出: (B, L, C)"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # x: (B, L, C)
        return self.net(x)


class AttentionBlock(nn.Module):
    """标准多头自注意力（全局，输出维保持 dim）。输入输出: (B, L, C)"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):  # x: (B, L, C)
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class LocalAttention2D(nn.Module):
    """
    基于 unfold/fold 的窗口注意力（每个 K×K 局部内做自注意力）。
    输入/输出: (B, H, W, C) 形状保持不变。
    """

    def __init__(self, kernel_size, stride, dim, heads, dim_head, dropout):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm = nn.LayerNorm(dim)
        self.attn = AttentionBlock(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):  # x: (B, H, W, C)
        B, H, W, C = x.shape
        x_chw = rearrange(x, "B H W C -> B C H W")

        # (B, C*K*K, L)
        patches = self.unfold(x_chw)
        patches = rearrange(patches, "B (C K1 K2) L -> (B L) (K1 K2) C", K1=self.kernel_size, K2=self.kernel_size)
        patches = self.norm(patches)

        # 局部窗口内自注意力: (B*L, K*K, C)
        out = self.attn(patches)

        # 还原并 fold 回图像
        out = rearrange(out, "(B L) (K1 K2) C -> B (C K1 K2) L", B=B, K1=self.kernel_size, K2=self.kernel_size)
        fold = nn.Fold(output_size=(H, W), kernel_size=self.kernel_size, stride=self.stride)
        out = fold(out)  # (B, C, H, W)

        # 归一重叠区域
        with torch.no_grad():
            norm = self.unfold(torch.ones((B, 1, H, W), device=x_chw.device))
            norm = fold(norm)  # (B, 1, H, W)
        out = out / (norm + 1e-6)

        return rearrange(out, "B C H W -> B H W C")


class Multipole_Attention2D(nn.Module):
    """
    多尺度局部注意力：逐级下采样做局部注意力，再逐级上采样聚合。
    输入/输出: (B, H, W, C)
    """

    def __init__(
            self,
            image_size: int,
            in_channels: int,
            local_attention_kernel_size: int,
            local_attention_stride: int,
            downsampling: Literal["avg_pool", "conv"],
            upsampling: Literal["avg_pool", "conv"],
            sampling_rate: int,
            heads: int,
            dim_head: int,
            dropout: float,
            channel_scale: int,
    ):
        super().__init__()

        # 自动计算最多能下采样多少层（直到不再可整除）
        levels = 0
        cur = image_size
        while cur % sampling_rate == 0 and cur > 1:
            cur //= sampling_rate
            levels += 1
        self.levels = max(1, levels)

        # 注意：本实现各层通道数不变（=in_channels），channel_scale 预留未使用
        self.Attention = LocalAttention2D(
            kernel_size=local_attention_kernel_size,
            stride=local_attention_stride,
            dim=in_channels,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        if downsampling == "avg_pool":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.AvgPool2d(kernel_size=sampling_rate, stride=sampling_rate),
                Rearrange("B C H W -> B H W C"),
            )
        elif downsampling == "conv":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                          kernel_size=sampling_rate, stride=sampling_rate, bias=False),
                Rearrange("B C H W -> B H W C"),
            )
        else:
            raise ValueError("downsampling must be 'avg_pool' or 'conv'")

        if upsampling == "avg_pool":
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Upsample(scale_factor=sampling_rate, mode="nearest"),
                Rearrange("B C H W -> B H W C"),
            )
        elif upsampling == "conv":
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=sampling_rate, stride=sampling_rate, bias=False),
                Rearrange("B C H W -> B H W C"),
            )
        else:
            raise ValueError("upsampling must be 'avg_pool' or 'conv'")

    def forward(self, x):  # x: (B, H, W, C)
        x_in = x
        outs = [self.Attention(x_in)]
        for _ in range(1, self.levels):
            x_in = self.down(x_in)
            outs.append(self.Attention(x_in))

        # 自顶向上聚合：逐级上采并加权融合
        res = outs.pop()  # 最低分辨率
        for l, feat in enumerate(reversed(outs)):  # 从次低到最高
            res = feat + (1.0 / (l + 1)) * self.up(res)
        return res


class Multipole_TransformerBlock(nn.Module):
    """堆叠多个 (Multipole_Attention2D + FeedForward)。输入/输出: (B, H, W, C)"""

    def __init__(
            self,
            image_size,
            in_channels,
            kernel_size,
            local_attention_stride,
            downsampling,
            upsampling,
            sampling_rate,
            dim,
            depth,
            heads,
            dim_head,
            att_dropout,
            channel_scale,
            mlp_dim,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Multipole_Attention2D(
                    image_size=image_size,
                    in_channels=in_channels,
                    local_attention_kernel_size=kernel_size,
                    local_attention_stride=local_attention_stride,
                    downsampling=downsampling,
                    upsampling=upsampling,
                    sampling_rate=sampling_rate,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=att_dropout,
                    channel_scale=channel_scale,
                ),
                FeedForward(dim, mlp_dim),
            ]) for _ in range(depth)
        ])

    def forward(self, x):  # x: (B, H, W, C)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class MANO(nn.Module):
    """Multipole Attention Neural Operator。输入/输出: (B, C, H, W)"""

    def __init__(
            self,
            device,
            image_size,
            dim,
            depth,
            heads,
            dim_head,
            att_dropout,
            channel_scale,
            mlp_dim,
            channels,
            emb_dropout,
            local_attention_span,
            local_attention_stride,
            att_sampling: Literal["avg_pool", "conv"],
            att_sampling_rate,
    ):
        super().__init__()
        self.in_channels = channels
        self.linear_p = nn.Linear(channels, dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Multipole_TransformerBlock(
            image_size=image_size,
            in_channels=dim,  # 注意：transformer 内部按 (B,H,W,C=dim) 运算
            kernel_size=local_attention_span,
            local_attention_stride=local_attention_stride,
            downsampling=att_sampling,
            upsampling=att_sampling,
            sampling_rate=att_sampling_rate,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            att_dropout=att_dropout,
            channel_scale=channel_scale,
            mlp_dim=mlp_dim,
        )
        self.linear_q = nn.Linear(dim, dim)
        self.output_layer = nn.Linear(dim, self.in_channels)
        self.activation = nn.Tanh()
        self.to(device)

    def forward(self, x):  # x: (B, C, H, W)
        x = rearrange(x, 'B C H W -> B H W C')
        x = self.linear_p(x)
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.linear_q(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return rearrange(x, 'B H W C -> B C H W')


# ----------------------------------------------------------------------
# 1. Converse2D 模块 (不变)
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
            self.weight.copy_(F.softmax(w, dim=-1).view_as(self.weight))  # 核归一化（按通道）
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))  # 可学习先验强度

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.padding > 0:
            x = F.pad(
                x,
                pad=[self.padding, self.padding, self.padding, self.padding],
                mode=self.padding_mode,
                value=0.0,
            )
        biaseps = torch.sigmoid(self.bias - 9.0) + self.eps  # 形状 (1, C, 1, 1)
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
        invWBR = FBR / (invW + biaseps + self.eps)  # 稳定除法
        FCBinvWBR = FBC * invWBR.repeat(1, 1, self.scale, self.scale)  # broadcast 回到 (B,C,Hs,Ws)
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
# 2. 基础卷积块 (不变)
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
# 3. 编码器 (下采样) 模块 (不变)
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
# 6. 解码器 (上采样) 模块 (不变, 沿用您的 Converse2D 版本)
# ----------------------------------------------------------------------
class Decoder(nn.Module):
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
            x_up = self.up_layers[j](x)
            shortcut = re_shortcuts[j]
            y = torch.concat([shortcut, x_up], dim=1)
            x = self.de_layers[j](y)
        return x


# ----------------------------------------------------------------------
# 7. 重构后的 SimpleUNet (主模块) (!!! 已修改 !!!)
# ----------------------------------------------------------------------
class SimpleUNet(nn.Module):
    def __init__(self,
                 in_channels,
                 num_cls,
                 image_size_hw,  # (!!! NEW !!!) 原始输入图像的 H 或 W (假设 H=W)
                 device,  # (!!! NEW !!!) MANO 初始化需要 "cpu" 或 "cuda"
                 ks=3,
                 dilation=1,
                 stage_channels=5 * [32],
                 num_blocks=5 * [1],
                 short_rate=0.5,
                 # (!!! NEW !!!) MANO 相关的超参数
                 use_mano_bottleneck=True,
                 mano_dim=128,
                 mano_depth=4,
                 mano_heads=4,
                 mano_dim_head=32,
                 mano_att_dropout=0.1,
                 mano_emb_dropout=0.1,
                 mano_local_span=2,
                 mano_local_stride=1,
                 mano_att_sampling="conv",
                 mano_att_sampling_rate=4
                 ):
        super(SimpleUNet, self).__init__()
        assert short_rate > 0, 'short_rate must be greater than 0!'
        assert len(stage_channels) == len(num_blocks), 'The length of stage_channels and num_blocks must match!'

        self.pad = dilation * (ks - 1) // 2
        self.depth = len(stage_channels)
        self.use_mano_bottleneck = use_mano_bottleneck

        # --- 1. 编码器 ---
        self.encoder = Encoder(in_channels, stage_channels, num_blocks, ks, self.pad, dilation)

        # --- 2. 跳跃连接 ---
        self.skip_connections = SkipConnections(stage_channels, short_rate)

        # --- 3. 瓶颈层 (包含 MANO) ---
        bottleneck_channels_in = stage_channels[-1]
        bottleneck_channels_out = int(short_rate * stage_channels[-1])

        # (原始的 Bottleneck 卷积)
        self.bottleneck = Bottleneck(
            in_channels=bottleneck_channels_in,
            mid_channels=bottleneck_channels_out,
            num_blocks=num_blocks[-1],
            ks=ks,
            pad=self.pad,
            dilation=dilation
        )

        # (!!! NEW !!!) (添加 MANO 模块)
        if self.use_mano_bottleneck:
            # 计算瓶颈层的空间尺寸 (H, W)
            # 假设输入 H=W，且每次下采样都 /2
            # 深度为 5 时, 下采样 4 次 (depth - 1)
            bottleneck_image_size = image_size_hw // (2 ** (self.depth - 1))

            self.mano_block = MANO(
                device=device,
                image_size=bottleneck_image_size,
                dim=mano_dim,
                depth=mano_depth,
                heads=mano_heads,
                dim_head=mano_dim_head,
                att_dropout=mano_att_dropout,
                channel_scale=2,  # 使用您 MANO 示例中的值
                mlp_dim=mano_dim,  # 使用您 MANO 示例中的值 (假设 mlp_dim = dim)
                channels=bottleneck_channels_out,  # (!!! 关键 !!!) 通道数匹配 bottleneck 的输出
                emb_dropout=mano_emb_dropout,
                local_attention_span=mano_local_span,
                local_attention_stride=mano_local_stride,
                att_sampling=mano_att_sampling,
                att_sampling_rate=mano_att_sampling_rate,
            )

        # --- 4. 解码器 ---
        self.decoder = Decoder(stage_channels, num_blocks, short_rate, ks, self.pad, dilation)

        # --- 5. 输出头 ---
        self.seg_head = SingleConv(int(short_rate * stage_channels[0]), num_cls, 1, 0, 1)

    def forward(self, x):
        # 编码
        x_to_bottleneck, shortcuts = self.encoder(x)

        # 跳跃连接
        refined_shortcuts = self.skip_connections(shortcuts)

        # 瓶颈层
        x_after_bottleneck = self.bottleneck(x_to_bottleneck)

        # (!!! NEW !!!) (在瓶颈层后应用 MANO)
        if self.use_mano_bottleneck:
            x_after_bottleneck = self.mano_block(x_after_bottleneck)

        # 解码
        x = self.decoder(x_after_bottleneck, refined_shortcuts)

        # 输出
        output = self.seg_head(x)
        return output


# ----------------------------------------------------------------------
# 8. 测试代码 (!!! 已修改 !!!)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 确保有可用的CUDA设备
    if torch.cuda.is_available():
        device = "cuda"
        input_size = 256  # (!!! NEW !!!)

        input = torch.randn(1, 3, input_size, input_size).to(device)

        # (!!! NEW !!!) (传入 image_size_hw 和 device)
        model = SimpleUNet(
            in_channels=3,
            num_cls=1,
            image_size_hw=input_size,  # (!!!)
            device=device,  # (!!!)
            ks=3,
            stage_channels=[16, 32, 64, 128, 256],  # 演示更典型的U-Net通道
            num_blocks=[1, 1, 1, 1, 1],
            short_rate=0.5,
            use_mano_bottleneck=True  # 启用 MANO
        ).to(device)

        flops, params = profile(model, inputs=(input,))
        output = model(input)

        print(f"\n--- Final Model Output (with Converse2D + MANO) ---")
        print(f"Input shape: {input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"FLOPs (G): {flops / 1e9}")
        print(f"Params (M): {params / 1e6}")
    else:
        print("CUDA not available. Please run this on a machine with a GPU.")