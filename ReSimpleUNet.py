import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
# (!!! 新增 !!!) 导入 VolSelfAttention 需要的库
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# ----------------------------------------------------------------------
# 0. 您提供的 Converse2D 模块 (不变)
# ----------------------------------------------------------------------
class Converse2D(nn.Module):
    """
    Converse2D: 频域闭式解型上采样-去卷积算子（深度可分）
    ... (代码与您提供的一致，此处折叠) ...
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
# (!!! 新增 !!!) 1. VolSelfAttention 及其依赖
# ----------------------------------------------------------------------
class DecayPos1d(nn.Module):
    """
    1D 衰减相对位置先验（按 head 设定不同衰减速率）
    ... (代码与您提供的一致，此处折叠) ...
    """

    def __init__(self, embed_dim: int, num_heads: int, initial_value: float, heads_range: float):
        super().__init__()
        # 频率角频率（未直接用到，保留以兼容可能的扩展）
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        # 每个 head 一个衰减速率（越靠后 head 衰减越慢/快，取决于 heads_range）
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

    def generate_1d_decay(self, l: int) -> torch.Tensor:
        idx = torch.arange(l, device=self.decay.device)
        dist = (idx[:, None] - idx[None, :]).abs()  # (L, L)
        mask = dist * self.decay[:, None, None]  # (H, L, L)
        return mask

    def forward(self, slen: int) -> torch.Tensor:
        return self.generate_1d_decay(int(slen))


class VolSelfAttention(nn.Module):
    """
    Volumetric Self-Attention
    ... (代码与您提供的一致，此处折叠) ...
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 相对位置偏置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)  # (2, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

        # token 自注意力（线性投影）
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 频谱位置先验：按 head 提供 (H, C_per_head, C_per_head) 的衰减偏置
        self.realPos = DecayPos1d(embed_dim=64, num_heads=num_heads, initial_value=2, heads_range=4)

        # 频谱分支：Conv1x1 + 深度可分离卷积
        self.qkv_C = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv_C = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.proj_C = nn.Conv2d(dim, dim, kernel_size=1)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        # 简单空间注意力门控（把 (B,H,N,N) 池化成 (B,N,1) 作为加权）
        self.Gao_spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_heads, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, N, C)，其中 N=Wh*Ww 必须与 window_size 匹配
        return: (B, N, C)
        """
        B, N, C = x.shape
        Wh, Ww = self.window_size
        assert N == Wh * Ww, f"N ({N}) 必须等于 window_size={self.window_size} 的乘积"
        # (!!!) 注意：这里的 hh * hh == N 约束了 N 必须是完全平方数
        # (!!!) 对于 (8,8) 窗口, N=64, hh=8, 这是 OK 的
        hh = int(math.isqrt(N))
        assert hh * hh == N, "N 应为完全平方数，确保可重排为 (hh, hh)"

        # -------- Token 维自注意力 --------
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, C//H)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, H, N, N)

        # 相对位置偏置
        rel_pos = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)  # (N,N,H)
        rel_pos = rel_pos.permute(2, 0, 1).contiguous()  # (H,N,N)
        attn = attn + rel_pos.unsqueeze(0)  # (B,H,N,N)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x1 = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x1 = self.proj_drop(self.proj(x1))

        # -------- 频谱/通道重排分支 --------
        # 频谱先验（按每头通道数）
        c_per_head = C // self.num_heads
        realPos = self.realPos(c_per_head)  # (H, CpH, CpH)

        x_s = rearrange(x, 'b (h w) c -> b c h w', h=hh, w=hh)  # (B,C,hh,hh)
        qkv_c = self.qkv_dwconv_C(self.qkv_C(x_s))
        q_c, k_c, v_c = qkv_c.chunk(3, dim=1)
        q_c = rearrange(q_c, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_c = rearrange(k_c, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_c = rearrange(v_c, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_c = F.normalize(q_c, dim=-1)
        k_c = F.normalize(k_c, dim=-1)

        attn_c = (q_c @ k_c.transpose(-2, -1)) * self.temperature + realPos  # 广播到 (B,H,CpH,CpH)
        attn_c = attn_c.softmax(dim=-1)

        x2 = (attn_c @ v_c)  # (B,H,CpH,(hh*hh))
        x2 = rearrange(x2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=hh, w=hh)
        x2 = self.proj_C(x2)  # (B,C,hh,hh)
        x2 = rearrange(x2, 'b c h w -> b (h w) c', h=hh, w=hh)  # (B,N,C)

        # -------- 体素式融合（空间权重）--------
        # (!!!) 关键约束: N 必须等于 64 才能让 reshape(Bsa, N, 1) 工作
        attn_spatial = self.Gao_spatial_attention(attn)  # (B,64,1,1)
        Bsa, _, _, _ = attn_spatial.shape
        attn_spatial = attn_spatial.reshape(Bsa, N, 1)  # (B, N, 1)
        x4 = attn_spatial * x2

        out = x1 + x2 + x4
        return out

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


# ----------------------------------------------------------------------
# (!!! 新增 !!!) 2. 窗口化/逆窗口化 辅助函数
# ----------------------------------------------------------------------
def window_partition(x, window_size):
    """
    将 (B, C, H, W) 划分为 (B*num_windows, N, C)，N=window_size*window_size
    使用 einops.rearrange
    """
    B, C, H, W = x.shape
    wh, ww = window_size
    assert H % wh == 0 and W % ww == 0, "H, W 必须能被 window_size 整除"
    # 1. (B, C, H, W) -> (B, C, h, wh, w, ww)  (h=H/wh, w=W/ww)
    # 2. -> (B, h, w, wh, ww, C)  (permute)
    # 3. -> (B*h*w, wh*ww, C)      (reshape)
    x = rearrange(x, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=wh, p2=ww)
    return x


def window_reverse(windows, window_size, H, W, B):
    """
    将 (B*num_windows, N, C) 逆转为 (B, C, H, W)
    使用 einops.rearrange
    """
    wh, ww = window_size
    h, w = H // wh, W // ww  # num_windows_h, num_windows_w
    # 1. (B*h*w, wh*ww, C) -> (B, h, w, wh, ww, C)
    # 2. -> (B, C, h, wh, w, ww) (permute)
    # 3. -> (B, C, H, W)       (reshape)
    x = rearrange(windows, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', h=h, w=w, p1=wh, p2=ww, b=B)
    return x


# ----------------------------------------------------------------------
# 3. 基础卷积块 (不变)
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
# 4. 编码器 (下采样) 模块 (不变)
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
# 5. 跳跃连接处理模块 (不变)
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
# 6. 瓶颈层 (Bottleneck) 模块 (!!! 已修改为 VolSelfAttention !!!)
# ----------------------------------------------------------------------
class Bottleneck(nn.Module):
    """
    使用 VolSelfAttention 重写的瓶颈层
    """

    def __init__(self, in_channels, mid_channels, num_blocks, ks, pad, dilation, num_heads):
        super().__init__()

        # 1. 使用 1x1 卷积将输入通道 (in_channels) 调整为注意力模块的通道 (mid_channels)
        #    我们复用 SingleConv (Conv+BN+ReLU)
        self.pre_conv = SingleConv(in_channels, mid_channels, 1, 0, 1)

        # 2. 定义窗口大小 (!!! 硬编码为 (8, 8) 以匹配 N=64 的约束 !!!)
        self.window_size = (8, 8)

        # 3. 实例化 VolSelfAttention
        #    注意：dim 必须是 mid_channels
        self.attn_block = VolSelfAttention(
            dim=mid_channels,
            window_size=self.window_size,
            num_heads=num_heads
        )

        # (原有的 convs 循环被移除了, num_blocks, ks, pad, dilation 不再在此处使用)

    def forward(self, x):
        # 1. 调整通道
        # x_in: (B, 16, 16, 16) -> (B, 8, 16, 16) [假设 mid_channels=8]
        x = self.pre_conv(x)

        B, C, H, W = x.shape

        # 检查特征图尺寸是否可以被窗口整除
        assert H % self.window_size[0] == 0 and W % self.window_size[1] == 0, \
            f"特征图尺寸 ({H}, {W}) 无法被 window_size ({self.window_size}) 整除"

        # 2. 窗口化: (B, C, H, W) -> (B*num_win, N, C)
        # (B, 8, 16, 16) -> (B*4, 64, 8)
        x_windows = window_partition(x, self.window_size)

        # 3. 应用注意力
        # (B*4, 64, 8) -> (B*4, 64, 8)
        attn_windows = self.attn_block(x_windows)

        # 4. 逆窗口化: (B*num_win, N, C) -> (B, C, H, W)
        # (B*4, 64, 8) -> (B, 8, 16, 16)
        x_out = window_reverse(attn_windows, self.window_size, H, W, B)

        return x_out


# ----------------------------------------------------------------------
# 7. 解码器 (上采样) 模块 (不变, 依赖于 Converse2D)
# ----------------------------------------------------------------------
class Decoder(nn.Module):
    """
    U-Net的解码器（上采样）路径。
    使用 Converse2D 作为上采样层。
    ... (代码与您提供的一致，此处折叠) ...
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
# 8. 重构后的 SimpleUNet (主模块) (!!! 已修改 !!!)
# ----------------------------------------------------------------------
class SimpleUNet(nn.Module):
    def __init__(self, in_channels, num_cls, ks=3, dilation=1, stage_channels=5 * [32], num_blocks=5 * [1],
                 short_rate=0.5,
                 num_heads=8  # (!!! 新增参数 !!!)
                 ):
        super(SimpleUNet, self).__init__()
        assert short_rate > 0, 'short_rate must be greater than 0!'
        assert len(stage_channels) == len(num_blocks), 'The length of stage_channels and num_blocks must match!'

        self.pad = dilation * (ks - 1) // 2

        self.encoder = Encoder(in_channels, stage_channels, num_blocks, ks, self.pad, dilation)
        self.skip_connections = SkipConnections(stage_channels, short_rate)

        # (!!! 已修改 !!!)
        # 将 num_heads 传递给 Bottleneck
        self.bottleneck = Bottleneck(
            in_channels=stage_channels[-1],
            mid_channels=int(short_rate * stage_channels[-1]),
            num_blocks=num_blocks[-1],
            ks=ks,
            pad=self.pad,
            dilation=dilation,
            num_heads=num_heads  # (!!! 传入 !!!)
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
# 9. 测试代码 (!!! 已修改 !!!)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 确保有可用的CUDA设备
    if torch.cuda.is_available():
        input = torch.randn(1, 3, 256, 256).cuda()

        # (!!! 已修改 !!!)
        # 传入 num_heads=8
        model = SimpleUNet(
            in_channels=3,
            num_cls=1,
            ks=3,
            stage_channels=[16, 16, 16, 16, 16],
            num_blocks=[1, 1, 1, 1, 1],
            short_rate=0.5,
            num_heads=8  # (!!!) 传入注意力头数
        ).cuda()

        flops, params = profile(model, inputs=(input,))
        output = model(input)

        print(f"\n--- Final Model Output (with Converse2D and VolSelfAttention) ---")
        print(f"Input shape: {input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"FLOPs (G): {flops / 1e9}")
        print(f"Params (M): {params / 1e6}")
    else:
        print("CUDA not available. Please run this on a machine with a GPU.")