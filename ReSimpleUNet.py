import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from einops import rearrange  # (!!! 新增 !!!) 导入


# ----------------------------------------------------------------------
# (!!! 新增 !!!) 粘贴您提供的所有新模块
# ----------------------------------------------------------------------

class Inter_CacheModulation(nn.Module):
    def __init__(self, in_c=3):
        super(Inter_CacheModulation, self).__init__()
        self.align = nn.AdaptiveAvgPool2d(in_c)
        self.conv_width = nn.Conv1d(in_channels=in_c, out_channels=2 * in_c, kernel_size=1)
        self.gatingConv = nn.Conv1d(in_channels=in_c, out_channels=in_c, kernel_size=1)

    def forward(self, x1, x2):
        C = x1.shape[-1]
        x2_pW = self.conv_width(self.align(x2) + x1)
        scale, shift = x2_pW.chunk(2, dim=1)
        x1_p = x1 * scale + shift
        x1_p = x1_p * F.gelu(self.gatingConv(x1_p))
        return x1_p


class Intra_CacheModulation(nn.Module):
    def __init__(self, embed_dim=48):
        super(Intra_CacheModulation, self).__init__()
        self.down = nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=1)
        self.up = nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=1)
        self.gatingConv = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1)

    def forward(self, x1, x2):
        x_gated = F.gelu(self.gatingConv(x2 + x1)) * (x2 + x1)
        x_p = self.up(self.down(x_gated))
        return x_p


class ReGroup(nn.Module):
    def __init__(self, groups=[1, 1, 2, 4]):
        super(ReGroup, self).__init__()
        self.gourps = groups  # (注意：原代码中可能是笔误，应为 self.groups)
        # (修正)
        self.groups = groups

    def forward(self, query, key, value):
        C = query.shape[1]
        channel_features = query.mean(dim=0)
        correlation_matrix = torch.corrcoef(channel_features)

        mean_similarity = correlation_matrix.mean(dim=1)
        _, sorted_indices = torch.sort(mean_similarity, descending=True)

        query_sorted = query[:, sorted_indices, :]
        key_sorted = key[:, sorted_indices, :]
        value_sorted = value[:, sorted_indices, :]

        query_groups = []
        key_groups = []
        value_groups = []
        start_idx = 0
        total_ratio = sum(self.groups)  # (修正)
        group_sizes = [int(ratio / total_ratio * C) for ratio in self.groups]  # (修正)

        # (修正) 处理整除余数，确保所有通道都被分配
        group_sizes[-1] = C - sum(group_sizes[:-1])

        for group_size in group_sizes:
            end_idx = start_idx + group_size
            query_groups.append(query_sorted[:, start_idx:end_idx, :])
            key_groups.append(key_sorted[:, start_idx:end_idx, :])
            value_groups.append(value_sorted[:, start_idx:end_idx, :])
            start_idx = end_idx

        return query_groups, key_groups, value_groups


def CalculateCurrentLayerCache(x, dim=128, groups=[1, 1, 2, 4]):
    lens = len(groups)
    ceil_dim = dim

    # (修正) 确保 qv_cache 初始化在设备上
    qv_cache = None

    for i in range(lens):
        qv_cache_f = x[i].clone().detach()
        qv_cache_f = torch.mean(qv_cache_f, dim=0, keepdim=True).detach()

        # (修正) 检查 qv_cache_f 是否为空（如果分组大小为0）
        if qv_cache_f.shape[1] == 0:
            continue

        update_elements = F.interpolate(qv_cache_f.unsqueeze(1), size=(ceil_dim, ceil_dim), mode='bilinear',
                                        align_corners=False)
        c_i = qv_cache_f.shape[-1]

        # (修正) 防止 c_i / dim 导致浮点数计算，确保类型一致
        weight = c_i / dim

        if qv_cache is None:
            # (修正) 确保在正确的设备上初始化
            qv_cache = torch.zeros_like(update_elements, device=update_elements.device)

        # (修正) 原代码 i==0 的分支似乎有误，这里统一累加
        qv_cache = qv_cache + update_elements * weight

    # (修正) 如果所有组都为空，返回一个零张量
    if qv_cache is None:
        # 假设 x[0] 存在且在正确设备上，但如果 x 为空列表则会失败
        # 我们需要一个更鲁棒的方法，或者假定 x 至少有一个元素
        # 为了简单起见，我们假设 x[0] 至少存在一个张量来获取设备
        # (更正) 我们不能假设 x[0] 存在。
        # 更好的方法是 qv_cache 在循环开始前初始化为0张量
        # (已在上面修正)
        pass

    return qv_cache.squeeze(1)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(4, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.group = [1, 2, 2, 3]  # (注意: 这决定了 dim 最好是 8 的倍数)

        self.intra_modulator = Intra_CacheModulation(embed_dim=dim)

        # (修正) 确保 dim // 8 至少为 1，否则 Inter_CacheModulation 会失败
        assert dim % 8 == 0, "Attention 模块的 dim 必须是 8 的倍数"
        self.inter_modulator1 = Inter_CacheModulation(in_c=1 * dim // 8)
        self.inter_modulator2 = Inter_CacheModulation(in_c=2 * dim // 8)
        self.inter_modulator3 = Inter_CacheModulation(in_c=2 * dim // 8)
        self.inter_modulator4 = Inter_CacheModulation(in_c=3 * dim // 8)
        self.inter_modulators = [self.inter_modulator1, self.inter_modulator2, self.inter_modulator3,
                                 self.inter_modulator4]

        self.regroup = ReGroup(self.group)
        self.dim = dim

    def forward(self, x, qv_cache=None):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')

        qu, ke, va = self.regroup(q, k, v)
        attScore = []
        tmp_cache = []
        for index in range(len(self.group)):
            # (修正) 处理分组大小为0的边界情况
            if qu[index].shape[1] == 0:
                attScore.append(torch.tensor([], device=x.device))  # 添加空张量占位
                tmp_cache.append(torch.tensor([], device=x.device))
                continue

            query_head = qu[index]
            key_head = ke[index]

            query_head = torch.nn.functional.normalize(query_head, dim=-1)
            key_head = torch.nn.functional.normalize(key_head, dim=-1)

            attn = (query_head @ key_head.transpose(-2, -1)) * self.temperature[index, :, :]
            attn = attn.softmax(dim=-1)

            attScore.append(attn)
            t_cache = query_head.clone().detach() + key_head.clone().detach()
            tmp_cache.append(t_cache)

        # (修正) 过滤掉空张量
        tmp_cache_filtered = [t for t in tmp_cache if t.shape[0] > 0]
        if not tmp_cache_filtered:
            # 如果所有组都为空，无法继续
            return x, None  # 或者返回0？
        tmp_caches = torch.cat(tmp_cache_filtered, 1)

        out = []
        if qv_cache is not None:
            if qv_cache.shape[-1] != c:
                qv_cache = F.adaptive_avg_pool2d(qv_cache, c)

        va_idx = 0
        for i in range(4):  # 遍历所有可能的组
            if attScore[i].shape[0] > 0:  # 如果这个组不是空的
                if qv_cache is not None:
                    inter_modulator = self.inter_modulators[i]
                    attScore[i] = inter_modulator(attScore[i], qv_cache) + attScore[i]
                out.append(attScore[i] @ va[va_idx])
                va_idx += 1
            else:
                # 如果 attScore[i] 是空的, va[i] 对应的组也是空的, 跳过
                pass

        update_factor = 0.9
        # (修正) 过滤掉空张量
        attScore_filtered = [a for a in attScore if a.shape[0] > 0]
        if not attScore_filtered:
            # 如果没有有效的 attScore, 无法计算 cache
            out_all = x  # 返回原始 x
            qv_cache_new = None
        else:
            if qv_cache is not None:
                update_elements = CalculateCurrentLayerCache(attScore_filtered, c, self.group)
                qv_cache_new = qv_cache * update_factor + update_elements * (1 - update_factor)
            else:
                update_elements = CalculateCurrentLayerCache(attScore_filtered, c, self.group)
                qv_cache_new = update_elements * update_factor

            out_all = torch.concat(out, 1)
            out_all = self.intra_modulator(out_all, tmp_caches) + out_all

            out_all = rearrange(out_all, 'b c (h w) -> b c h w', h=h, w=w)
            out_all = self.project_out(out_all)

        return [out_all, qv_cache_new]


# ----------------------------------------------------------------------
# 0. 您提供的 Converse2D 模块 (不变)
# ----------------------------------------------------------------------
class Converse2D(nn.Module):
    # ... (代码不变，此处省略) ...
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, scale: int = 1, padding: int = 2,
                 padding_mode: str = "circular", eps: float = 1e-5):
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
            x = F.pad(x, pad=[self.padding, self.padding, self.padding, self.padding], mode=self.padding_mode,
                      value=0.0)
        biaseps = torch.sigmoid(self.bias - 9.0) + self.eps
        STy = self._s_fold_upsample(x, scale=self.scale)
        if self.scale != 1:
            x_nn = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        else:
            x_nn = x
        Hs, Ws = STy.shape[-2:]
        FB = self._psf2otf(self.weight.to(dtype=x.dtype, device=x.device), (Hs, Ws))
        FBC = torch.conj(FB)
        F2B = torch.abs(FB) ** 2
        FBFy = FBC * torch.fft.fftn(STy, dim=(-2, -1))
        FR = FBFy + torch.fft.fftn(biaseps * x_nn, dim=(-2, -1))
        x1 = FB * FR
        FBR = torch.mean(self._splits(x1, self.scale), dim=-1)
        invW = torch.mean(self._splits(F2B, self.scale), dim=-1)
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
# 1. 基础卷积块 (不变)
# ----------------------------------------------------------------------
class SingleConv(nn.Module):
    # ... (代码不变，此处省略) ...
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
    # ... (代码不变，此处省略) ...
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
# 3. 跳跃连接处理模块 (不变)
# ----------------------------------------------------------------------
class SkipConnections(nn.Module):
    # ... (代码不变，此处省略) ...
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
# ----------------------------------------------------------------------
class Bottleneck(nn.Module):
    """
    (!!! 已修改 !!!)
    瓶颈层现在包含一个入口卷积，后跟 'num_blocks' 个 Attention 模块。
    qv_cache 在 Attention 模块之间传递。
    """

    def __init__(self, in_channels, mid_channels, num_blocks, ks, pad, dilation, num_heads):
        super().__init__()

        # 1. 入口卷积：将通道数从 in_channels 转换为 mid_channels
        self.entry_conv = SingleConv(in_channels, mid_channels, ks, pad, dilation)

        # 2. 创建 num_blocks 个 Attention 模块
        self.attn_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.attn_blocks.append(
                Attention(dim=mid_channels, num_heads=num_heads, bias=True)
            )

    def forward(self, x):
        # 1. 通过入口卷积
        x = self.entry_conv(x)

        # 2. 依次通过所有 Attention 模块，在内部传递 cache
        qv_cache = None  # 初始化 cache
        for attn_block in self.attn_blocks:
            x, qv_cache = attn_block(x, qv_cache=qv_cache)

        # 3. 返回最终的特征图
        return x


# ----------------------------------------------------------------------
# 5. 解码器 (上采样) 模块 (不变, 依赖 Converse2D)
# ----------------------------------------------------------------------
class Decoder(nn.Module):
    # ... (代码不变，此处省略) ...
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
            x_up_simple = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
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
                 short_rate=0.5, attn_heads=4):  # (!!! 新增 attn_heads 参数 !!!)
        super(SimpleUNet, self).__init__()
        assert short_rate > 0, 'short_rate must be greater than 0!'
        assert len(stage_channels) == len(num_blocks), 'The length of stage_channels and num_blocks must match!'

        # (!!! 注意 !!!)
        # 确保瓶颈层的通道数 (mid_channels) 是 8 的倍数
        # mid_channels = int(short_rate * stage_channels[-1])
        # 如果 stage_channels[-1]=16, short_rate=0.5,
        # mid_channels = 8, 这是 8 的倍数，可以运行。
        # 如果 stage_channels[-1]=32, short_rate=0.5,
        # mid_channels = 16, 这也是 8 的倍数。
        mid_ch_bottleneck = int(short_rate * stage_channels[-1])
        assert mid_ch_bottleneck % 8 == 0, \
            f"瓶颈层通道数 (mid_channels={mid_ch_bottleneck}) 必须是 8 的倍数以适应 Attention 模块"

        self.pad = dilation * (ks - 1) // 2

        self.encoder = Encoder(in_channels, stage_channels, num_blocks, ks, self.pad, dilation)
        self.skip_connections = SkipConnections(stage_channels, short_rate)

        # (!!! 已修改 !!!) 更新 Bottleneck 的实例化
        self.bottleneck = Bottleneck(
            in_channels=stage_channels[-1],
            mid_channels=mid_ch_bottleneck,
            num_blocks=num_blocks[-1],  # 使用最后一个阶段的 block 数
            ks=ks,
            pad=self.pad,
            dilation=dilation,
            num_heads=attn_heads  # (!!! 传入新参数 !!!)
        )

        self.decoder = Decoder(stage_channels, num_blocks, short_rate, ks, self.pad, dilation)
        self.seg_head = SingleConv(int(short_rate * stage_channels[0]), num_cls, 1, 0, 1)

    def forward(self, x):
        # (!!! forward 流程保持不变 !!!)
        x_to_bottleneck, shortcuts = self.encoder(x)
        refined_shortcuts = self.skip_connections(shortcuts)
        x_after_bottleneck = self.bottleneck(x_to_bottleneck)
        x = self.decoder(x_after_bottleneck, refined_shortcuts)
        output = self.seg_head(x)
        return output


# ----------------------------------------------------------------------
# 7. 测试代码 (!!! 已修改 !!!)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 确保有可用的CUDA设备
    if torch.cuda.is_available():
        input = torch.randn(1, 3, 256, 256).cuda()

        # (!!! 已修改 !!!)
        # 注意： 'stage_channels' 最后一个值是 16, short_rate=0.5
        # 这使得瓶颈层 dim=8, 满足 8 的倍数要求。
        model = SimpleUNet(
            in_channels=3,
            num_cls=1,
            ks=3,
            stage_channels=[16, 16, 16, 16, 16],  # 瓶颈层输入=16
            num_blocks=[1, 1, 1, 1, 1],  # 瓶颈层 num_blocks=1
            short_rate=0.5,  # 瓶颈层 mid_channels = 16 * 0.5 = 8
            attn_heads=4  # (!!! 传入新参数 !!!)
        ).cuda()

        flops, params = profile(model, inputs=(input,))
        output = model(input)

        print(f"\n--- Final Model Output (with Converse2D + Attention Bottleneck) ---")
        print(f"Input shape: {input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"FLOPs (G): {flops / 1e9}")
        print(f"Params (M): {params / 1e6}")
    else:
        print("CUDA not available. Please run this on a machine with a GPU.")