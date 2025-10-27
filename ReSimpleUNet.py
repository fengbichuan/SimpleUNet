import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
import torchvision.transforms as transforms

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
    """
    U-Net的编码器（下采样）路径。
    它处理输入，并逐层下采样，同时保存用于跳跃连接的特征图。
    """

    def __init__(self, in_channels, stage_channels, num_blocks, ks, pad, dilation):
        super().__init__()
        self.depth = len(stage_channels)
        self.down = nn.MaxPool2d(2)

        # 第一层
        self.en_layer0 = SingleConv(in_channels, stage_channels[0], ks, pad, dilation)

        # 后续的编码器层
        self.en_layers = nn.ModuleList()
        for i in range(1, self.depth):
            layers = []
            for _ in range(num_blocks[i] - 1):
                layers.append(SingleConv(stage_channels[i - 1], stage_channels[i - 1], ks, pad, dilation))
            layers.append(SingleConv(stage_channels[i - 1], stage_channels[i], ks, pad, dilation))
            self.en_layers.append(nn.Sequential(*layers))

    def forward(self, x):
        shortcuts = []

        # Layer 0
        x = self.en_layer0(x)
        shortcuts.append(x)



        # Layers 1 到 depth-2 (倒数第二层)
        for i in range(self.depth - 2):
            x = self.down(x)
            x = self.en_layers[i](x)
            shortcuts.append(x)

        # Layer depth-1 (最后一层)
        x = self.down(x)
        x = self.en_layers[self.depth - 2](x)

        return x, shortcuts  # x 是送入瓶颈层的输入, shortcuts 是跳跃连接列表





##############添加小波跳跃链接


class iAFF(nn.Module):
    def __init__(self, channels=4, r=4):  # 确保 channels 默认为 4
        super(iAFF, self).__init__()
        inter_channels = max(1, int(channels // r))

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        if x.dim() == 3:
            x = x.unsqueeze(0)
            residual = residual.unsqueeze(0)
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)
        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        if xo.size(0) == 1 and x.dim() == 3:
            xo = xo.squeeze(0)
        return xo


def Legendrescale(p):
    # (... 粘贴 Legendrescale 完整代码 ...)
    if p == 2:
        H0 = torch.tensor([[1 / math.sqrt(2), 0], [-math.sqrt(3) / (2 * math.sqrt(2)), 1 / (2 * math.sqrt(2))]])
        G0 = torch.tensor([[0, -1 / math.sqrt(2)], [1 / (2 * math.sqrt(2)), math.sqrt(3) / (2 * math.sqrt(2))]])
        H1 = torch.zeros((p, p))
        G1 = torch.zeros((p, p))
        for i in range(p):
            for j in range(p):
                H1[i][j] = (-1) ** (i + j - 2) * H0[i][j]
                G1[i][j] = (-1) ** (i + j + p - 2) * G0[i][j]
        H = torch.cat((H0, H1), 1)
        G = torch.cat((G0, G1), 1)
    elif p == 3:
        H1 = torch.zeros((p, p))
        G1 = torch.zeros((p, p))
        H0 = torch.tensor([[1 / math.sqrt(2), 0, 0],
                           [-math.sqrt(3) / (2 * math.sqrt(2)), 1 / (2 * math.sqrt(2)), 0],
                           [0, -math.sqrt(3) * math.sqrt(5) / (4 * math.sqrt(2)), 1 / (4 * math.sqrt(2))]])
        G0 = torch.tensor(
            [[-math.sqrt(5) / (6 * math.sqrt(2)), -math.sqrt(15) / (6 * math.sqrt(2)), 2 / (3 * math.sqrt(2))],
             [0, 1 / (4 * math.sqrt(2)), math.sqrt(15) / (4 * math.sqrt(2))],
             [-1 / (3 * math.sqrt(2)), -1 / math.sqrt(6), (-math.sqrt(5) / (3 * math.sqrt(2)))]])
        for i in range(p):
            for j in range(p):
                H1[i][j] = (-1) ** (i + j - 2) * H0[i][j]
                G1[i][j] = (-1) ** (i + j + p - 2) * G0[i][j]
        H = torch.cat((H0, H1), 1)
        G = torch.cat((G0, G1), 1)
    return H, G


def Legendrede(img, p):
    # (... 粘贴 Legendrede 完整代码 ...)
    X = img
    V = X.shape[1]  # % 图像维数
    p = 2;  # %两个小波
    T = torch.as_tensor(X)
    Y = T[..., ::2, ::]
    X = T[..., 1::2, ::]
    if p == 2:
        xy = torch.cat([Y.unsqueeze(0), Y.unsqueeze(0)], 0)
        xyff = torch.cat([X.unsqueeze(0), X.unsqueeze(0)], 0)
    elif p == 3:
        xy = torch.cat([Y.unsqueeze(0), Y.unsqueeze(0), Y.unsqueeze(0)], 0)
        xyff = torch.cat([X.unsqueeze(0), X.unsqueeze(0), X.unsqueeze(0)], 0)

    device = img.device
    xy = torch.as_tensor(xy, dtype=torch.float64).to(device)
    xyff = torch.as_tensor(xyff, dtype=torch.float64).to(device)
    [H, G] = Legendrescale(p)
    H0 = torch.as_tensor(H[:, :p], dtype=torch.float64).to(device)
    H1 = torch.as_tensor(H[:, p:], dtype=torch.float64).to(device)
    G0 = torch.as_tensor(G[:, :p], dtype=torch.float64).to(device)
    G1 = torch.as_tensor(G[:, p:], dtype=torch.float64).to(device)
    xy_reshaped = xy.view(p, -1).to(device)
    xyff_reshaped = xyff.view(p, -1).to(device)
    s = torch.matmul(H0, xy_reshaped) + torch.matmul(H1, xyff_reshaped)
    d = torch.matmul(G0, xy_reshaped) + torch.matmul(G1, xyff_reshaped)
    s = s.view(p, V // 2, V)
    d = d.view(p, V // 2, V)
    target_s = torch.zeros((p, V // 2, V)).to(device)
    target_d = torch.zeros((p, V // 2, V)).to(device)
    target_s[:, :, :] = s
    target_d[:, :, :] = d
    row = torch.zeros((p, V, V)).to(device)
    row[:, :V // 2, :] = s
    row[:, V // 2:, :] = d
    xr = row[..., ::, ::2].to(device)
    xf = row[..., ::, 1::2].to(device)
    xr = torch.as_tensor(xr, dtype=torch.float64).to(device)
    xf = torch.as_tensor(xf, dtype=torch.float64).to(device)
    xr_reshaped = xr.view(p, -1).to(device)
    xf_reshaped = xf.view(p, -1).to(device)
    s = torch.matmul(H0, xr_reshaped) + torch.matmul(H1, xf_reshaped).to(device)
    d = torch.matmul(G0, xr_reshaped) + torch.matmul(G1, xf_reshaped).to(device)
    sv1 = s.view(p, V, V // 2).to(device)
    dv1 = d.view(p, V, V // 2).to(device)
    target_sv1 = torch.zeros((p, V, V // 2)).to(device)
    target_dv1 = torch.zeros((p, V, V // 2)).to(device)
    target_sv1[:, :, :] = sv1
    target_dv1[:, :, :] = dv1
    decomposition = torch.zeros((p, V, V)).to(device)
    decomposition[:, :, :V // 2] = sv1
    decomposition[:, :, V // 2:] = dv1
    x = torch.zeros((p, V // 2, V // 2)).to(device)
    xr = torch.zeros((p, V // 2, V // 2)).to(device)
    r = torch.zeros((p, V // 2, V // 2)).to(device)
    rr = torch.zeros((p, V // 2, V // 2)).to(device)
    for k0 in range(p):
        x[k0, :, :] = decomposition[k0, :V // 2, :V // 2]
        xr[k0, :, :] = decomposition[k0, :V // 2, V // 2:]
        r[k0, :, :] = decomposition[k0, V // 2:, :V // 2]
        rr[k0, :, :] = decomposition[k0, V // 2:, V // 2:]
    if p == 2:
        A0, B0, C0, D0 = x[0, :, :], xr[0, :, :], r[0, :, :], rr[0, :, :]
        A1, B1, C1, D1 = x[1, :, :], xr[1, :, :], r[1, :, :], rr[1, :, :]
        return A0, B0, C0, D0, A1, B1, C1, D1


# 这是我们的小波模块 (已修改为输出 4 通道)
class WaveletFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, p=2, num_components=8):
        super(WaveletFeatureExtractor, self).__init__()
        self.p = p
        self.num_components = num_components

        fusion_channels = num_components // 2  # 4
        self.iaff = iAFF(channels=fusion_channels)

        if in_channels == 3:
            self.gray_transform = transforms.Grayscale(num_output_channels=1)
        else:
            self.gray_transform = nn.Identity()

    def forward(self, batch_img):
        device = batch_img.device
        decomposed_images_g1 = []
        decomposed_images_g2 = []

        for i in range(batch_img.size(0)):
            img = batch_img[i]
            gray_tensor = self.gray_transform(img)
            gray_tensor = gray_tensor.squeeze(0)
            components = Legendrede(gray_tensor.double(), self.p)

            group1_list = [c.float() for c in components[0:4]]
            group1_tensor = torch.stack(group1_list, dim=0)
            decomposed_images_g1.append(group1_tensor)

            group2_list = [c.float() for c in components[4:8]]
            group2_tensor = torch.stack(group2_list, dim=0)
            decomposed_images_g2.append(group2_tensor)

        batch_g1 = torch.stack(decomposed_images_g1, dim=0)
        batch_g2 = torch.stack(decomposed_images_g2, dim=0)
        output = self.iaff(batch_g1, batch_g2)  # (B, 4, H/2, W/2)
        return output


class SkipConnections(nn.Module):
    """
    处理跳跃连接，并融合小波特征。
    """

    # <--- 修改 1: __init__ 接收原始图像的 in_channels
    def __init__(self, in_channels, stage_channels, short_rate):
        super().__init__()
        self.depth = len(stage_channels)

        # <--- 修改 2: 实例化你的小波模块
        # 它将用于处理 128x128 级别的特征
        self.wavelet_enc = WaveletFeatureExtractor(in_channels, p=2, num_components=8)
        self.wavelet_out_channels = 4  # 我们的模块输出 4 个通道

        self.short_layers = nn.ModuleList()

        for i in range(self.depth - 1):  # 循环 4 次 (i = 0, 1, 2, 3)
            in_ch = stage_channels[i]  # 64, 128, 256, 512
            out_ch = int(short_rate * stage_channels[i])  # 32, 64, 128, 256

            # <--- 修改 3: 针对 i=1 (128x128层) 的特殊处理
            if i == 1:
                # 原始_in_ch = 128, 小波_in_ch = 4
                # 融合后的通道 = 128 + 4 = 132
                in_ch_fused = in_ch + self.wavelet_out_channels

                # 这个 1x1 卷积将处理融合后的 132 通道，并压缩到 64
                layer = SingleConv(in_ch_fused, out_ch, 1, 0, 1)
                # print(f"Skip Layer {i} (Fused): InChannels={in_ch_fused}, OutChannels={out_ch}")
            else:
                # 其他层保持不变
                layer = SingleConv(in_ch, out_ch, 1, 0, 1)
                # print(f"Skip Layer {i} (Standard): InChannels={in_ch}, OutChannels={out_ch}")

            self.short_layers.append(layer)

    # <--- 修改 4: forward 接收原始图像 x_original
    def forward(self, x_original, shortcuts):

        # 1. 仅在开始时计算一次小波特征
        # x_original 是 (B, 3, 256, 256)
        # wavelet_feat 是 (B, 4, 128, 128)
        wavelet_feat = self.wavelet_enc(x_original)

        refined_shortcuts = []
        for i in range(len(shortcuts)):  # 循环 4 次 (i = 0, 1, 2, 3)
            skip_tensor = shortcuts[i]

            # <--- 修改 5: 在 i=1 (128x128层) 执行拼接
            if i == 1:
                # 确保数据类型和设备一致
                wavelet_feat = wavelet_feat.to(skip_tensor.device, dtype=skip_tensor.dtype)

                # 安全检查：以防万一输入图像不是 256x256
                if wavelet_feat.shape[2:] != skip_tensor.shape[2:]:
                    wavelet_feat = F.interpolate(wavelet_feat, size=skip_tensor.shape[2:], mode='bilinear',
                                                 align_corners=False)

                # 拼接：(B, 128, 128, 128) + (B, 4, 128, 128) -> (B, 132, 128, 128)
                input_tensor = torch.cat([skip_tensor, wavelet_feat], dim=1)
            else:
                # 其他层不变
                input_tensor = skip_tensor

            # 2. 将 input_tensor (可能是原始的，也可能是融合后的) 送入 1x1 卷积
            refined = self.short_layers[i](input_tensor)
            refined_shortcuts.append(refined)

        return refined_shortcuts


# ----------------------------------------------------------------------
# 4. 瓶颈层 (Bottleneck) 模块 (不变)
# ----------------------------------------------------------------------
class Bottleneck(nn.Module):
    """
    U-Net最底部的瓶颈层。
    """

    def __init__(self, in_channels, mid_channels, num_blocks, ks, pad, dilation):
        super().__init__()
        layers = []
        # 第一个卷积: in_channels -> mid_channels
        layers.append(SingleConv(in_channels, mid_channels, ks, pad, dilation))
        # 后续 (num_blocks - 1) 个卷积: mid_channels -> mid_channels
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
    接收来自Bottleneck和SkipConnections的特征，并逐层上采样。
    """

    # 移除了 'adw' 参数
    def __init__(self, stage_channels, num_blocks, short_rate, ks, pad, dilation):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.depth = len(stage_channels)

        re_stage_channels = stage_channels[::-1]
        re_num_blocks = num_blocks[::-1]

        self.de_layers = nn.ModuleList()

        for i in range(1, self.depth):  # 循环创建 (depth-1) 个解码器块
            # 解码器卷积块
            de_layers_block = []

            # 拼接后的输入通道计算保持不变，因为它基于SkipConnections和上一级Decoder的输出
            in_ch_concat = int(short_rate * re_stage_channels[i - 1]) + int(short_rate * re_stage_channels[i])
            out_ch = int(short_rate * re_stage_channels[i])

            de_layers_block.append(SingleConv(in_ch_concat, out_ch, ks, pad, dilation))

            for _ in range(re_num_blocks[i] - 1):
                de_layers_block.append(SingleConv(out_ch, out_ch, ks, pad, dilation))
            self.de_layers.append(nn.Sequential(*de_layers_block))

            # 2. 注意力权重 (Alpha 和 Beta) <-- 已移除

    def forward(self, x_from_bottleneck, refined_shortcuts):
        # 将跳跃连接反转，以便从深到浅使用
        re_shortcuts = refined_shortcuts[::-1]

        x = x_from_bottleneck  # 从瓶颈层的输出开始

        # 循环上采样
        for j in range(self.depth - 1):  # 循环 (depth-1) 次

            # 1. 对来自深层的特征 'x' 进行上采样
            x_up = self.up(x)

            # 2. 获取对应的跳跃连接
            shortcut = re_shortcuts[j]

            # 3. 标准U-Net融合：直接拼接 (不再使用 alpha 和 beta)
            y = torch.concat([shortcut, x_up], dim=1)

            # 4. 通过解码器卷积块
            x = self.de_layers[j](y)

        return x  # 返回解码器最后一层的输出


class SimpleUNet(nn.Module):
    def __init__(self, in_channels, num_cls, ks=3, dilation=1, stage_channels=5 * [32], num_blocks=5 * [1],
                 short_rate=0.5):
        super(SimpleUNet, self).__init__()
        assert short_rate > 0, 'short_rate must be greater than 0!'
        assert len(stage_channels) == len(num_blocks), 'The length of stage_channels and num_blocks must match!'

        self.pad = dilation * (ks - 1) // 2
        self.encoder = Encoder(in_channels, stage_channels, num_blocks, ks, self.pad, dilation)

        # <--- 修改 6: 将 in_channels 传递给 SkipConnections
        self.skip_connections = SkipConnections(in_channels, stage_channels, short_rate)

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
        # 1. 编码器路径
        x_to_bottleneck, shortcuts = self.encoder(x)

        # 打印原始 shortcuts 形状 (用于调试)
        # for i in range(4):
        #     print(f"Original shortcut[{i}]:", shortcuts[i].shape)

        # <--- 修改 7: 将原始图像 x 传递给 skip_connections
        refined_shortcuts = self.skip_connections(x, shortcuts)

        # 3. 瓶颈层
        x_after_bottleneck = self.bottleneck(x_to_bottleneck)

        # 4. 解码器路径
        x = self.decoder(x_after_bottleneck, refined_shortcuts)

        # 5. 分割头
        output = self.seg_head(x)
        return output
# ----------------------------------------------------------------------
# 7. 测试代码 (!!! 已修改 !!!)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 确保有可用的CUDA设备
    if torch.cuda.is_available():
        input = torch.randn(1, 3, 256, 256).cuda()

        # 移除了 'adw=True'
        model = SimpleUNet(in_channels=3, num_cls=1, stage_channels=[64, 128, 256, 512, 1024], num_blocks=[1, 1, 1, 1, 1],
                           short_rate=0.5).cuda()

        flops, params = profile(model, inputs=(input,))
        output = model(input)

        print(f"\n--- Final Model Output ---")
        print(f"Input shape: {input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"FLOPs (G): {flops / 1e9}")
        print(f"Params (M): {params / 1e6}")
    else:
        print("CUDA not available. Please run this on a machine with a GPU.")