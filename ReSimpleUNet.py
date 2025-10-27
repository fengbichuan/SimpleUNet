import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


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


# ----------------------------------------------------------------------
# 3. 跳跃连接处理模块 (保留了您的 print)
# ----------------------------------------------------------------------
class SkipConnections(nn.Module):
    """
    处理编码器传来的跳跃连接特征图。
    """

    def __init__(self, stage_channels, short_rate):
        super().__init__()
        self.depth = len(stage_channels)
        self.short_layers = nn.ModuleList()

        for i in range(self.depth - 1):
            layer = SingleConv(stage_channels[i], int(short_rate * stage_channels[i]), 1, 0, 1)
            in_ch = stage_channels[i]
            out_ch = int(short_rate * stage_channels[i])
            #print(f"Skip Layer {i}: InChannels={in_ch}, OutChannels={out_ch}")
            self.short_layers.append(layer)

    def forward(self, shortcuts):
        refined_shortcuts = []
        for i in range(len(shortcuts)):
            input_tensor = shortcuts[i]
            #print(f"Skip Layer {i} Input (BCHW):  {input_tensor.shape}")
            refined = self.short_layers[i](shortcuts[i])
            # 4. 打印输出张量的BCHW形状
            #print(f"Skip Layer {i} Output (BCHW): {refined.shape}")
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


# ----------------------------------------------------------------------
# 6. 重构后的 SimpleUNet (主模块) (!!! 已修改 !!!)
# ----------------------------------------------------------------------
class SimpleUNet(nn.Module):
    # 移除了 'adw' 参数
    def __init__(self, in_channels, num_cls, ks=3, dilation=1, stage_channels=5 * [32], num_blocks=5 * [1],
                 short_rate=0.5):
        super(SimpleUNet, self).__init__()
        assert short_rate > 0, 'short_rate must be greater than 0!'
        assert len(stage_channels) == len(num_blocks), 'The length of stage_channels and num_blocks must match!'

        self.pad = dilation * (ks - 1) // 2

        # 1. 实例化编码器
        self.encoder = Encoder(in_channels, stage_channels, num_blocks, ks, self.pad, dilation)

        # 2. 实例化跳跃连接处理器
        self.skip_connections = SkipConnections(stage_channels, short_rate)

        # 3. 实例化瓶颈层
        self.bottleneck = Bottleneck(
            in_channels=stage_channels[-1],  # 编码器最深层的输出通道
            mid_channels=int(short_rate * stage_channels[-1]),  # 瓶颈层内部通道
            num_blocks=num_blocks[-1],  # 瓶颈层的块数
            ks=ks,
            pad=self.pad,
            dilation=dilation
        )

        # 4. 实例化解码器 (移除了 'adw' 参数传递)
        self.decoder = Decoder(stage_channels, num_blocks, short_rate, ks, self.pad, dilation)

        # 5. 实例化最终的分割头 (1x1 卷积)
        self.seg_head = SingleConv(int(short_rate * stage_channels[0]), num_cls, 1, 0, 1)

    def forward(self, x):
        # 1. 编码器路径
        x_to_bottleneck, shortcuts = self.encoder(x)
        for i in range(4):
            print("shortcut:", shortcuts[i].shape)

        # 2. 处理跳跃连接
        refined_shortcuts = self.skip_connections(shortcuts)

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