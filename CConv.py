import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosinConv2D(nn.Conv2d):
    """
    Cosine-Similarity Convolution (标量幂缩放)
    - 思路：先对 kernel 与局部 patch 做 L2 归一化，计算 cos 相似度；再做带符号的幂缩放 sign(x)*(|x|+eps)^p。
    - 关键点：
      1) 可选 shared weights（Depthwise 情况自动关闭）；
      2) p 为可学习参数并下限截断到 p_min；
      3) q 为可学习的正标量（通过 log_q 参数化），加入到输入范数分母，稳定训练。
    Inputs : x ∈ (B, C_in, H, W)
    Outputs: y ∈ (B, C_out, H_out, W_out)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 0,
        stride: int = 1,
        groups: int = 1,
        shared_weights: bool = False,
        w_max: float = 1.0,
        p_min: float = 0.1,
        q_init: float = 1e-3,
        eps: float = 1e-6,
    ):
        # 分组合法性
        assert groups == 1 or groups == in_channels, (
            "'groups' needs to be 1 or in_channels "
            f"({in_channels})."
        )
        assert out_channels % groups == 0, (
            f"out_channels ({out_channels}) must be a multiple of groups ({groups})."
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        # depthwise 情况禁用 shared_weights
        self.shared_weights = False if groups == 1 else shared_weights

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=False,
            padding=padding,
            stride=stride,
            groups=groups,
        )

        # 将 kernel_size 存为 int（假设方核）
        self.kernel_size = int(kernel_size)

        # 核心超参/形状
        self.channels_per_kernel = self.in_channels // self.groups
        self.w_max = float(w_max)
        self.p_min = float(p_min)
        self.eps = float(eps)

        # 需要学习的 kernel（支持 shared_weights）
        if self.shared_weights:
            self.n_kernels = self.out_channels // self.groups
        else:
            self.n_kernels = self.out_channels

        scaled_weight = np.random.uniform(
            low=-self.w_max, high=self.w_max,
            size=(self.n_kernels, self.channels_per_kernel, self.kernel_size, self.kernel_size)
        )
        # 覆盖父类的 weight
        self.weight = nn.Parameter(torch.as_tensor(scaled_weight, dtype=self.weight.dtype))

        # 学习的 p（逐输出通道/核）
        p_values = np.random.uniform(low=1.0, high=3.0, size=(1, self.n_kernels, 1, 1))
        self.p = nn.Parameter(torch.as_tensor(p_values, dtype=self.weight.dtype))

        # 学习的 q（正数，通过 log 参数化）
        self.log_q = nn.Parameter(torch.full((1, 1, 1, 1), float(np.log(q_init)), dtype=self.weight.dtype))

        # “全 1 卷积核”作为 buffer，用于输入范数卷积（按设备/类型在 forward 中转换）
        ones = torch.ones(
            self.groups, self.channels_per_kernel, self.kernel_size, self.kernel_size,
            dtype=self.weight.dtype
        )
        self.register_buffer("ones_kernel", ones, persistent=False)

    # ------------------ 核心计算 ------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 约束 learnable 参数范围
        with torch.no_grad():
            self.weight.clamp_(min=-self.w_max, max=self.w_max)
            self.p.clamp_(min=self.p_min)

        p = self.p
        q = torch.exp(self.log_q)

        # 根据 shared_weights 展开权重/参数
        if self.shared_weights:
            weight = self.weight.repeat(self.groups, 1, 1, 1)   # (C_out, Cin/G, k, k)
            p = p.repeat(1, self.groups, 1, 1)                  # (1, C_out, 1, 1)
        else:
            weight = self.weight

        return self._cosine_power_conv(x, weight, p, q)

    def _cosine_power_conv(self, x: torch.Tensor, weight: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # 1) 归一化 kernel
        w_norm = self._weight_norm(weight)                      # (C_out,1,1,1)
        weight_n = weight / (w_norm + self.eps)

        # 2) 计算归一化后的 cos 相似度（分母为输入局部 L2 范数 + q）
        x_norm = self._input_norm(x, q)                         # (B, C_out, H_out, W_out)
        cos_sim = F.conv2d(
            x, weight_n, stride=self.stride, padding=self.padding, groups=self.groups
        ) / (x_norm + self.eps)

        # 3) 带符号幂缩放
        return cos_sim.sign() * (cos_sim.abs() + self.eps) ** p

    def _weight_norm(self, weight: torch.Tensor) -> torch.Tensor:
        # 每个 kernel 的 L2 范数
        return weight.square().sum(dim=(1, 2, 3), keepdim=True).sqrt()

    def _input_norm(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # 用“全 1 卷积核”对 x^2 做卷积，相当于局部窗口 L2 范数（未开方）
        ones = self.ones_kernel.to(device=x.device, dtype=x.dtype)
        x2_sum = F.conv2d(
            x.square(),
            ones,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )  # (B, groups, H_out, W_out)

        # L2 范数 + q
        xnorm = (x2_sum + self.eps).sqrt() + q.to(device=x.device, dtype=x.dtype)

        # 按组扩展到 C_out 通道
        outputs_per_group = self.out_channels // self.groups
        return torch.repeat_interleave(xnorm, repeats=outputs_per_group, dim=1)


# ==================== 示例（统一“五步模板”）====================
if __name__ == "__main__":
    # 1) 配置
    B, H, W, C_in, C_out = 1, 256, 256, 32, 32
    k, s, p, g = 3, 1, 1, 1

    # 2) 构造输入：形状 (B, C_in, H, W)
    x = torch.randn(B, C_in, H, W)

    # 3) 实例化模块（统一命名为 block）
    block = CosinConv2D(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=k,
        padding=p,
        stride=s,
        groups=g,
        shared_weights=False
    )

    # 4) 前向计算（不计算梯度）
    with torch.no_grad():
        y = block(x)

    # 5) 打印结构与张量形状
    print(block)
    print("x.shape =", x.shape)   # [B, C_in, H, W]
    print("y.shape =", y.shape)   # [B, C_out, H_out, W_out]
