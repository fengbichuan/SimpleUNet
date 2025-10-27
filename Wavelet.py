import torch
import torch.nn as nn
import math
import torchvision.transforms as transforms
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1. 小波模块的依赖项 (iAFF 和辅助函数)
# (iAFF 和 Legendrescale, Legendrede 函数保持不变)
# ----------------------------------------------------------------------
class iAFF(nn.Module):
    def __init__(self, channels=8, r=4):
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
        if xo.size(0) == 1 and x.dim() == 3:  # 修复以匹配原始逻辑
            xo = xo.squeeze(0)
        return xo


# <--- 修改 1: 这个函数不再需要，因为我们将在 forward 中直接调用 iaff
# def fusion_process(tensors, iaff_module):
#     fused_batch = iaff_module(tensors, tensors)
#     return fused_batch


def Legendrescale(p):
    # (... 代码未变 ...)
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
    # (... 代码未变 ...)
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

    # 确保张量在正确的设备上
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
    elif p == 3:
        # ... (p=3 logic) ...
        pass


# ----------------------------------------------------------------------
# 2. 这是新的小波模块 (nn.Module)
# ----------------------------------------------------------------------
class WaveletFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, p=2, num_components=8):
        super(WaveletFeatureExtractor, self).__init__()
        self.p = p
        # num_components (8) 仍然代表 Legendrede 产生的总分量数
        self.num_components = num_components

        # <--- 修改 2: iAFF 应该融合两组，每组 4 个通道 (num_components // 2)
        fusion_channels = num_components // 2
        self.iaff = iAFF(channels=fusion_channels)

        # 创建一个灰度转换
        if in_channels == 3:
            self.gray_transform = transforms.Grayscale(num_output_channels=1)
        else:
            self.gray_transform = nn.Identity()

    def forward(self, batch_img):
        device = batch_img.device

        # <--- 修改 3.1: 我们需要两个列表来分别存储两组分量
        decomposed_images_g1 = []  # 存储 (A0, B0, C0, D0)
        decomposed_images_g2 = []  # 存储 (A1, B1, C1, D1)

        for i in range(batch_img.size(0)):  # 遍历批次
            img = batch_img[i]  # (C, H, W)

            # 1. 转换为灰度图 (1, H, W)
            gray_tensor = self.gray_transform(img)  # (1, H, W)
            gray_tensor = gray_tensor.squeeze(0)  # (H, W)

            # 2. 执行2D Legendre分解
            components = Legendrede(gray_tensor.double(), self.p)  # 返回 8 个 (H/2, W/2) 张量

            # <--- 修改 3.2: 将 8 个分量分成两组

            # 组 1: (A0, B0, C0, D0)
            group1_list = [c.float() for c in components[0:4]]
            group1_tensor = torch.stack(group1_list, dim=0)  # (4, H/2, W/2)
            decomposed_images_g1.append(group1_tensor)

            # 组 2: (A1, B1, C1, D1)
            group2_list = [c.float() for c in components[4:8]]
            group2_tensor = torch.stack(group2_list, dim=0)  # (4, H/2, W/2)
            decomposed_images_g2.append(group2_tensor)

        # <--- 修改 3.3: 组合两个批次
        batch_g1 = torch.stack(decomposed_images_g1, dim=0)  # (B, 4, H/2, W/2)
        batch_g2 = torch.stack(decomposed_images_g2, dim=0)  # (B, 4, H/2, W/2)

        # <--- 修改 3.4: 通过 iAFF 融合两个不同的特征组
        output = self.iaff(batch_g1, batch_g2)  # (B, 4, H/2, W/2)

        return output

    # ----------------------------------------------------------------------
    # 3. 测试代码
    # ----------------------------------------------------------------------


if __name__ == "__main__":

    # 1. 定义测试参数
    BATCH_SIZE = 4
    CHANNELS = 3
    HEIGHT = 256
    WIDTH = 256

    # 2. 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Test ---")
    print(f"Using device: {device}")

    # 3. 实例化你的模型
    # p=2, num_components=8 (总共8个分量)
    model = WaveletFeatureExtractor(in_channels=CHANNELS, p=2, num_components=8)
    model.to(device)

    # 4. 创建模拟的输入数据 (B, C, H, W)
    test_input = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH, dtype=torch.float32).to(device)
    print(f"Input tensor shape: {test_input.shape}")

    try:
        # 5. 将模型设置为评估模式
        model.eval()

        # 6. 'no_grad' 上下文
        with torch.no_grad():
            output = model(test_input)

        # 7. 打印输出结果
        print(f"Output tensor shape: {output.shape}")

        # <--- 修改 4: 预期输出通道数现在是 4 (因为融合了两组 4 通道)
        expected_shape = (BATCH_SIZE, 4, HEIGHT // 2, WIDTH // 2)
        assert output.shape == expected_shape

        print(f"Success! Output shape {output.shape} matches expected {expected_shape}.")

    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"An error occurred during the model's forward pass:")
        print(e)
        import traceback

        traceback.print_exc()

    print(f"--- Test Complete ---")