import torch
import torch.nn as nn


class SpatialGate(nn.Module):
    """
    空间门控模块：用深度可分离卷积生成空间注意力图
    与 SE（通道注意力）正交互补：SE 决定"哪些通道重要"，SpatialGate 决定"哪些位置重要"
    """
    def __init__(self, c1, c2):
        super().__init__()
        assert c1 == c2
        # 深度可分离卷积：参数量极小，只学习空间关系
        self.dwconv = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 生成空间权重图：(N, C, H, W) → 每个空间位置一个重要性权重
        attn = self.sigmoid(self.bn(self.dwconv(x)))
        out = x * attn  # 逐像素加权：缺陷中心增强，边缘噪声抑制
        return x + out  # 残差连接


if __name__ == '__main__':
    x = torch.randn(1, 256, 80, 80)
    m = SpatialGate(256, 256)
    out = m(x)
    print("输入:", x.shape, "输出:", out.shape, "无NaN:", not torch.isnan(out).any())
    params = sum(p.numel() for p in m.parameters())
    print(f"参数量: {params/1e6:.3f}M")  # 约 0.06M