import torch
import torch.nn as nn


class EdgeEnhance(nn.Module):
    """
    边缘感知精炼模块 (EdgeEnhance)
    - 用级联 depthwise 3x3 提取高频边缘/纹理突变特征
    - SE 门控自适应加权，避免干扰已有语义特征
    - 残差连接，可学习绕过（若边缘信息有害则 gate→0）
    """
    def __init__(self, c1, c2):
        super().__init__()
        assert c1 == c2, "Input/output channels must match for residual"
        
        # 级联 depthwise 3x3：等效轻量高通滤波器，提取边缘梯度
        self.edge_dw = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
        )
        
        # 1x1 融合降维
        self.pw_fuse = nn.Conv2d(c1, c2, kernel_size=1, stride=1, bias=False)
        
        # SE 门控：自适应决定注入多少边缘信息
        hidden = max(c2 // 16, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, c2, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 提取高频边缘特征
        edge = self.edge_dw(x)
        edge = self.pw_fuse(edge)
        
        # 门控：如果边缘信息对当前特征有害，自动抑制
        w = self.gate(edge)
        out = edge * w
        
        return x + out  # 残差连接：不破坏 MRFFE-SE-SpatialGate 已学特征


if __name__ == '__main__':
    x = torch.randn(1, 256, 80, 80)
    m = EdgeEnhance(256, 256)
    out = m(x)
    params = sum(p.numel() for p in m.parameters())
    print(f"输入: {x.shape}, 输出: {out.shape}, 无NaN: {not torch.isnan(out).any()}, 参数量: {params/1e6:.3f}M")