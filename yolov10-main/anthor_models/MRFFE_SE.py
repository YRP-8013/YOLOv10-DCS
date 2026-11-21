import torch
import torch.nn as nn


# 内嵌轻量 Conv，避免循环导入
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2 if d == 1 else d * (k - 1) // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MRFFE_SE(nn.Module):
    """
    对称 MRFFE + 轻量 SE 注意力
    - 对称卷积消除方向偏见（保护 scratches/inclusion）
    - SE 注意力筛选通道（抑制背景噪声）
    - 只输出 c2 通道，无 concat 冗余，参数量极简
    """
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert c1 == c2, "Input/output channels must match for residual"
        c_ = int(c2 * e)
        
        # ========== 对称 MRFFE：两个分支都是 3×3（各向同性） ==========
        # 局部分支：1×1 保持细节
        self.local = Conv(c1, c_, 1, 1)
        # 上下文分支：两个 3×3 对称卷积（等效 5×5 感受野，无方向偏见）
        self.context = nn.Sequential(
            Conv(c1, c_, 3, 1),
            Conv(c_, c_, 3, 1)
        )
        # 融合降维：直接输出 c2 通道
        self.fuse = Conv(c_ * 2, c2, 1, 1)
        
        # ========== 轻量 SE 注意力：只重标定 fuse 后的通道 ==========
        hidden = max(c2 // 16, 4)
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(c2, hidden, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, c2, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 对称多尺度空间特征
        x_local = self.local(x)
        x_context = self.context(x)
        x_fused = self.fuse(torch.cat([x_local, x_context], dim=1))
        
        # SE 通道注意力：增强有用通道，抑制噪声（比 EFC 轻量 10 倍）
        w = self.se(self.se_pool(x_fused))
        out = x_fused * w
        
        return x + out  # 残差连接：SE 只重标定，不改变空间结构


if __name__ == '__main__':
    x = torch.randn(1, 256, 80, 80)
    m = MRFFE_SE(256, 256)
    out = m(x)
    print("输入:", x.shape, "输出:", out.shape, "无NaN:", not torch.isnan(out).any())
    # 参数量统计
    params = sum(p.numel() for p in m.parameters())
    print(f"参数量: {params/1e6:.3f}M")