import torch
import torch.nn as nn


# ========== 内嵌轻量 Conv，避免循环导入 ultralytics ==========
class Conv(nn.Module):
    """标准卷积：Conv + BN + SiLU（与 ultralytics 的 Conv 行为完全一致）"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2 if d == 1 else d * (k - 1) // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MRFFE_EFC(nn.Module):
    """
    MRFFE + EFC 组合模块（Concat 拼接版）
    空间多尺度分支(MRFFE) + 通道注意力分支(EFC) → Concat → 融合
    单输入，输入输出通道相等，带残差连接
    """
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert c1 == c2, "Input/output channels must match for residual"
        c_ = int(c2 * e)  # 隐藏层通道数，默认一半
        
        # ========== MRFFE 分支：对称多尺度空间特征 ==========
        # 局部分支：1x1 保持细节
        self.mrffe_local = Conv(c1, c_, 1, 1)
        # 上下文分支：两个 3x3 级联（等效 5x5 感受野，各向同性，无方向偏见）
        self.mrffe_context = nn.Sequential(
            Conv(c1, c_, 3, 1),
            Conv(c_, c_, 3, 1)
        )
        # MRFFE 内部分支先融合降维
        self.mrffe_fuse = Conv(c_ * 2, c_, 1, 1)
        
        # ========== EFC 分支：轻量通道注意力（SE-style，无硬分离） ==========
        self.efc_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = max(c1 // 16, 4)  # 最少 4 通道，防止过小
        self.efc_fc = nn.Sequential(
            nn.Conv2d(c1, hidden_dim, 1, bias=False),
            nn.ReLU(),  # 无 inplace，安全
            nn.Conv2d(hidden_dim, c1, 1, bias=False),
            nn.Sigmoid()
        )
        
        # ========== Concat 融合：MRFFE(c_) + EFC(c1) → 降维回 c2 ==========
        self.fuse = Conv(c_ + c1, c2, 1, 1)
        
    def forward(self, x):
        # MRFFE 分支：提取多尺度空间特征
        x_local = self.mrffe_local(x)
        x_context = self.mrffe_context(x)
        x_mrffe = self.mrffe_fuse(torch.cat([x_local, x_context], dim=1))
        
        # EFC 分支：通道注意力重标定（逐通道加权）
        w = self.efc_pool(x)
        w = self.efc_fc(w)
        x_efc = x * w  # 增强关键通道，抑制背景噪声
        
        # Concat 拼接两个分支（空间信息 + 通道信息互补）
        x_fuse = torch.cat([x_mrffe, x_efc], dim=1)
        out = self.fuse(x_fuse)
        
        return x + out  # 残差连接：学不好就绕过，不破坏原有特征


if __name__ == '__main__':
    x = torch.randn(1, 256, 80, 80)
    m = MRFFE_EFC(256, 256)
    out = m(x)
    print("输入:", x.shape, "输出:", out.shape, "无NaN:", not torch.isnan(out).any())