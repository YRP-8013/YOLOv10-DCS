#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MRFFE 模块单独测试脚本
"""
import sys
import os

# 添加 anthor_models 到路径
# sys.path.insert(0, os.path.dirname(__file__))

import torch
from anthor_models.MRFFE import MRFFE


def test_mrffe():
    """测试 MRFFE 模块"""
    print("=" * 50)
    print("MRFFE 模块测试")
    print("=" * 50)

    # [1] 初始化模块
    print("\n[1] 初始化 MRFFE...")
    in_channels = 256
    mrffe = MRFFE(in_channels=in_channels, reduction=4)  # 修正参数名
    print(f"    输入通道: {in_channels}")
    print(f"    输出通道: {in_channels} (默认与输入相同)")

    # [2] 生成模拟输入 (模拟 YOLOv10 P3 层: 80x80 分辨率)
    print("\n[2] 生成模拟输入...")
    batch_size = 2
    height, width = 80, 80
    x = torch.randn(batch_size, in_channels, height, width)
    print(f"    输入形状: {x.shape}")

    # [3] 前向传播
    print("\n[3] 执行前向传播...")
    mrffe.eval()
    with torch.no_grad():
        output = mrffe(x)

    print(f"    输出形状: {output.shape}")
    print(f"\n[✓] 测试通过！")


if __name__ == '__main__':
    test_mrffe()