#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EFC模块单独测试脚本
"""

import sys
import os

# # 添加 anthor_models 到路径
# sys.path.insert(0, os.path.dirname(__file__))

import torch
from anthor_models.EFC import EFC


def test_efc():
    """测试 EFC 模块"""
    print("=" * 50)
    print("EFC 模块测试")
    print("=" * 50)

    # [1] 初始化模块
    print("\n[1] 初始化 EFC...")
    # 模拟Neck中P4融合：输入P5_up(512)和P4(512)，输出512
    c1 = 512  # 第一个输入通道（如P5上采样后的特征）
    c2 = 512  # 第二个输入通道，也是输出通道（如P4特征）
    efc = EFC(c1=c1, c2=c2)
    print(f"    c1 (输入1通道): {c1}")
    print(f"    c2 (输入2通道/输出通道): {c2}")

    # [2] 生成模拟输入 (模拟Neck融合时的两个特征)
    print("\n[2] 生成模拟输入...")
    batch_size = 2
    height, width = 40, 40  # P4层 40x40分辨率

    # 两个输入：上采样后的深层特征 + 同层特征
    x1 = torch.randn(batch_size, c1, height, width)  # P5上采样后的特征
    x2 = torch.randn(batch_size, c2, height, width)  # P4特征

    print(f"    输入1 x1 (P5_up): {x1.shape}")
    print(f"    输入2 x2 (P4):    {x2.shape}")

    # [3] 前向传播
    print("\n[3] 执行前向传播...")
    efc.eval()
    with torch.no_grad():
        output = efc(x1, x2)  # 注意：两个分开的参数，不是列表

    print(f"    输出形状: {output.shape}")
    print(f"\n[✓] 测试通过！")


if __name__ == '__main__':
    test_efc()