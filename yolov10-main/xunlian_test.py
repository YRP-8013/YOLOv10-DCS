#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

# sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO
from anthor_models.MRFFE import MRFFE
from anthor_models.EFC import EFC

def test_model():
    print("=" * 60)
    print("验证 YOLOv10 + MRFFE + EFC 模型创建")
    print("=" * 60)

    try:
        # 1. 加载配置
        print(r"\n[1] 加载 YAML 配置...")
        model = YOLO(r'D:\yolov10-shiyan\yolov10-main\ultralytics\cfg\models\v10\yolov10s_ME.yaml')
        print("    ✓ 配置加载成功")

        # 2. 检查模型结构
        print(r"\n[2] 检查模型结构...")
        model_str = str(model.model)

        checks = {
            'MRFFE': 'MRFFE' in model_str,
            'EFC': 'EFC' in model_str,
        }

        for name, found in checks.items():
            status = "✓" if found else "✗"
            print(f"    {status} 包含 {name}: {found}")

        if not all(checks.values()):
            print("\n    ✗ 模型结构检查失败")
            return False

        # 3. 测试前向传播
        print("\n[3] 测试前向传播...")
        import torch
        x = torch.randn(1, 3, 640, 640)  # 小尺寸测试

        model.model.eval()
        with torch.no_grad():
            output = model.model(x)

        print(f"    ✓ 输入: {x.shape}")
        print(f"    ✓ 输出数量: {len(output)}")
        print(f"    ✓ 前向传播成功")

        # 4. 尝试加载预训练权重
        print("\n[4] 尝试加载预训练权重...")
        try:
            model.load('yolov10s.pt')
            print("    ✓ 权重加载成功（部分匹配）")
        except Exception as e:
            print(f"    ⚠ 权重加载警告: {e}")
            print("    （新模块无预训练权重，这是正常的）")

        print("\n" + "=" * 60)
        print("✓ 模型创建验证通过！可以开始训练")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_model()
    sys.exit(0 if success else 1)