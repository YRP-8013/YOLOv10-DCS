# test_register.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ultralytics.nn.modules import MRFFE, EFC

print("✓ MRFFE:", MRFFE)
print("✓ EFC:", EFC)
print("✓ 模块注册成功！")



























# if __name__ == '__main__':
#     # 加载训练好的权重文件
#     model = YOLOv10(r"D:\newyolov10\yolov10-main\yolov10-main\train_6\weights\best.pt")
#     metrics = model.val(data=r"D:\newyolov10\yolov10-main\yolov10-main\orange_1.yaml",
#                         imgsz=640,
#                         device=0)
#     print('mAP50-95:', metrics.box.map)
#     print('mAP50   :', metrics.box.map50)
#
#
#



# from ultralytics import YOLO
#
# if __name__ == '__main__':
#     try:
#         # 尝试加载模型
#         model = YOLO(r"D:\newyolov10\yolov10-main\yolov10-main\train_6\weights\best.pt")
#         print("模型加载成功！")
#         print("模型配置信息：", model.model)  # 打印模型结构信息
#     except Exception as e:
#         print("模型加载失败！错误信息：", e)











