import os
from ultralytics import YOLO



DATA_PATH = '/root/NEU-DET/data.yaml'
YAML_PATH = 'ultralytics/cfg/trackers/models/v10/yolov10s_EFC.yaml'  # 根据你实际路径改

# 预训练权重路径（你的实际位置）
WEIGHT_PATH = '/root/yolov10-shiyan/yolov10-main/yolov10s.pt'

print(f"[INFO] 加载模型配置: {YAML_PATH}")
print(f"[INFO] 预训练权重: {WEIGHT_PATH}")
print(f"[INFO] 数据集: {DATA_PATH}")

# ========== 加载结构 + 预训练权重 ==========
model = YOLO(YAML_PATH)
model.load(WEIGHT_PATH)  # ✅ 加载你本地的 yolov10s.pt

print("[INFO] 模型加载成功，Backbone/Neck加载预训练权重，MRFFE层随机初始化")

# ========== 训练（超参和 baseline 完全一致） ==========
model.train(
    data=DATA_PATH,
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    patience=20,
    save=True,
    project='runs/efc',
    name='yolov10s_neu_det_efc',
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    augment=False,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    verbose=True,

)

# ========== 验证 ==========
print("\n" + "="*50)
metrics = model.val(data=DATA_PATH, split='val')
print(f"[RESULT] mAP@0.5: {metrics.box.map50:.4f}")
print(f"[RESULT] mAP@0.5:0.95: {metrics.box.map:.4f}")

