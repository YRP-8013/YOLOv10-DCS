import os
from ultralytics import YOLO

os.environ['OMP_NUM_THREADS'] = '1'

DATA_PATH = '/root/NEU-DET/data.yaml'
WEIGHT_PATH = '/root/yolov10-shiyan/yolov10-main/yolov10s.pt'

print(f"[INFO] 加载 YOLOv10s: {WEIGHT_PATH}")
print(f"[INFO] 数据集: {DATA_PATH}")

# ✅ 直接加载 .pt，自动解析 v10Detect 结构 + 权重
model = YOLO(WEIGHT_PATH)

# 验证检测头（调试用，确认后删掉也行）
print(f"[DEBUG] 检测头类型: {type(model.model.model[-1]).__name__}")
# 必须输出: v10Detect

print("[INFO] YOLOv10s 加载成功")

# ========== 训练 ==========
model.train(
    data=DATA_PATH,
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    patience=20,
    save=True,
    project='runs/baseline_train',
    name='yolov10s_',
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
metrics = model.val(
    data=DATA_PATH,
    split='val',
    plots=True,
    save_json=True,
)
print(f"[YOLOv10s] mAP@0.5: {metrics.box.map50:.4f}")
print(f"[YOLOv10s] mAP@0.5:0.95: {metrics.box.map:.4f}")