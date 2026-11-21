from ultralytics import YOLO

# 加载训练好的最佳权重
model = YOLO('runs/yolov10s_MRFFE_MRFFESE_SG_EE/yolov10s_neu_det_yolov10s_MRFFE_MRFFESE_SG_EE/weights/best.pt')

# 手动验证
metrics = model.val(data='/root/NEU-DET/data.yaml', split='val')

print("="*50)
print(f"[MRFFE_MRFFESE_SG_EE最终结果] mAP@0.5: {metrics.box.map50:.4f}")
print(f"[MRFFE_MRFFESE_SG_EE最终结果] mAP@0.5:0.95: {metrics.box.map:.4f}")
print("="*50)