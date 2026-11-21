from ultralytics import YOLO

# 加载训练好的最佳权重
model = YOLO('runs/2_MRFFE_MRFFESE_SG/yolov10s_neu_det_2_MRFFE_MRFFESE_SG/weights/best.pt')

# 手动验证
metrics = model.val(data='/root/NEU-DET/data.yaml', split='val')

print("="*50)
print(f"[MRFFE最终结果] mAP@0.5: {metrics.box.map50:.4f}")
print(f"[MRFFE最终结果] mAP@0.5:0.95: {metrics.box.map:.4f}")
print("="*50)

# # 对比 baseline
# baseline_map50 = 0.5164
# improvement = (metrics.box.map50 - baseline_map50) / baseline_map50 * 100
# print(f"[对比] Baseline: {baseline_map50:.4f}")
# print(f"[对比] +MRFFE: {metrics.box.map50:.4f}")
# print(f"[对比] 相对提升: {improvement:+.2f}%")