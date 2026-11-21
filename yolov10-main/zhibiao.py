from ultralytics import YOLO

# 加载你训练好的最佳权重（以三段串联 79.07% 的模型为例）
model = YOLO('runs/MRFFE_MRFFESE_SG/yolov10s_neu_det_MRFFE_MRFFESE_SG/weights/best.pt')

# 运行验证
metrics = model.val(data='/root/NEU-DET/data.yaml', split='val')

print("="*60)
print(f"[整体指标]")
print(f"mAP@0.5:      {metrics.box.map50:.4f}  ({metrics.box.map50*100:.2f}%)")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}   ({metrics.box.map*100:.2f}%)")
print(f"平均 Precision: {metrics.box.mp:.4f}")
print(f"平均 Recall:    {metrics.box.mr:.4f}")
print("="*60)

# 逐类详细指标
print(f"\n[逐类指标 - NEU-DET 6类缺陷]")
print(f"{'类别':<<20} {'Precision':>10} {'Recall':>10} {'AP50':>10} {'AP50-95':>10}")
print("-"*60)

class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 
               'rolled-in_scale', 'scratches']

for i, name in enumerate(class_names):
    p = metrics.box.p[i] if metrics.box.p is not None else 0
    r = metrics.box.r[i] if metrics.box.r is not None else 0
    ap50 = metrics.box.ap50[i] if metrics.box.ap50 is not None else 0
    ap = metrics.box.ap[i] if metrics.box.ap is not None else 0
    print(f"{name:<20} {p:>10.4f} {r:>10.4f} {ap50:>10.4f} {ap:>10.4f}")

print("="*60)