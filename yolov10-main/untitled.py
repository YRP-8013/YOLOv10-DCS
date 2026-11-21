from ultralytics import YOLO
model = YOLO('runs/MRFFE_SE_SG/yolov10s_neu_det_MRFFE_MRFFESE_SG/weights/best.pt')
model.val(
    data='/root/NEU-DET/data.yaml',
    split='val',
    save_json=True,   # 保存结果json
    iou=0.3,
    plots=True        # 自动生成PR曲线、混淆矩阵等图
)
