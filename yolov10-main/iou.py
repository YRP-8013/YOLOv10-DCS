from ultralytics import YOLO
model = YOLO('runs/MRFFE_SE_SG/yolov10s_neu_det_MRFFE_MRFFESE_SG/weights/best.pt')
train_args = model.ckpt['train_args']
print(f"iou = {train_args.get('iou')}")
print(f"conf = {train_args.get('conf')}")
