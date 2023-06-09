from ultralytics import YOLO

model = YOLO('yolov8m.pt')

model.train(data='data.yaml', epochs=100, imgsz=256, patience=0)