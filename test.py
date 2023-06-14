from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

model.predict(source='test/a62.jpg', show=False, save=True)