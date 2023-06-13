from ultralytics import YOLO

model = YOLO('best.pt')

model.predict(source='test/images/b12.jpg', show=False, save=True)