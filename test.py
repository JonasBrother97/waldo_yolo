from ultralytics import YOLO

model = YOLO('best.pt')

model.predict(source='test/images/19_2_3.jpg', show=False, save=True)