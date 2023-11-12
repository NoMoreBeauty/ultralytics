from ultralytics import YOLO

# Load a model
model = YOLO('yolov3u.pt') 

model.predict('000483.png', save=True, save_crop=True)