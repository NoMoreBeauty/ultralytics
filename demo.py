from ultralytics import YOLO

# Load a model
model = YOLO('C:\\Users\\wangjiahui\\Desktop\\浙软\\2023\\秋\\智能驾驶中的深度学习应用与实践\\作业\\hw2\\ultralytics\\runs\\train\\weights\\best.pt') 

model.predict('000483.png', save=True, save_crop=True)