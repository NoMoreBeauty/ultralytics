import os
import random
import shutil

train_data_path = "D:\\workspace\\ultralytics\\datasets\\kitti\\images\\train"
train_label_path = "D:\\workspace\\ultralytics\\datasets\\kitti\\labels\\train"
eval_data_path = "D:\\workspace\\ultralytics\\datasets\\kitti\\images\\eval"
eval_label_path = "D:\\workspace\\ultralytics\\datasets\\kitti\\labels\\eval"

L = []
r = 0.8

for data in os.listdir(train_data_path):
    image_path = os.path.join(train_data_path, data)
    label_path = os.path.join(train_label_path, data[0:6]+".txt")
    if os.path.exists(image_path) and os.path.exists(label_path):
        # print(image_path)
        L.append([image_path, label_path])

print(len(L))
random.shuffle(L)
num_train = int(len(L)* r) 
num_eval = len(L) - num_train 
print(num_eval)
print(num_train)

print("moving ...")
try:
    for i in range(num_eval):
        #print(L[i][0])
        shutil.move(L[i][0],eval_data_path)
        shutil.move(L[i][1],eval_label_path)
except Exception:
    print("error ")
else:
    print("success !")

# test
# shutil.move("D:\\workspace\\ultralytics\\datasets\\kitti\\images\\train\\003430.png","D:\\workspace\\ultralytics\\datasets")