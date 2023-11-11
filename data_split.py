import os
import random
import shutil
import argparse

train_data_path = "datasets/kitti/images/train"
train_label_path = "datasets/kitti/labels/train"
eval_data_path = "datasets/kitti/images/eval"
eval_label_path = "datasets/kitti/labels/eval"

def split_data(train_data_path, train_label_path, eval_data_path, eval_label_path, r):
    L = []
    for data in os.listdir(train_data_path):
        image_path = os.path.join(train_data_path, data)
        label_path = os.path.join(train_label_path, data[0:6]+".txt")
        if os.path.exists(image_path) and os.path.exists(label_path):
            L.append([image_path, label_path])

    random.shuffle(L)
    num_train = int(len(L)* r) 
    num_eval = len(L) - num_train 

    print("moving ...")
    try:
        print()
        print()
        print()
        print()
        print()
        # for i in range(num_eval):
        #     shutil.move(L[i][0],eval_data_path)
        #     shutil.move(L[i][1],eval_label_path)
    except Exception:
        print("error ")
    else:
        print("success !")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SPLIT DATA INTO TRAIN/TEST BY R")
    parser.add_argument("--train_data_path", default="datasets/kitti/images/train", type=str, help="path of train data")
    parser.add_argument("--train_label_path", default="datasets/kitti/labels/train", type=str, help="path of train label")
    parser.add_argument("--eval_data_path", default="datasets/kitti/images/eval", type=str, help="path of eval data")
    parser.add_argument("--eval_label_path", default="datasets/kitti/labels/eval", type=str, help="path of eval label")
    parser.add_argument("--r", type=float, default=0.8, help="percentage of the samples to be used for training between 0.0 and 1.0.")
    args = parser.parse_args()
    print(args.train_data_path)
    print(args.train_label_path)
    print(args.eval_data_path)
    print(args.eval_label_path)
    print(args.r)
# python .\data_split.py --train_data_path datasets/kitti/images/train --train_label_path datasets/kitti/labels/train --eval_data_path datasets/kitti/images/eval --eval_label_path datasets/kitti/labels/eval --r 0.8