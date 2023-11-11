<div align="center">

[简体中文](README.zh-CN.md)
<br>

[Ultralytics](https://ultralytics.com) [YOLOv8](https://github.com/ultralytics/ultralytics) 是一款前沿、最先进（SOTA）的模型，基于先前 YOLO 版本的成功，引入了新功能和改进，进一步提升性能和灵活性。YOLOv8 设计快速、准确且易于使用，使其成为各种物体检测与跟踪、实例分割、图像分类和姿态估计任务的绝佳选择。<br>

<span style="color: lightskyblue;">此项目使用yolov8在KITTI数据集上进行微调，得到了车辆和行人检测模型</span>

<img width="1024" src="https://github.com/NoMoreBeauty/ultralytics/blob/main/runs/detect/predict3/000483.png" alt="Train Result">


<span style="color: lightskyblue;">模型的检测精度如下表所示：</span>
<div align="center">

| Class      | Images | Instances | Box(P | R | mAP50 | mAP50-95) |
|------------|--------|-----------|-------|---|-------|-----------|
| all        | 1497   | 7465      | 0.934 | 0.866 | 0.929 | 0.713 |
| Pedestrian | 1497   | 965       | 0.922 | 0.772 | 0.877 | 0.554 |
| Car        | 1497   | 6500      | 0.945 | 0.96 | 0.982 | 0.873 |

</div>

以`Car`为例说明，精确率`P=0.945`，召回率`R=0.96`，`mAP50=0.982`，`mAP50-95=0.873`。<br>
评估指标显示`Car`的召回率和`mAP50-95`比`Pedestrian`高出较多，这是因为整个`KITTI`数据集中行人的数据量大量少于车的数据量。<br>上述指标的计算公式如[博客](https://blog.csdn.net/qq_63708623/article/details/128508776)所描述。


</div>

## <div align="center">文档</div>

文档前半部分包括环境准备和如何使用此项目的<span style="color: red;">车辆-行人检测模型</span>。<br><br>
后半部分包括如何<span style="color: red;">微调</span>自己的模型。<br><br>
最后一部分的预训练模型描述摘自[YOLOv8](https://github.com/ultralytics/ultralytics)
### <div align="center">车辆-行人检测模型</div>
环境准备（同微调自定义的模型的环境准备）
<details open>
<summary>安装</summary>

使用Pip或Conda在一个[**Python>=3.8**](https://www.python.org/)环境中安装`ultralytics`包。

```bash
pip install ultralytics
```
根据本地实验环境的CUDA版本，安装对应的[**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)，下面给出PyTorch2.0.1-cuda11.8的安装命令。
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```
安装其余必要的[依赖项](https://github.com/NoMoreBeauty/ultralytics/blob/main/requirements.txt)。

```bash
pip install -r requirements.txt
```
</details>

<details open>
<summary>验证</summary>

在命令行中运行命令：
```bash
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```
该命令会下载`yolov8n`的预训练模型和一张图片，并对该图片进行检测，若命令运行成功，则在`./runs/detect/predict`目录下会出现检测结果图。

<div align="center">

<img width="280" src="https://github.com/NoMoreBeauty/ultralytics/blob/main/runs/detect/predict/bus.jpg" 
alt="Confirm install">

</div>

</details>

<details open>
<summary>使用</summary>

#### CLI

在本地命令行界面（CLI）中直接使用，只需输入 `yolo` 命令：

```bash
yolo predict model=./runs/train/weights/best.pt source=./car.png
```
可以通过修改`source`指定待检测的图片，包含预测结果的图片的具体位置（取决于第几次预测）由命令行输出给出。

#### Python

模型也可以在 Python 环境中直接使用，并接受与上述 CLI 示例中相同的[参数](https://docs.ultralytics.com/usage/cfg/)：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("./runs/train/weights/best.pt")  # 加载车辆-行人检测模型

# 使用模型
model.predict('car.png', save=True)
```
可以通过修改source指定待检测的图片，包含预测结果的图片的具体位置（取决于第几次预测）由命令行输出给出。


</details>


### <div align="center">微调自定义模型</div>
环境准备（同上述环境准备），以[KITTI数据集](https://www.cvlibs.net/datasets/kitti/eval_object.php)为例。

<details open>
<summary>数据集</summary>

首先需要将KITTI数据集格式转换为YOLO格式，可以通过github上的开源项目[kitti_for_yolo](https://github.com/oberger4711/kitti_for_yolo)完成该步骤。
<br><br>KITTI数据集内容如下：
<div align="center">

| 标签 | 类别 |
|:-------:|:-------:|
| Pedestrian | 行人 |
| Cyclist | 自行车骑者 |
| Car | 车辆 |
| Van | 货车 |
| Misc | 杂项 |
| Truck | 卡车 |
| Person_sitting | 坐姿行人 |
| Tram | 电车 |
| DontCare | 其他 |

</div>

由于训练资源有限，此项目只聚焦于行人和车辆，因此把`Tram`，`Truck`，`Van`都归到和`Car`一类；`Person_sitting`，`Cyclist`，都归到`Pedestrian`一类进行训练。
<br><br>
数据集的结构应该如下所示：<br>
```plaintext
datasets
│
└── kitti
    ├── images
    │   ├── eval
    │   │   ├── 000003.png
    │   │   └── ...
    │   └── train
    │       ├── 000001.png
    │       └── ...
    └── labels
        ├── eval
        │   ├── 000003.txt
        │   └── ...
        └── train
            ├── 000001.txt
            └── ...
```
此项目的`data_split.py`可以用于从源数据集中拆分训练集和测试集，使用方法:<br>
```bash
python .\data_split.py --train_data_path datasets/kitti/images/train --train_label_path datasets/kitti/labels/train --eval_data_path datasets/kitti/images/eval --eval_label_path datasets/kitti/labels/eval --r 0.8
```
可以修改参数适应具体配置：
<div align="center">

| 参数 | 含义 |
|:-------:|:-------:|
| train_data_path | 训练集数据路径（需要文件夹存在） |
| train_label_path | 训练集标签路径（需要文件夹存在） |
| eval_data_path | 测试集数据路径（需要文件夹存在） |
| eval_label_path | 测试集标签路径（需要文件夹存在） |
| r | 训练集与测试集比例 |

</div>
<br>

此外还需要准备一个关于数据集的配置文件，位置任意，后续训练时使用，此项目的`kitti.yaml`存储在`ultralytics\cfg\datasets`下，内容包括训练数据位置，验证数据位置，测试数据位置（可省略）和类别情况，`kitti.yaml`的内容已经是最简短的，可以参照`kitti.yaml`修改。

</details>


<details open>
<summary>预训练模型</summary>

YOLOv8提供了不同尺寸的预训练模型，具体信息如[模型](o)中描述。可以下载预训练模型到本地，也可以训练时自动下载。<br>



</details>

<details open>
<summary>训练</summary>

在本地命令行界面（CLI）中直接使用，只需输入 `yolo` 命令：
```bash
yolo detect train data=ultralytics/cfg/datasets/kitti.yaml model=yolov8s.yaml pretrained=./yolov8s.pt epochs=300 batch=4 lr0=0.01 resume=True
```
此项目使用了YOLOv8s的预训练模型，可以按照修改上面的参数以更改训练配置。<br>
<div align="center">

| 参数 | 含义 |
|:-------:|:-------:|
| data | 数据集的描述配置文件路径 |
| model | 目标模型的配置文件（只需修改名称选择使用什么模型） |
| pretrained | 预训练模型（可以是本地路径也可以是训练时下载） |
| epochs | 迭代次数 |
| batch | 批次大小（根据GPU的显存大小设置） |
| lr0 | 学习率 |
| resume | 中断后是否可恢复继续训练 |

</div>




</details>


<details open>
<summary>训练结果</summary>

训练完成后会在`train`目录下保存最优模型`best.pt`，以及训练过程中的各种数据图。
```plaintext
runs
│
└── train
    ├── weights
    │   ├── best.pt
    │   └── last.pt
    ├── results.csv
    ├── results.png
    └── ...
```
训练损失，测试损失等数据在`results.png`中，如下图所示：
<img width="1024" src="https://github.com/NoMoreBeauty/ultralytics/blob/main/runs/train/results.png" alt="Train Result">
</details>


## <div align="center">模型（取自YOLOv8）</div>

这里只给出在[COCO](https://docs.ultralytics.com/datasets/detect/coco)数据集上预训练的YOLOv8 [检测](https://docs.ultralytics.com/tasks/detect)，[分割](https://docs.ultralytics.com/tasks/segment)和[姿态](https://docs.ultralytics.com/tasks/pose)模型。其余预训练模型可以在[YOLOv8主页](https://github.com/ultralytics/ultralytics/tree/main)找到<br>
<br>
<img width="1024" src="https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png" alt="Ultralytics YOLO supported tasks">

所有[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)在首次使用时会自动从最新的Ultralytics [发布版本](https://github.com/ultralytics/assets/releases)下载。

<details open><summary>检测 (COCO)</summary>

查看[检测文档](https://docs.ultralytics.com/tasks/detect/)以获取这些在[COCO](https://docs.ultralytics.com/datasets/detect/coco/)上训练的模型的使用示例，其中包括80个预训练类别。

| 模型                                                                                   | 尺寸<br><sup>(像素) | mAP<sup>val<br>50-95 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------- | -------------------- | --------------------------- | -------------------------------- | -------------- | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640             | 37.3                 | 80.4                        | 0.99                             | 3.2            | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640             | 44.9                 | 128.4                       | 1.20                             | 11.2           | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640             | 50.2                 | 234.7                       | 1.83                             | 25.9           | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640             | 52.9                 | 375.2                       | 2.39                             | 43.7           | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640             | 53.9                 | 479.1                       | 3.53                             | 68.2           | 257.8             |

- **mAP<sup>val</sup>** 值是基于单模型单尺度在 [COCO val2017](http://cocodataset.org) 数据集上的结果。
  <br>通过 `yolo val detect data=coco.yaml device=0` 复现
- **速度** 是使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例对 COCO val 图像进行平均计算的。
  <br>通过 `yolo val detect data=coco.yaml batch=1 device=0|cpu` 复现

</details>