# 手写数字识别系统

这个项目使用MNIST数据集训练YOLOv8模型实现手写数字识别，并通过摄像头进行实时检测。

## 项目结构

```
.
├── data/                      # 数据集目录
│   └── with_notrans/          # 原始MNIST数据
│   └── yolo_mnist/            # 转换后的YOLO格式数据(自动生成)
├── models/                    # 模型保存目录(自动生成) mnist_digits10.pt 为训练得到的模型
├── convert_mnist_to_yolo.py   # 数据集转换脚本
├── train.py                   # 模型训练与摄像头识别脚本
└── README.md                  # 项目说明
```

## 使用方法

1. **转换数据集**

首先需要将MNIST数据集转换为YOLO格式：

```bash
python convert_mnist_to_yolo.py
```

2. **训练模型并启动摄像头识别**

运行以下命令开始训练并在训练完成后自动启动摄像头识别：

```bash
python train.py
```

## 环境依赖

本项目需要安装以下Python包：

- ultralytics (YOLOv8)
- torch
- opencv-python
- numpy
- pillow

安装依赖：

```bash
pip install ultralytics opencv-python pillow numpy torch
```

## 使用说明

1. 运行`convert_mnist_to_yolo.py`转换数据集，转换后的数据会保存在`data/yolo_mnist`目录下
2. 运行`train.py`训练模型，训练完成的模型会保存在`models/mnist_digits.pt`
3. 训练完成后，系统会自动启动摄像头进行实时识别
4. 在摄像头窗口中按"q"键退出识别程序

## 注意事项

1. 确保您的电脑配置有摄像头并且能够正常工作
2. 运行前确保已安装所有必要的依赖项
3. 对于手写数字识别，摄像头中的数字应该尽量清晰且背景简单 