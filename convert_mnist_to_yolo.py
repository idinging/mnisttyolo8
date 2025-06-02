import os
import numpy as np
from torchvision.datasets import MNIST
from PIL import Image
import shutil

# 下载数据集
train_dataset_no = MNIST('./data/with_notrans',train=True,download=True)
test_dataset_no = MNIST('./data/with_notrans',train=False,download=True)

def convert_mnist_to_yolo(dataset_path='./data/with_notrans', output_path='./data/yolo_mnist'):
    """
    将MNIST数据集转换为YOLO格式
    
    参数:
        dataset_path: MNIST数据集的路径
        output_path: YOLO格式数据的输出路径
    """
    # 创建输出目录结构
    train_images_dir = os.path.join(output_path, 'images', 'train')
    train_labels_dir = os.path.join(output_path, 'labels', 'train')
    val_images_dir = os.path.join(output_path, 'images', 'val')
    val_labels_dir = os.path.join(output_path, 'labels', 'val')
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # 加载MNIST数据集
    print("加载训练集...")
    train_dataset = MNIST(dataset_path, train=True, download=False)
    print("加载测试集...")
    test_dataset = MNIST(dataset_path, train=False, download=False)
    
    # 处理训练集
    print("处理训练集...")
    _process_dataset(train_dataset, train_images_dir, train_labels_dir)
    
    # 处理测试集（作为验证集）
    print("处理验证集...")
    _process_dataset(test_dataset, val_images_dir, val_labels_dir)
    
    # 创建数据配置文件
    _create_data_yaml(output_path)
    
    print(f"转换完成！数据保存在: {output_path}")

def _process_dataset(dataset, images_dir, labels_dir):
    """处理数据集并转换为YOLO格式"""
    for idx in range(len(dataset)):
        if idx % 1000 == 0:
            print(f"处理图像: {idx}/{len(dataset)}")
            
        img, label = dataset[idx]
        
        # 保存图像
        img_path = os.path.join(images_dir, f"{idx:05d}.png")
        img.save(img_path)
        
        # 创建YOLO格式标签文件
        # 对于MNIST，将数字视为占据图像中心的对象，大小为图像的80%
        label_path = os.path.join(labels_dir, f"{idx:05d}.txt")
        with open(label_path, 'w') as f:
            # YOLO格式: <class_id> <center_x> <center_y> <width> <height>
            # 所有值都是归一化的（0-1）
            # 中心点在图像中心，宽高设为0.8
            f.write(f"{label} 0.5 0.5 0.8 0.8\n")

def _create_data_yaml(output_path):
    """创建data.yaml配置文件"""
    yaml_content = """
# MNIST数据集配置
train: images/train
val: images/val

# 类别数量和名称
nc: 10  # 10个数字类别（0-9）
names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
"""
    
    with open(os.path.join(output_path, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    convert_mnist_to_yolo() 