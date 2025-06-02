import os
import torch
import yaml
from PIL import Image
import cv2
import numpy as np
import time
import glob # 用于查找文件

# 检查依赖
try:
    import ultralytics
except ImportError:
    print("正在安装YOLOv8依赖...")
    os.system("pip install ultralytics")
    import ultralytics

from ultralytics import YOLO

def get_latest_run_dir(base_dir="runs/detect", prefix="mnist_digits"):
    """获取最新的YOLOv8运行目录"""
    list_of_dirs = glob.glob(os.path.join(base_dir, f"{prefix}*"))
    if not list_of_dirs:
        return None
    latest_dir = max(list_of_dirs, key=os.path.getctime)
    return latest_dir

def train_model(data_yaml_path="./data/yolo_mnist/data.yaml", 
              epochs=50,                 # 训练轮数，增加可提高精度
              imgsz=32,                  # 输入图像尺寸，MNIST推荐32
              batch=64,                  # 批次大小，较小批次可能提高泛化能力
              device=None,               # 训练设备
              model_size="n",            # 模型大小：n(nano), s(small), m(medium), l(large), x(xlarge)
              lr0=0.01,                  # 初始学习率
              lrf=0.01,                  # 最终学习率（作为初始学习率的比例）
              warmup_epochs=3.0,         # 预热轮数，逐渐增加学习率
              optimizer="auto",          # 优化器：SGD, Adam, AdamW, auto
              weight_decay=0.0005,       # 权重衰减，防止过拟合
              momentum=0.937,            # SGD动量参数
              # 数据增强参数
              fliplr=0.5,                # 水平翻转概率
              flipud=0.0,                # 垂直翻转概率
              mosaic=1.0,                # 马赛克增强强度
              mixup=0.1,                 # 混合增强概率
              copy_paste=0.1,            # 复制粘贴概率
              degrees=10.0,              # 旋转角度范围
              translate=0.2,             # 平移范围
              scale=0.5,                 # 缩放范围
              shear=0.0,                 # 剪切变换范围
              perspective=0.0,           # 透视变换范围
              hsv_h=0.015,               # HSV-色调增强
              hsv_s=0.7,                 # HSV-饱和度增强
              hsv_v=0.4,                 # HSV-亮度增强
              patience=50                # 早停耐心值，当验证集性能多少轮不提升就停止
              ):
    """
    训练手写数字识别模型
    
    参数:
        data_yaml_path: YOLO格式数据配置文件路径
        epochs: 训练轮数，增加轮数通常可以提高模型性能
        imgsz: 输入图像尺寸，必须是32的倍数
        batch: 批次大小，较小的批次大小有时能提高泛化能力
        device: 训练设备，可以是'cpu', '0', '0,1,2,3'等
        model_size: 模型大小，'n'(纳米)是最小最快的，'x'(超大)精度最高但最慢
        lr0: 初始学习率，影响训练速度和稳定性
        lrf: 最终学习率，作为lr0的比例
        warmup_epochs: 预热轮数，在这些轮中学习率从较小值逐渐增加到lr0
        optimizer: 优化器类型
        weight_decay: 权重衰减参数，用于正则化
        momentum: SGD优化器的动量参数
        
        数据增强参数:
        fliplr, flipud: 水平和垂直翻转的概率
        mosaic: 马赛克增强强度，将4张图像拼接
        mixup: 混合增强概率，混合两张图像
        copy_paste: 复制粘贴概率
        degrees: 旋转角度范围
        translate: 平移范围
        scale: 缩放范围
        shear: 剪切变换范围
        perspective: 透视变换范围
        hsv_h, hsv_s, hsv_v: HSV色彩空间增强参数
        
        patience: 早停耐心值，如果验证集性能连续多少轮没有提升就停止训练
    """
    print("开始训练模型...")
    
    # 检测GPU可用性
    if device is None:
        device = '0' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {device}")
    
    # 确保模型文件夹存在
    os.makedirs("models", exist_ok=True)
    
    # 选择模型大小
    model_types = {
        'n': 'yolov8n.yaml',      # 纳米 (最小最快)
        's': 'yolov8s.yaml',      # 小型
        'm': 'yolov8m.yaml',      # 中型
        'l': 'yolov8l.yaml',      # 大型
        'x': 'yolov8x.yaml'       # 超大型 (最高精度但最慢)
    }
    
    model_type = model_types.get(model_size.lower(), 'yolov8n.yaml')
    print(f"使用模型类型: {model_type}")
    
    # 创建YOLOv8模型
    model = YOLO(model_type)
    
    # 开始训练，YOLO会自动处理运行目录名
    run_name = "mnist_digits" 
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=run_name,
        device=device,
        exist_ok=False,
        # 学习率参数
        lr0=lr0,
        lrf=lrf,
        # 优化器参数
        optimizer=optimizer,
        weight_decay=weight_decay,
        momentum=momentum,
        warmup_epochs=warmup_epochs,
        # 数据增强参数
        fliplr=fliplr,
        flipud=flipud,
        mosaic=mosaic, 
        mixup=mixup,
        copy_paste=copy_paste,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        # 早停参数
        patience=patience,
        # 验证设置
        val=True,
        plots=True,          # 生成训练过程的图表
    )
    
    # 获取最新的训练目录
    latest_run_directory = get_latest_run_dir(prefix=run_name)
    if not latest_run_directory:
        print("错误：无法找到训练运行目录")
        return None
        
    print(f"最新的训练目录: {latest_run_directory}")
    trained_model_path = os.path.join(latest_run_directory, "weights", "best.pt")
    
    # 复制最佳模型到models文件夹
    if os.path.exists(trained_model_path):
        # 使用固定名称"mnist_digits.pt"，不带数字后缀
        final_model_path = os.path.join("models", "mnist_digits.pt")
        try:
            best_model = YOLO(trained_model_path)
            best_model.save(final_model_path)
            print(f"模型已成功复制并保存至: {final_model_path}")
            return final_model_path
        except Exception as e:
            print(f"复制和保存模型到 'models' 目录失败: {e}")
            print(f"将直接使用原始训练模型路径: {trained_model_path}")
            return trained_model_path 
    else:
        print(f"训练失败或未找到保存的模型于: {trained_model_path}")
        return Non

def preprocess_frame(frame):
    """
    预处理摄像头帧图像
    
    参数:
        frame: 摄像头捕获的帧
    返回:
        预处理后的图像和包含数字区域的原始帧
    """
    # 保留原始帧以显示检测结果
    original_frame = frame.copy()
    
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 应用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 自适应阈值处理提取手写区域
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # 寻找轮廓 - 可能是数字的区域
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 用于绘制找到的轮廓
    contour_image = np.zeros_like(thresh)
    
    # 过滤掉太小的轮廓，这些可能是噪声
    min_contour_area = 100  # 最小轮廓面积（可根据实际情况调整）
    digit_regions = []
    
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        
        # 忽略小轮廓
        if area < min_contour_area:
            continue
        
        # 获取轮廓的边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 扩大边界矩形以确保数字完整捕获（可选）
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(thresh.shape[1] - x, w + 2 * padding)
        h = min(thresh.shape[0] - y, h + 2 * padding)
        
        # 提取这个区域
        digit_region = thresh[y:y+h, x:x+w]
        
        # 将轮廓添加到列表中供后续处理
        digit_regions.append((x, y, w, h, digit_region))
        
        # 在轮廓图像上绘制这个轮廓
        cv2.drawContours(contour_image, [contour], -1, 255, -1)
        
        # 在原始帧上标记这个区域
        cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # 将轮廓图像与阈值图像组合
    # 这样可以同时保留原始阈值处理的效果和提取的轮廓
    combined = cv2.bitwise_or(thresh, contour_image)
    
    # 返回预处理后的图像和标记了数字区域的原始帧
    return combined, original_frame, digit_regions

def camera_recognition(model_path="models/mnist_digits10.pt"):
    """
    使用摄像头实时识别手写数字
    
    参数:
        model_path: 训练好的模型路径
    """
    # 加载模型
    if not os.path.exists(model_path):
        print(f"错误：未找到模型文件 {model_path}")
        return
    
    model = YOLO(model_path)
    print("模型加载成功，启动摄像头...")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("摄像头已打开，开始实时识别...")
    print("按 'q' 键退出")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取摄像头画面")
                break
            
            # 预处理图像 - 现在返回多个值
            processed_single_channel, marked_frame, digit_regions = preprocess_frame(frame)
            
            # 创建显示图像的副本使用已标记的帧
            display_frame = marked_frame.copy()
            
            # 将整个预处理后的单通道图像转换为三通道用于模型输入
            if processed_single_channel.ndim == 2 or processed_single_channel.shape[2] == 1:
                processed_for_model = cv2.cvtColor(processed_single_channel, cv2.COLOR_GRAY2BGR)
            else:
                processed_for_model = processed_single_channel 

            # 对整个图像进行检测
            results = model(processed_for_model, verbose=False, conf=0.6)
            
            # 也对每个识别到的数字区域单独进行检测
            for x, y, w, h, digit_region in digit_regions:
                # 将数字区域缩放到固定大小（32x32）
                resized_digit = cv2.resize(digit_region, (32, 32), interpolation=cv2.INTER_AREA)
                
                # 转换为3通道
                resized_digit_3ch = cv2.cvtColor(resized_digit, cv2.COLOR_GRAY2BGR)
                
                # 保存调试图像
                # debug_filename = f"debug_digit_{time.time()}.png"
                # cv2.imwrite(debug_filename, resized_digit_3ch)
                
                # 单独对这个区域进行预测
                region_results = model(resized_digit_3ch, verbose=False, conf=0.3)
                
                # 在原图上显示识别结果
                for region_result in region_results:
                    boxes = region_result.boxes
                    for box in boxes:
                        # 获取预测类别和置信度
                        cls = int(box.cls[0].item())
                        conf = box.conf[0].item()
                        
                        # 在原始位置绘制识别结果
                        cv2.putText(display_frame, f"{cls}: {conf:.2f}", (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            detected_digits_in_frame = [] # 用于存储当前帧检测到的数字

            # 处理全局检测结果
            for result in results:
                boxes = result.boxes
                if boxes: # 只有当检测到边界框时才处理
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        # 获取预测类别和置信度
                        cls = int(box.cls[0].item())
                        conf = box.conf[0].item()
                        
                        detected_digits_in_frame.append(str(cls)) # 将检测到的数字（类别）添加到列表
                        
                        # 在图像上绘制检测结果
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"{cls}: {conf:.2f}", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            # 显示原始图像和处理后的图像
            cv2.imshow("Camera Feed", display_frame)
            # cv2.imshow("Processed", processed_single_channel) # 显示原始的单通道处理结果
            
            # 按'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("摄像头已关闭")

def main():
    """主函数"""
    # 检查数据集是否已转换为YOLO格式
    data_yaml_path = "./data/yolo_mnist/data.yaml"
    if not os.path.exists(data_yaml_path):
        print("YOLO格式数据集不存在，请先运行convert_mnist_to_yolo.py脚本")
        return

    # 检测是否有可用的GPU
    if torch.cuda.is_available():
        print(f"检测到可用的GPU: {torch.cuda.get_device_name(0)}")
        device = '0'  # 使用第一个GPU
    else:
        print("未检测到GPU，将使用CPU进行训练")
        device = 'cpu'
    
    # 训练模型，调整参数以提高精度
    # 高精度训练（耗时长但精度更高）
    # model_path = train_model(
    #     data_yaml_path=data_yaml_path,
    #     device=device,
    #     epochs=10,          # 大量训练轮数
    #     batch=64,            # 较小的批次大小可以提高泛化性能
    #     model_size="m",      # 中型模型，精度更好
    #     lr0=0.005,           # 较小的学习率，更稳定
    #     patience=100,        # 较大的耐心值，避免过早停止
    #     # 加强数据增强
    #     mixup=0.2,           # 中等混合增强
    #     copy_paste=0.2,      # 中等复制粘贴增强
    #     degrees=15.0,        # 较大旋转范围
    #     translate=0.25,      # 较大平移范围
    #     scale=0.6,           # 较大缩放范围
    #     mosaic=1.0,          # 最大马赛克增强
    #     hsv_h=0.02,          # 更强的色调变化
    #     hsv_s=0.8,           # 更强的饱和度变化
    #     hsv_v=0.5,           # 更强的亮度变化
    #     optimizer="AdamW",   # 使用AdamW优化器代替自动选择
    # )
    
    # 使用现有模型（如果不想重新训练）
    model_path = 'models/mnist_digits10.pt'
    
    if model_path and os.path.exists(model_path):
        # 延迟2秒，让用户看到训练已完成的消息
        time.sleep(2)
        
        # 启动摄像头识别
        camera_recognition(model_path)
    else:
        print("错误：模型训练失败，无法启动摄像头识别")

if __name__ == "__main__":
    main()
