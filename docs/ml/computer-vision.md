---
sidebar_position: 22
title: 🖼️ 计算机视觉
---

# 计算机视觉基础

计算机视觉让机器理解和处理图像与视频。

## 图像处理基础

```python
import cv2
import numpy as np

# 读取图像
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 调整大小
resized = cv2.resize(img, (224, 224))

# 滤波
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# 边缘检测
edges = cv2.Canny(gray, 100, 200)
```

## 图像分类

### 使用 PyTorch

```python
import torch
import torchvision.transforms as transforms
from torchvision import models

# 预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载预训练模型
model = models.resnet50(pretrained=True)
model.eval()

# 预测
with torch.no_grad():
    output = model(image_tensor.unsqueeze(0))
    pred = torch.argmax(output, dim=1)
```

### 迁移学习

```python
# 冻结特征提取层
for param in model.parameters():
    param.requires_grad = False

# 替换分类头
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 只训练新的分类头
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

## 目标检测

### YOLO

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')

# 检测
results = model('image.jpg')

# 获取结果
for r in results:
    boxes = r.boxes  # 边界框
    for box in boxes:
        cls = box.cls      # 类别
        conf = box.conf    # 置信度
        xyxy = box.xyxy    # 坐标
```

### 常见检测模型

| 模型         | 特点     | 速度 |
| ------------ | -------- | ---- |
| YOLOv8       | 实时检测 | 快   |
| Faster R-CNN | 高精度   | 中   |
| SSD          | 平衡     | 中   |

## 图像分割

### 语义分割

```python
from transformers import SegformerForSemanticSegmentation

model = SegformerForSemanticSegmentation.from_pretrained(
    'nvidia/segformer-b0-finetuned-ade-512-512'
)
```

### 实例分割

```python
# Mask R-CNN
model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
```

## 数据增强

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## 常见任务

| 任务     | 描述             | 输出          |
| -------- | ---------------- | ------------- |
| 图像分类 | 识别图像类别     | 类别标签      |
| 目标检测 | 定位并分类物体   | 边界框 + 类别 |
| 语义分割 | 像素级分类       | 分割掩码      |
| 实例分割 | 区分同类不同实例 | 实例掩码      |
| 姿态估计 | 检测人体关键点   | 关键点坐标    |
