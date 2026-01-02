---
sidebar_position: 36
title: 🛡️ 对抗鲁棒性
---

# 对抗攻击与鲁棒性

对抗样本是精心设计的输入扰动，可导致模型产生错误预测。

## 对抗样本

```python
import torch

# 原始图像 + 微小扰动 = 错误预测
# x_adv = x + perturbation
# ||perturbation|| < epsilon
```

## 攻击方法

### FGSM (快速梯度符号法)

```python
def fgsm_attack(model, x, y, epsilon):
    x.requires_grad = True
    output = model(x)
    loss = nn.functional.cross_entropy(output, y)
    loss.backward()

    # 沿梯度方向扰动
    perturbation = epsilon * x.grad.sign()
    x_adv = x + perturbation
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv
```

### PGD (投影梯度下降)

```python
def pgd_attack(model, x, y, epsilon, alpha, num_iter):
    x_adv = x.clone()

    for _ in range(num_iter):
        x_adv.requires_grad = True
        output = model(x_adv)
        loss = nn.functional.cross_entropy(output, y)
        loss.backward()

        # 梯度上升
        x_adv = x_adv + alpha * x_adv.grad.sign()

        # 投影到 epsilon 球内
        perturbation = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + perturbation, 0, 1).detach()

    return x_adv
```

### AutoAttack

```python
from autoattack import AutoAttack

adversary = AutoAttack(model, norm='Linf', eps=8/255)
x_adv = adversary.run_standard_evaluation(x, y)
```

## 防御方法

### 对抗训练

```python
def adversarial_training(model, dataloader, epsilon, epochs):
    for epoch in range(epochs):
        for x, y in dataloader:
            # 生成对抗样本
            x_adv = pgd_attack(model, x, y, epsilon, alpha=2/255, num_iter=10)

            # 在对抗样本上训练
            optimizer.zero_grad()
            output = model(x_adv)
            loss = nn.functional.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
```

### 输入预处理

```python
def input_transformation(x):
    # JPEG 压缩
    # 随机缩放
    # 添加噪声
    return transformed_x
```

### 对抗检测

```python
class AdversarialDetector:
    def __init__(self, model):
        self.model = model

    def detect(self, x):
        # 基于预测不一致性检测
        output1 = self.model(x)
        output2 = self.model(x + torch.randn_like(x) * 0.01)

        inconsistency = (output1.argmax(1) != output2.argmax(1))
        return inconsistency
```

## 评估指标

| 指标                | 描述           |
| ------------------- | -------------- |
| Clean Accuracy      | 干净样本准确率 |
| Robust Accuracy     | 对抗样本准确率 |
| Attack Success Rate | 攻击成功率     |

## 应用场景

| 领域     | 风险           |
| -------- | -------------- |
| 自动驾驶 | 交通标志误识别 |
| 人脸识别 | 身份伪造       |
| 垃圾邮件 | 绕过检测       |
| 恶意软件 | 逃避检测       |
