---
sidebar_position: 37
title: 📐 不确定性量化
---

# 不确定性量化

不确定性量化估计模型预测的置信度，对于安全关键应用至关重要。

## 不确定性类型

| 类型                     | 描述         | 来源               |
| ------------------------ | ------------ | ------------------ |
| 认知不确定性 (Epistemic) | 模型不确定性 | 数据不足、模型局限 |
| 偶然不确定性 (Aleatoric) | 数据固有噪声 | 无法消除           |

## 方法概览

### MC Dropout

```python
import torch
import torch.nn as nn

class MCDropoutModel(nn.Module):
    def __init__(self, model, dropout_rate=0.1):
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, n_samples=100):
        self.train()  # 保持 dropout 开启
        predictions = []

        for _ in range(n_samples):
            pred = self.model(self.dropout(x))
            predictions.append(pred)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)  # 不确定性

        return mean, std
```

### Deep Ensemble

```python
class DeepEnsemble:
    def __init__(self, model_class, n_models=5):
        self.models = [model_class() for _ in range(n_models)]

    def fit(self, X, y):
        for model in self.models:
            # 每个模型用不同随机种子训练
            model.fit(X, y)

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        predictions = np.stack(predictions)

        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)

        return mean, std
```

### 贝叶斯神经网络

```python
import torch.nn as nn
import torch.nn.functional as F

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) - 3)
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.ones(out_features) * -3)

    def forward(self, x):
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)

        return F.linear(x, weight, bias)
```

### 温度缩放 (Calibration)

```python
class TemperatureScaling:
    def __init__(self, model):
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def calibrate(self, val_loader):
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01)

        def eval():
            loss = 0
            for x, y in val_loader:
                logits = self.model(x) / self.temperature
                loss += nn.functional.cross_entropy(logits, y)
            return loss

        optimizer.step(eval)

    def predict(self, x):
        logits = self.model(x) / self.temperature
        return torch.softmax(logits, dim=1)
```

## 评估指标

```python
from sklearn.calibration import calibration_curve

# 可靠性图
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

# ECE (Expected Calibration Error)
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += mask.sum() * abs(acc - conf)
    return ece / len(y_true)
```

## 应用场景

| 领域     | 应用                   |
| -------- | ---------------------- |
| 医疗诊断 | 识别需要人工复核的病例 |
| 自动驾驶 | 低置信度时交还控制权   |
| 主动学习 | 选择最不确定的样本标注 |
| 异常检测 | 高不确定性可能是异常   |
