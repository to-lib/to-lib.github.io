---
sidebar_position: 33
title: 📊 在线学习
---

# 在线学习与增量学习

在线学习从流式数据中持续学习，无需重新训练。

## 核心概念

| 术语     | 描述                   |
| -------- | ---------------------- |
| 在线学习 | 逐样本更新模型         |
| 增量学习 | 批量更新，不访问旧数据 |
| 流式学习 | 处理无限数据流         |

## 在线学习算法

### 随机梯度下降

```python
class OnlineSGD:
    def __init__(self, n_features, lr=0.01):
        self.weights = np.zeros(n_features)
        self.lr = lr

    def partial_fit(self, x, y):
        pred = np.dot(x, self.weights)
        error = y - pred
        self.weights += self.lr * error * x

    def predict(self, x):
        return np.dot(x, self.weights)
```

### sklearn 增量学习

```python
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB

# 支持 partial_fit 的模型
model = SGDClassifier()

# 增量训练
for X_batch, y_batch in data_stream:
    model.partial_fit(X_batch, y_batch, classes=classes)
```

### River (流式学习库)

```python
from river import linear_model, metrics, preprocessing

model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
metric = metrics.Accuracy()

for x, y in stream:
    y_pred = model.predict_one(x)
    metric.update(y, y_pred)
    model.learn_one(x, y)
```

## 概念漂移检测

```python
from river import drift

# ADWIN 检测器
adwin = drift.ADWIN()

for i, (x, y) in enumerate(stream):
    adwin.update(error)  # 更新错误率
    if adwin.drift_detected:
        print(f"检测到漂移: {i}")
        # 重置或调整模型
```

## 增量神经网络

```python
import torch

class IncrementalModel:
    def __init__(self, model, replay_buffer_size=1000):
        self.model = model
        self.buffer = []
        self.buffer_size = replay_buffer_size

    def learn(self, new_data, epochs=1):
        # 合并新数据和回放缓冲区
        combined = new_data + self.buffer[-self.buffer_size:]

        # 训练
        for _ in range(epochs):
            for x, y in combined:
                self.model.train_step(x, y)

        # 更新缓冲区
        self.buffer.extend(new_data)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
```

## 应用场景

| 场景       | 特点             |
| ---------- | ---------------- |
| 推荐系统   | 实时更新用户偏好 |
| 欺诈检测   | 适应新的欺诈模式 |
| 传感器数据 | 处理连续数据流   |
| 广告点击   | 实时优化投放     |
