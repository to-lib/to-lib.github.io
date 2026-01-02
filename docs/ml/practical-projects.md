---
sidebar_position: 11
title: 💡 实战项目
---

# 机器学习实战项目

## 项目一：手写数字识别 (MNIST)

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 加载数据
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# 训练模型
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 评估
print(classification_report(y_test, y_pred))
```

### PyTorch CNN 版本

```python
import torch.nn as nn

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))
```

## 项目二：房价预测

```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# 加载数据
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target)

# 构建 Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(n_estimators=100, learning_rate=0.1))
])

pipeline.fit(X_train, y_train)
print(f"R² Score: {pipeline.score(X_test, y_test):.4f}")
```

## 项目三：文本情感分析

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 示例数据
texts = ["这个产品很好", "服务太差了", "非常满意", "不推荐购买"]
labels = [1, 0, 1, 0]  # 1: 正面, 0: 负面

# Pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

text_clf.fit(texts, labels)
print(text_clf.predict(["质量不错"]))  # [1]
```

## 项目清单

| 项目           | 类型     | 技术栈       | 难度     |
| -------------- | -------- | ------------ | -------- |
| MNIST 手写数字 | 图像分类 | CNN          | ⭐⭐     |
| 房价预测       | 回归     | XGBoost      | ⭐⭐     |
| 情感分析       | NLP      | TF-IDF, LSTM | ⭐⭐⭐   |
| 客户流失预测   | 分类     | 随机森林     | ⭐⭐     |
| 推荐系统       | 协同过滤 | SVD          | ⭐⭐⭐   |
| 图像风格迁移   | 生成     | CNN          | ⭐⭐⭐⭐ |
