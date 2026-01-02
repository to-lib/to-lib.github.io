---
sidebar_position: 20
title: 📊 贝叶斯方法
---

# 贝叶斯方法

贝叶斯方法基于概率论，将不确定性融入模型中。

## 贝叶斯定理

$$
P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}
$$

| 术语     | 符号    | 描述                   |
| -------- | ------- | ---------------------- |
| 后验概率 | P(θ\|D) | 观测数据后对参数的信念 |
| 似然     | P(D\|θ) | 给定参数下数据的概率   |
| 先验概率 | P(θ)    | 观测前对参数的信念     |
| 边际似然 | P(D)    | 归一化因子             |

## 朴素贝叶斯

假设特征之间条件独立。

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# 高斯朴素贝叶斯（连续特征）
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 多项式朴素贝叶斯（计数特征，如词频）
mnb = MultinomialNB()

# 伯努利朴素贝叶斯（二元特征）
bnb = BernoulliNB()
```

### 文本分类示例

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB())
])

text_clf.fit(texts, labels)
predictions = text_clf.predict(new_texts)
```

## 高斯过程

非参数贝叶斯方法，用于回归和分类。

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
gpr.fit(X_train, y_train)

# 预测均值和标准差
mean, std = gpr.predict(X_test, return_std=True)
```

## 贝叶斯优化

用于超参数调优，适合昂贵的目标函数。

```python
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier

opt = BayesSearchCV(
    RandomForestClassifier(),
    {
        'n_estimators': (50, 500),
        'max_depth': (3, 20),
        'min_samples_split': (2, 20)
    },
    n_iter=50,
    cv=5
)
opt.fit(X_train, y_train)
print(f"最佳参数: {opt.best_params_}")
```

### Optuna

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    }
    model = XGBClassifier(**params)
    return cross_val_score(model, X, y, cv=5).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

## 贝叶斯神经网络

```python
import torch
import torch.nn as nn

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.randn(out_features))
        self.bias_sigma = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        weight = self.weight_mu + torch.exp(self.weight_sigma) * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + torch.exp(self.bias_sigma) * torch.randn_like(self.bias_mu)
        return torch.nn.functional.linear(x, weight, bias)
```

## 方法对比

| 方法       | 适用场景   | 优点         | 缺点         |
| ---------- | ---------- | ------------ | ------------ |
| 朴素贝叶斯 | 文本分类   | 快速、简单   | 特征独立假设 |
| 高斯过程   | 小数据回归 | 不确定性估计 | 计算复杂度高 |
| 贝叶斯优化 | 超参数调优 | 样本效率高   | 维度灾难     |
