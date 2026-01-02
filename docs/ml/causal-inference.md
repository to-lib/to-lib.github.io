---
sidebar_position: 30
title: ⚖️ 因果推断
---

# 因果推断

因果推断研究变量之间的因果关系，而不仅是相关性。

## 相关 vs 因果

> "相关不等于因果"

```mermaid
graph LR
    A[冰淇淋销量] --- B[溺水事故]
    C[夏天] --> A
    C --> B
```

## 核心概念

| 概念           | 描述             |
| -------------- | ---------------- |
| Treatment (T)  | 干预/处理变量    |
| Outcome (Y)    | 结果变量         |
| Confounder (X) | 混淆变量         |
| ATE            | 平均因果效应     |
| CATE           | 条件平均因果效应 |

$$
ATE = E[Y(1) - Y(0)]
$$

## Rubin 因果模型

```python
import numpy as np

# 潜在结果框架
# Y(0): 未处理的结果
# Y(1): 处理后的结果
# 个体因果效应 ITE = Y(1) - Y(0)
# 但我们只能观察到一个

# 估计 ATE
def estimate_ate_naive(y_treated, y_control):
    return np.mean(y_treated) - np.mean(y_control)
```

## 倾向得分匹配 (PSM)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

def propensity_score_matching(X, T, Y):
    # 1. 估计倾向得分
    ps_model = LogisticRegression()
    ps_model.fit(X, T)
    propensity_scores = ps_model.predict_proba(X)[:, 1]

    # 2. 匹配
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(propensity_scores[control_idx].reshape(-1, 1))

    matched_control = []
    for idx in treated_idx:
        _, match_idx = nn.kneighbors([[propensity_scores[idx]]])
        matched_control.append(control_idx[match_idx[0][0]])

    # 3. 估计 ATE
    ate = np.mean(Y[treated_idx] - Y[matched_control])
    return ate
```

## 双重机器学习 (DML)

```python
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor

# 使用 EconML
dml = LinearDML(
    model_y=RandomForestRegressor(),
    model_t=RandomForestRegressor()
)

dml.fit(Y, T, X=X)
ate = dml.ate(X)
cate = dml.effect(X)  # 条件效应
```

## 因果森林

```python
from econml.grf import CausalForest

cf = CausalForest(n_estimators=100)
cf.fit(X, T, Y)

# 个体因果效应
ite = cf.predict(X)
```

## 工具变量

```python
from linearmodels.iv import IV2SLS

# Y = β₀ + β₁T + ε  (T 是内生的)
# 使用工具变量 Z

model = IV2SLS.from_formula('Y ~ 1 + [T ~ Z]', data=df)
result = model.fit()
print(result.summary)
```

## 方法选择

| 方法             | 适用场景   | 假设         |
| ---------------- | ---------- | ------------ |
| 随机实验 (RCT)   | 可控实验   | 随机分配     |
| 倾向得分匹配     | 观察性数据 | 无未观测混淆 |
| 双重差分 (DID)   | 政策评估   | 平行趋势     |
| 工具变量         | 存在内生性 | 工具有效     |
| 回归不连续 (RDD) | 阈值效应   | 连续性       |

## 应用场景

| 领域 | 应用         |
| ---- | ------------ |
| 营销 | 广告效果评估 |
| 医疗 | 治疗效果分析 |
| 政策 | 政策干预效果 |
| 推荐 | 因果推荐     |
