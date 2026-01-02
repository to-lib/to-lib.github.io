---
sidebar_position: 9
title: 📏 模型评估
---

# 模型评估

## 数据集划分

```python
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# 划分训练/验证/测试集
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)

# 交叉验证
scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring='accuracy')
print(f"CV 准确率: {scores.mean():.2%} ± {scores.std():.2%}")
```

## 分类指标

### 混淆矩阵与核心指标

|        | 预测正 | 预测负 |
| ------ | ------ | ------ |
| 实际正 | TP     | FN     |
| 实际负 | FP     | TN     |

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print(classification_report(y_test, y_pred))

# AUC
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
```

| 指标   | 公式           | 适用场景     |
| ------ | -------------- | ------------ |
| 准确率 | (TP+TN)/总数   | 类别平衡     |
| 精确率 | TP/(TP+FP)     | 假正例代价高 |
| 召回率 | TP/(TP+FN)     | 假负例代价高 |
| F1     | 2×P×R/(P+R)    | 类别不平衡   |
| AUC    | ROC 曲线下面积 | 排序能力     |

## 回归指标

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

| 指标     | 特点         |
| -------- | ------------ |
| MSE/RMSE | 对大误差敏感 |
| MAE      | 对异常值鲁棒 |
| R²       | 解释方差比例 |

## 学习曲线

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(model, X, y, cv=5)
# 高偏差：两条曲线都低 → 增加复杂度
# 高方差：训练高验证低 → 增加数据/正则化
```
