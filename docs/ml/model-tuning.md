---
sidebar_position: 10
title: ⚙️ 模型调优
---

# 模型调优

## 超参数搜索

### 网格搜索

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(), param_grid,
    cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳分数: {grid_search.best_score_:.2%}")
```

### 随机搜索

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 20),
    'learning_rate': uniform(0.01, 0.3)
}

random_search = RandomizedSearchCV(
    XGBClassifier(), param_dist,
    n_iter=50, cv=5, random_state=42
)
random_search.fit(X_train, y_train)
```

### Optuna (贝叶斯优化)

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
    }
    model = XGBClassifier(**params)
    return cross_val_score(model, X, y, cv=5).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(f"最佳参数: {study.best_params}")
```

## 正则化

| 方法           | 描述                 | 适用场景     |
| -------------- | -------------------- | ------------ |
| L1 (Lasso)     | 稀疏解，特征选择     | 高维数据     |
| L2 (Ridge)     | 防止权重过大         | 多重共线性   |
| Dropout        | 随机丢弃神经元       | 深度学习     |
| Early Stopping | 验证集性能下降时停止 | 任何迭代模型 |

```python
# L2 正则化
model = LogisticRegression(C=0.1)  # C = 1/λ

# Dropout (PyTorch)
nn.Dropout(0.5)

# Early Stopping
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_iter_no_change=10, validation_fraction=0.1)
```

## 类别不平衡处理

```python
# 方法1: 类别权重
model = LogisticRegression(class_weight='balanced')

# 方法2: 过采样 (SMOTE)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# 方法3: 欠采样
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)
```
