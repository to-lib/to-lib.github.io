---
sidebar_position: 4
title: 🔧 数据预处理
---

# 数据预处理与特征工程

> 数据和特征决定了模型的上限，而算法只是逼近这个上限。

## 数据预处理流程

```mermaid
graph LR
    A[原始数据] --> B[数据清洗]
    B --> C[特征工程]
    C --> D[数据转换]
    D --> E[特征选择]
    E --> F[模型就绪数据]
```

## 数据清洗

### 处理缺失值

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

df = pd.DataFrame({
    'age': [25, 30, np.nan, 35, 28],
    'salary': [50000, np.nan, 75000, np.nan, 60000],
    'city': ['北京', '上海', np.nan, '深圳', '广州']
})

# 查看缺失值
print(df.isnull().sum())

# 方法1: 删除缺失值
df_clean = df.dropna()

# 方法2: 填充缺失值
df['age'].fillna(df['age'].mean(), inplace=True)        # 均值填充
df['salary'].fillna(df['salary'].median(), inplace=True) # 中位数填充
df['city'].fillna(df['city'].mode()[0], inplace=True)   # 众数填充

# 方法3: 使用 sklearn Imputer
imputer = SimpleImputer(strategy='mean')
df[['age', 'salary']] = imputer.fit_transform(df[['age', 'salary']])
```

| 填充策略 | 适用场景           | 优点         | 缺点           |
| -------- | ------------------ | ------------ | -------------- |
| 均值     | 正态分布的连续变量 | 简单         | 易受异常值影响 |
| 中位数   | 偏态分布的连续变量 | 鲁棒         | 可能不够精确   |
| 众数     | 分类变量           | 保持分布     | 可能引入偏差   |
| KNN      | 任意类型           | 利用相似样本 | 计算开销大     |

### 处理异常值

```python
import matplotlib.pyplot as plt

# 检测异常值 - IQR 方法
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

# 检测异常值 - Z-score 方法
from scipy import stats

def detect_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores > threshold]

# 处理异常值 - 截断
def clip_outliers(data, column, lower_percentile=1, upper_percentile=99):
    lower = data[column].quantile(lower_percentile / 100)
    upper = data[column].quantile(upper_percentile / 100)
    data[column] = data[column].clip(lower, upper)
    return data
```

### 处理重复值

```python
# 检测重复值
duplicates = df.duplicated()
print(f"重复行数: {duplicates.sum()}")

# 删除重复值
df_unique = df.drop_duplicates()

# 保留最后一条
df_unique = df.drop_duplicates(keep='last')
```

## 特征工程

### 数值特征处理

#### 标准化 (Standardization)

将特征转换为均值为 0，标准差为 1 的分布。

$$
z = \frac{x - \mu}{\sigma}
$$

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 逆转换
X_original = scaler.inverse_transform(X_scaled)
```

#### 归一化 (Normalization)

将特征缩放到 [0, 1] 范围。

$$
x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
```

| 方法           | 公式                | 适用场景           | 对异常值敏感 |
| -------------- | ------------------- | ------------------ | ------------ |
| StandardScaler | $(x-\mu)/\sigma$    | 正态分布、梯度下降 | 是           |
| MinMaxScaler   | $(x-min)/(max-min)$ | 需要固定范围       | 是           |
| RobustScaler   | $(x-median)/IQR$    | 有异常值           | 否           |

```python
from sklearn.preprocessing import RobustScaler

# 对异常值鲁棒的缩放
robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)
```

### 分类特征编码

#### One-Hot 编码

```python
from sklearn.preprocessing import OneHotEncoder

# 原始数据
categories = [['红色'], ['蓝色'], ['绿色'], ['红色']]

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(categories)
print(encoded)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]]

# Pandas 方法
df = pd.DataFrame({'color': ['红色', '蓝色', '绿色', '红色']})
df_encoded = pd.get_dummies(df, columns=['color'])
```

#### Label 编码

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
labels = ['低', '中', '高', '中', '低']
encoded = le.fit_transform(labels)  # [0, 2, 1, 2, 0]

# 逆转换
original = le.inverse_transform(encoded)
```

#### 目标编码 (Target Encoding)

```python
# 用目标变量的均值替换类别
def target_encode(df, column, target):
    means = df.groupby(column)[target].mean()
    return df[column].map(means)

# 示例
df['city_encoded'] = target_encode(df, 'city', 'salary')
```

| 编码方法 | 适用场景           | 优点       | 缺点         |
| -------- | ------------------ | ---------- | ------------ |
| One-Hot  | 无序类别，类别数少 | 不引入顺序 | 维度爆炸     |
| Label    | 有序类别           | 维度不变   | 引入虚假顺序 |
| Target   | 高基数类别         | 保留信息   | 易过拟合     |

### 特征构造

```python
# 创建新特征
df['age_squared'] = df['age'] ** 2
df['salary_per_age'] = df['salary'] / df['age']
df['log_salary'] = np.log1p(df['salary'])

# 时间特征
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'] >= 5

# 文本特征
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
```

### 特征分箱 (Binning)

```python
from sklearn.preprocessing import KBinsDiscretizer

# 等宽分箱
df['age_bin'] = pd.cut(df['age'], bins=5, labels=['青年', '青中', '中年', '中老', '老年'])

# 等频分箱
df['salary_bin'] = pd.qcut(df['salary'], q=4, labels=['低', '中低', '中高', '高'])

# sklearn 分箱
binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
df['age_binned'] = binner.fit_transform(df[['age']])
```

## 特征选择

### 过滤法 (Filter)

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# 基于方差
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)

# 基于统计检验
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# 查看特征分数
scores = pd.DataFrame({
    'feature': feature_names,
    'score': selector.scores_
}).sort_values('score', ascending=False)
```

### 嵌入法 (Embedded)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# 使用随机森林的特征重要性
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 获取特征重要性
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# 选择重要特征
selector = SelectFromModel(rf, threshold='mean')
X_selected = selector.fit_transform(X, y)
```

### 包装法 (Wrapper)

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# 递归特征消除
model = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=model, n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)

# 查看排名
ranking = pd.DataFrame({
    'feature': feature_names,
    'ranking': rfe.ranking_
}).sort_values('ranking')
```

## 数据处理 Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# 定义数值和分类特征
numeric_features = ['age', 'salary']
categorical_features = ['city', 'gender']

# 数值特征处理
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 分类特征处理
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 组合转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 完整 Pipeline（包含模型）
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# 训练和预测
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## 小结

| 阶段     | 关键操作               | 常用工具                         |
| -------- | ---------------------- | -------------------------------- |
| 数据清洗 | 缺失值、异常值、重复值 | `pandas`, `SimpleImputer`        |
| 数值处理 | 标准化、归一化         | `StandardScaler`, `MinMaxScaler` |
| 分类编码 | One-Hot、Label、Target | `OneHotEncoder`, `LabelEncoder`  |
| 特征构造 | 交互、多项式、时间     | `pandas`, `PolynomialFeatures`   |
| 特征选择 | 过滤、嵌入、包装       | `SelectKBest`, `RFE`             |
| Pipeline | 自动化流程             | `Pipeline`, `ColumnTransformer`  |
