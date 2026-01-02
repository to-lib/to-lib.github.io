---
sidebar_position: 40
title: 🔧 特征工程进阶
---

# 特征工程进阶

高级特征工程技术与自动化方法。

## 特征交叉

```python
from sklearn.preprocessing import PolynomialFeatures

# 多项式特征
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X)

# 手动交叉
df['feature_cross'] = df['feature1'] * df['feature2']
df['feature_ratio'] = df['feature1'] / (df['feature2'] + 1e-8)
```

## Target Encoding

```python
import category_encoders as ce

# 目标编码
encoder = ce.TargetEncoder(cols=['category_col'])
X_encoded = encoder.fit_transform(X, y)

# 带平滑的目标编码
def target_encode_smooth(df, col, target, m=10):
    global_mean = df[target].mean()
    agg = df.groupby(col)[target].agg(['mean', 'count'])
    smooth = (agg['count'] * agg['mean'] + m * global_mean) / (agg['count'] + m)
    return df[col].map(smooth)
```

## 时间特征

```python
def create_time_features(df, date_col):
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['hour'] = df[date_col].dt.hour
    df['is_weekend'] = df['dayofweek'] >= 5
    df['is_month_start'] = df[date_col].dt.is_month_start
    df['is_month_end'] = df[date_col].dt.is_month_end

    # 周期性编码
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df
```

## 聚合特征

```python
def create_agg_features(df, group_col, agg_col):
    aggs = df.groupby(group_col)[agg_col].agg(['mean', 'std', 'min', 'max', 'count'])
    aggs.columns = [f'{agg_col}_{stat}' for stat in aggs.columns]
    return df.merge(aggs, on=group_col, how='left')
```

## 自动特征工程

### Featuretools

```python
import featuretools as ft

# 定义实体
es = ft.EntitySet(id='data')
es.add_dataframe(dataframe=df, dataframe_name='main', index='id')

# 深度特征合成
features, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name='main',
    max_depth=2
)
```

### OpenFE

```python
from openfe import OpenFE

ofe = OpenFE()
features = ofe.fit(X_train, y_train, n_jobs=-1)

# 应用特征
X_train_new = ofe.transform(X_train)
X_test_new = ofe.transform(X_test)
```

## 特征选择进阶

### Boruta

```python
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1)
boruta = BorutaPy(rf, n_estimators='auto', verbose=2)
boruta.fit(X.values, y.values)

selected_features = X.columns[boruta.support_].tolist()
```

### SHAP 特征选择

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 基于 SHAP 重要性选择
importance = np.abs(shap_values).mean(axis=0)
top_features = np.argsort(importance)[-20:]
```

## 特征存储

```python
# Feast 示例
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# 获取历史特征
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["user_features:age", "user_features:income"]
).to_df()

# 在线获取
online_features = store.get_online_features(
    features=["user_features:age"],
    entity_rows=[{"user_id": 1001}]
).to_dict()
```
