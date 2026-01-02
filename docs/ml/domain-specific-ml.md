---
sidebar_position: 44
title: 🏥 领域特定 ML
---

# 领域特定机器学习

不同领域有其特定的数据特点、挑战和最佳实践。

## 医疗健康

### 医学影像

```python
# 医学图像处理
import SimpleITK as sitk
import nibabel as nib

# 读取 DICOM/NIfTI
image = sitk.ReadImage("scan.nii.gz")
data = sitk.GetArrayFromImage(image)

# 预处理
def preprocess_medical(image):
    # 窗位窗宽
    image = np.clip(image, -100, 400)
    # 标准化
    image = (image - image.mean()) / image.std()
    return image
```

### 挑战与解决方案

| 挑战         | 解决方案           |
| ------------ | ------------------ |
| 数据稀缺     | 迁移学习、数据增强 |
| 类别不平衡   | Focal Loss、过采样 |
| 隐私保护     | 联邦学习、差分隐私 |
| 可解释性要求 | Grad-CAM、SHAP     |

## 金融

### 欺诈检测

```python
# 极端不平衡处理
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

smote = SMOTE(sampling_strategy=0.5)
X_res, y_res = smote.fit_resample(X_train, y_train)

# 或使用专门的分类器
clf = BalancedRandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
```

### 风控特征

```python
def create_risk_features(df):
    # 行为特征
    df['txn_count_1h'] = df.groupby('user_id')['txn_time'].transform(
        lambda x: x.rolling('1H').count()
    )
    # 统计特征
    df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    # 图特征 - 资金网络
    return df
```

## 工业制造

### 预测性维护

```python
# 时间序列异常检测
from pyod.models.iforest import IForest

# 传感器数据特征
features = create_sensor_features(sensor_data)
detector = IForest(contamination=0.01)
anomalies = detector.fit_predict(features)
```

### 缺陷检测

```python
# 使用预训练模型 + 少样本
from anomalib.models import Padim

model = Padim()
model.fit(normal_images)  # 只需正常样本
predictions = model.predict(test_images)
```

## 零售电商

### 需求预测

```python
# 多层级时序预测
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, ETS

sf = StatsForecast(
    models=[AutoARIMA(), ETS()],
    freq='D'
)
forecasts = sf.forecast(df, h=30)
```

### 推荐系统

```python
# 实时推荐
from recbole.quick_start import run_recbole

config = {
    'model': 'SASRec',
    'dataset': 'ml-1m',
    'epochs': 100
}
run_recbole(config_dict=config)
```

## 领域通用建议

| 领域 | 数据特点             | 关键技术           |
| ---- | -------------------- | ------------------ |
| 医疗 | 高维、小样本、隐私   | 迁移学习、联邦学习 |
| 金融 | 不平衡、时序、对抗   | 异常检测、图网络   |
| 工业 | 传感器、实时、可靠性 | 时序分析、边缘部署 |
| 零售 | 高维稀疏、季节性     | 推荐、需求预测     |
