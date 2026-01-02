---
sidebar_position: 18
title: 🔍 可解释性
---

# 模型可解释性

理解模型为什么做出这样的预测，对于模型调试和业务信任至关重要。

## 特征重要性

### 树模型内置重要性

```python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# 特征重要性
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# 可视化
plt.barh(importance['feature'][:10], importance['importance'][:10])
plt.xlabel('重要性')
plt.title('特征重要性 Top 10')
```

### 排列重要性

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10)

importance = pd.DataFrame({
    'feature': feature_names,
    'importance': result.importances_mean,
    'std': result.importances_std
}).sort_values('importance', ascending=False)
```

## SHAP

SHAP (SHapley Additive exPlanations) 基于博弈论，为每个特征分配贡献值。

```python
import shap

# 创建解释器
explainer = shap.TreeExplainer(model)  # 树模型
# explainer = shap.KernelExplainer(model.predict, X_train[:100])  # 通用

# 计算 SHAP 值
shap_values = explainer.shap_values(X_test)
```

### 全局解释

```python
# 特征重要性汇总图
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# 条形图
shap.summary_plot(shap_values, X_test, plot_type='bar')
```

### 单样本解释

```python
# 瀑布图 - 解释单个预测
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=feature_names
))

# 力图
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

### 特征交互

```python
# 依赖图
shap.dependence_plot('feature_name', shap_values, X_test)
```

## LIME

LIME (Local Interpretable Model-agnostic Explanations) 用简单模型局部近似复杂模型。

```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=['负', '正'],
    mode='classification'
)

# 解释单个预测
exp = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10
)

exp.show_in_notebook()
# 或保存为 HTML
exp.save_to_file('explanation.html')
```

## 部分依赖图 (PDP)

展示特征与预测的边际效应。

```python
from sklearn.inspection import PartialDependenceDisplay

# 单特征 PDP
PartialDependenceDisplay.from_estimator(model, X_train, ['feature1', 'feature2'])
plt.show()

# 双特征交互 PDP
PartialDependenceDisplay.from_estimator(
    model, X_train, [('feature1', 'feature2')]
)
```

## ICE 图

Individual Conditional Expectation - 每个样本的 PDP。

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    model, X_train, ['feature1'],
    kind='both'  # 同时显示 PDP 和 ICE
)
```

## 方法对比

| 方法       | 作用范围  | 模型无关    | 计算速度 |
| ---------- | --------- | ----------- | -------- |
| 特征重要性 | 全局      | 否 (树模型) | 快       |
| 排列重要性 | 全局      | 是          | 中       |
| SHAP       | 全局+局部 | 是          | 慢       |
| LIME       | 局部      | 是          | 中       |
| PDP        | 全局      | 是          | 中       |

## 最佳实践

1. **从特征重要性开始**：快速了解哪些特征重要
2. **用 SHAP 深入分析**：理解特征如何影响预测
3. **用 LIME 解释个例**：向业务方解释具体预测
4. **交叉验证解释结果**：不同方法结果应该一致
5. **警惕相关特征**：高度相关的特征会分散重要性
