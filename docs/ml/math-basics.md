---
sidebar_position: 3
title: 📐 数学基础
---

# 机器学习数学基础

机器学习的核心建立在三大数学支柱之上：**线性代数**、**概率统计**和**微积分**。

## 线性代数

### 向量与矩阵

```python
import numpy as np

# 向量：一维数组
v = np.array([1, 2, 3])

# 矩阵：二维数组
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(f"向量形状: {v.shape}")  # (3,)
print(f"矩阵形状: {A.shape}")  # (3, 3)
```

### 常用运算

| 运算       | 符号                          | NumPy 实现                | 应用场景         |
| ---------- | ----------------------------- | ------------------------- | ---------------- |
| 点积       | $\mathbf{a} \cdot \mathbf{b}$ | `np.dot(a, b)` 或 `a @ b` | 相似度计算       |
| 矩阵乘法   | $AB$                          | `A @ B`                   | 神经网络前向传播 |
| 转置       | $A^T$                         | `A.T`                     | 数据变换         |
| 逆矩阵     | $A^{-1}$                      | `np.linalg.inv(A)`        | 求解线性方程组   |
| 行列式     | $\det(A)$                     | `np.linalg.det(A)`        | 判断矩阵可逆性   |
| 特征值分解 | $A = Q\Lambda Q^{-1}$         | `np.linalg.eig(A)`        | PCA 降维         |

```python
# 向量点积
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = a @ b  # 1*4 + 2*5 + 3*6 = 32

# 矩阵乘法
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B

# 特征值分解
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")
```

### 范数 (Norm)

范数衡量向量的"大小"，在正则化中广泛使用。

$$
\|\mathbf{x}\|_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p}
$$

| 范数    | 公式                | 含义         | 应用         |
| ------- | ------------------- | ------------ | ------------ | ---------- | ------------ |
| L1 范数 | $\sum               | x_i          | $            | 曼哈顿距离 | Lasso 正则化 |
| L2 范数 | $\sqrt{\sum x_i^2}$ | 欧几里得距离 | Ridge 正则化 |
| L∞ 范数 | $\max               | x_i          | $            | 最大绝对值 | 鲁棒优化     |

```python
x = np.array([3, 4])

l1_norm = np.linalg.norm(x, ord=1)  # 7
l2_norm = np.linalg.norm(x, ord=2)  # 5
linf_norm = np.linalg.norm(x, ord=np.inf)  # 4
```

## 概率统计

### 概率基础

**条件概率**：

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}
$$

**贝叶斯定理**：

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

```python
# 朴素贝叶斯分类器示例
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

nb = GaussianNB()
nb.fit(X, y)

# 预测新样本的概率分布
proba = nb.predict_proba([[5.0, 3.4, 1.5, 0.2]])
print(f"各类别概率: {proba}")
```

### 常见概率分布

| 分布       | 公式/特点                     | Python 实现                   | 应用         |
| ---------- | ----------------------------- | ----------------------------- | ------------ |
| 正态分布   | $\mu$：均值，$\sigma$：标准差 | `np.random.normal(mu, sigma)` | 连续特征建模 |
| 伯努利分布 | 二元结果，成功概率 $p$        | `np.random.binomial(1, p)`    | 二分类       |
| 多项分布   | 多个离散结果                  | `np.random.multinomial`       | 多分类       |
| 均匀分布   | 区间内等概率                  | `np.random.uniform(a, b)`     | 随机初始化   |

```python
import matplotlib.pyplot as plt
from scipy import stats

# 正态分布
x = np.linspace(-4, 4, 100)
y = stats.norm.pdf(x, loc=0, scale=1)

plt.figure(figsize=(10, 4))
plt.plot(x, y, label='标准正态分布 N(0,1)')
plt.fill_between(x, y, alpha=0.3)
plt.legend()
plt.title('正态分布概率密度函数')
plt.show()
```

### 统计指标

```python
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 集中趋势
mean = np.mean(data)       # 均值: 5.5
median = np.median(data)   # 中位数: 5.5

# 离散程度
variance = np.var(data)    # 方差: 8.25
std = np.std(data)         # 标准差: 2.87

# 相关性
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
correlation = np.corrcoef(x, y)[0, 1]  # 相关系数
print(f"相关系数: {correlation:.2f}")
```

### 最大似然估计 (MLE)

寻找使观测数据出现概率最大的参数值。

$$
\hat{\theta} = \arg\max_\theta \prod_{i=1}^{n} P(x_i | \theta)
$$

```python
# 用 MLE 估计正态分布参数
from scipy.stats import norm

# 观测数据
data = np.random.normal(loc=5, scale=2, size=1000)

# MLE 估计
mu_mle = np.mean(data)      # 均值估计
sigma_mle = np.std(data)    # 标准差估计

print(f"估计的均值: {mu_mle:.2f} (真实值: 5)")
print(f"估计的标准差: {sigma_mle:.2f} (真实值: 2)")
```

## 微积分

### 导数与梯度

**导数**：函数在某点的变化率

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

**梯度**：多元函数对各变量的偏导数组成的向量

$$
\nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n} \right)
$$

```python
# 数值求导
def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# 示例：f(x, y) = x^2 + y^2
def f(x):
    return x[0]**2 + x[1]**2

point = np.array([3.0, 4.0])
grad = numerical_gradient(f, point)
print(f"在点 (3, 4) 处的梯度: {grad}")  # [6, 8]
```

### 梯度下降

机器学习中最核心的优化算法。

$$
\theta_{new} = \theta_{old} - \eta \cdot \nabla L(\theta)
$$

其中 $\eta$ 是**学习率**，$L$ 是**损失函数**。

```python
def gradient_descent(f, grad_f, x0, learning_rate=0.1, max_iter=100):
    """梯度下降优化"""
    x = x0.copy()
    history = [x.copy()]

    for _ in range(max_iter):
        grad = grad_f(x)
        x = x - learning_rate * grad
        history.append(x.copy())

        if np.linalg.norm(grad) < 1e-6:
            break

    return x, history

# 示例：最小化 f(x, y) = x^2 + y^2
def f(x):
    return x[0]**2 + x[1]**2

def grad_f(x):
    return np.array([2*x[0], 2*x[1]])

x0 = np.array([5.0, 5.0])
x_min, history = gradient_descent(f, grad_f, x0, learning_rate=0.1)
print(f"最小值点: {x_min}")  # 接近 [0, 0]
```

### 常见损失函数的导数

| 损失函数      | 公式                           | 导数                      | 应用 |
| ------------- | ------------------------------ | ------------------------- | ---- |
| MSE           | $\frac{1}{n}\sum(y-\hat{y})^2$ | $-\frac{2}{n}(y-\hat{y})$ | 回归 |
| Cross-Entropy | $-\sum y\log\hat{y}$           | $\hat{y} - y$             | 分类 |
| Hinge Loss    | $\max(0, 1-y\cdot\hat{y})$     | $-y$ if $y\hat{y} < 1$    | SVM  |

```python
# MSE 损失及其梯度
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_gradient(y_true, y_pred):
    n = len(y_true)
    return -2 * (y_true - y_pred) / n
```

## 核心公式速查表

### 线性代数

| 公式                                                                             | 描述       |
| -------------------------------------------------------------------------------- | ---------- |
| $\|\mathbf{x}\|_2 = \sqrt{\sum x_i^2}$                                           | L2 范数    |
| $\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$ | 余弦相似度 |
| $A^T A$                                                                          | Gram 矩阵  |

### 概率统计

| 公式                                 | 描述       |
| ------------------------------------ | ---------- |
| $P(A\|B) = \frac{P(B\|A)P(A)}{P(B)}$ | 贝叶斯定理 |
| $\text{Var}(X) = E[(X-\mu)^2]$       | 方差       |
| $\sigma = \sqrt{\text{Var}(X)}$      | 标准差     |

### 微积分

| 公式                                          | 描述         |
| --------------------------------------------- | ------------ |
| $\theta := \theta - \eta \nabla L$            | 梯度下降更新 |
| $\frac{\partial}{\partial x}(x^n) = nx^{n-1}$ | 幂函数导数   |
| $\frac{d}{dx}\ln(x) = \frac{1}{x}$            | 对数导数     |
