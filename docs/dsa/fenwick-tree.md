---
sidebar_position: 25
title: 树状数组（Fenwick Tree）
---

# 树状数组（Fenwick Tree）

树状数组（Fenwick Tree / Binary Indexed Tree, BIT）用于维护一个数组的前缀信息，支持：

- `add(i, delta)`：单点增量更新
- `sum(i)`：查询前缀和 `a[1..i]`

它的核心优势是把“前缀查询 + 单点更新”都做到 `O(log n)`，实现也比线段树更轻量。

## 📌 适用场景

- 单点更新 + 区间求和（或求最大值等可逆/可组合信息）
- 频繁修改数组元素，同时需要快速得到前缀/区间统计
- 典型题：动态前缀和、逆序对、动态频次统计、离散化后计数

## 🧠 lowbit 与结构含义

定义：

- `lowbit(x) = x & (-x)`：取出 `x` 的二进制最低位的 1 所代表的值。

树状数组用 1-based 下标，`tree[i]` 管理一个长度为 `lowbit(i)` 的区间：

- `tree[i]` 覆盖 `(i - lowbit(i) + 1 .. i)`

## ✅ Java 8 模板（前缀和）

```java
class FenwickTree {
    private final int n;
    private final long[] tree;

    public FenwickTree(int n) {
        this.n = n;
        this.tree = new long[n + 1];
    }

    private int lowbit(int x) {
        return x & -x;
    }

    // 单点加：a[idx] += delta
    public void add(int idx, long delta) {
        for (int i = idx; i <= n; i += lowbit(i)) {
            tree[i] += delta;
        }
    }

    // 前缀和：sum(a[1..idx])
    public long sum(int idx) {
        long res = 0;
        for (int i = idx; i > 0; i -= lowbit(i)) {
            res += tree[i];
        }
        return res;
    }

    // 区间和：sum(a[l..r])
    public long rangeSum(int l, int r) {
        if (l > r) return 0;
        return sum(r) - sum(l - 1);
    }
}
```

## 🎯 常见扩展

### 1) 动态数组初始化

如果你有初始数组 `a[1..n]`，可以逐个 `add(i, a[i])` 构建。

### 2) 逆序对 / 统计“小于等于 x 的个数”

常见套路：

- 对值域做离散化（把值映射到 `1..m`）
- 从左到右或从右到左扫描
- 用 BIT 维护已出现元素的频次

示例（从左到右统计“前面有多少个比当前大”）：

- `seen = i - 1`
- `le = bit.sum(rank)`（&lt;= 当前的数量）
- `greater = seen - le`

### 3) 区间加 + 单点查（差分 BIT）

如果你要支持：

- 区间 `[l, r]` 全部加 `delta`
- 查询某个点 `a[i]`

可以维护差分数组 `d` 的 BIT：

- `rangeAdd(l, r, delta)`：`add(l, delta)`, `add(r+1, -delta)`
- `get(i)`：`sum(i)`

## ✅ 复杂度

- `add`：`O(log n)`
- `sum` / `rangeSum`：`O(log n)`
- 空间：`O(n)`

## 💡 常见坑

- 下标从 1 开始：如果你的原数组是 0-based，需要整体 `+1` 映射。
- 用 `int` 可能溢出：求和建议用 `long`。
- 只有“前缀可拆分”的信息适合 BIT：比如和、频次、异或等；复杂区间信息更适合线段树。
