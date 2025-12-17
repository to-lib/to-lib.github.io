---
sidebar_position: 22
title: 前缀和与差分
---

# 前缀和与差分

前缀和（Prefix Sum）与差分（Difference Array）是处理“区间统计 / 区间更新”问题的两大基础工具。

## 📌 前缀和（Prefix Sum）

### 1) 一维前缀和：区间和查询

定义：
- `pre[i]` 表示 `nums[0..i-1]` 的和（长度 i 的前缀）
- 则区间 `[l, r]`（闭区间）的和为：`pre[r+1] - pre[l]`

```java
public int[] buildPrefixSum(int[] nums) {
    int[] pre = new int[nums.length + 1];
    for (int i = 0; i < nums.length; i++) {
        pre[i + 1] = pre[i] + nums[i];
    }
    return pre;
}

public int rangeSum(int[] pre, int l, int r) {
    return pre[r + 1] - pre[l];
}
```

### 2) 前缀和 + 哈希：子数组和等于 K

核心：如果 `pre[j] - pre[i] = k`，则 `pre[i] = pre[j] - k`。

```java
public int subarraySumEqualsK(int[] nums, int k) {
    Map<Integer, Integer> freq = new HashMap<>();
    freq.put(0, 1);

    int pre = 0;
    int ans = 0;
    for (int x : nums) {
        pre += x;
        ans += freq.getOrDefault(pre - k, 0);
        freq.put(pre, freq.getOrDefault(pre, 0) + 1);
    }
    return ans;
}
```

## 📌 差分（Difference Array）

差分数组常用于“多次区间加/减”的场景：
- 通过在区间端点打标记，把一次区间更新变为 O(1)
- 最后对差分数组做一次前缀和还原原数组

### 1) 一维差分：区间加法

对 `[l, r]` 区间每个元素加 `delta`：
- `diff[l] += delta`
- `diff[r + 1] -= delta`（若 `r + 1` 未越界）

```java
public int[] rangeAdd(int n, int[][] updates) {
    int[] diff = new int[n];

    for (int[] u : updates) {
        int l = u[0], r = u[1], delta = u[2];
        diff[l] += delta;
        if (r + 1 < n) diff[r + 1] -= delta;
    }

    int[] res = new int[n];
    int cur = 0;
    for (int i = 0; i < n; i++) {
        cur += diff[i];
        res[i] = cur;
    }

    return res;
}
```

## 🎯 适用场景总结

- **前缀和**：
  - 区间和查询
  - 子数组/子串统计（配合哈希）
- **差分**：
  - 多次区间更新
  - 扫描线类问题（差分 + 前缀还原）
