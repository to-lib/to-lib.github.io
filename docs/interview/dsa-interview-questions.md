---
sidebar_position: 18
title: 面试题集
---

# 面试题集

## 🧩 技巧模板速查

- [双指针](/docs/dsa/two-pointers)
- [滑动窗口](/docs/dsa/sliding-window)
- [前缀和与差分](/docs/dsa/prefix-sum-diff)
- [位运算](/docs/dsa/bit-manipulation)
- [并查集](/docs/dsa/union-find)

## 📚 数组与字符串

1. **两数之和** - 哈希表 O(n)
2. **三数之和** - 排序 + 双指针
3. **最长无重复子串** - 滑动窗口
4. **最大子数组和** - 动态规划/Kadane

## 🔗 链表

1. **反转链表** - 迭代/递归
2. **检测环** - 快慢指针
3. **合并有序链表** - 双指针
4. **删除倒数第 N 个节点** - 快慢指针

## 🌳 树

1. **二叉树遍历** - 前中后层序
2. **最大深度** - 递归/BFS
3. **验证 BST** - 中序遍历
4. **最近公共祖先** - 递归

## 📊 动态规划

1. **爬楼梯** - dp[i] = dp[i-1] + dp[i-2]
2. **背包问题** - 选/不选
3. **最长递增子序列** - O(n log n)
4. **编辑距离** - 二维 DP

## 🎯 高频算法

### 快速选择 - 第 K 大元素

```java
public int findKthLargest(int[] nums, int k) {
    return quickSelect(nums, 0, nums.length - 1, nums.length - k);
}

private int quickSelect(int[] nums, int l, int r, int k) {
    int pivot = nums[r], p = l;
    for (int i = l; i < r; i++) {
        if (nums[i] <= pivot) swap(nums, i, p++);
    }
    swap(nums, p, r);
    if (p == k) return nums[p];
    return p < k ? quickSelect(nums, p+1, r, k) : quickSelect(nums, l, p-1, k);
}
```

### 并查集

```java
class UnionFind {
    private int[] parent;

    public UnionFind(int n) {
        parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    public int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }

    public void union(int x, int y) {
        parent[find(x)] = find(y);
    }
}
```

## 💡 面试技巧

1. 确认输入范围和边界
2. 先说思路再写代码
3. 分析时间空间复杂度
4. 主动测试边界用例
