---
sidebar_position: 26
title: 线段树（Segment Tree）
---

# 线段树（Segment Tree）

线段树用于维护区间信息，典型支持：

- `update(pos, val)`：单点更新（或单点增量）
- `query(l, r)`：区间查询（和 / 最小值 / 最大值 等）

它的核心优势是：在 `O(log n)` 内完成更新与查询，适合“需要动态修改 + 区间统计”的问题。

## 📌 适用场景

- 区间求和 / 区间最值（min/max）
- 需要频繁更新数组元素，并查询任意区间的统计
- 进阶：懒标记（lazy propagation）可支持区间更新（区间加、区间赋值）

## ✅ Java 8 模板（单点更新 + 区间求和）

约定：原数组使用 0-based 下标，线段树维护 `[0..n-1]`。

```java
class SegmentTree {
    private final int n;
    private final long[] tree;

    public SegmentTree(int[] nums) {
        this.n = nums.length;
        this.tree = new long[Math.max(1, 4 * n)];
        if (n > 0) build(1, 0, n - 1, nums);
    }

    private void build(int node, int l, int r, int[] nums) {
        if (l == r) {
            tree[node] = nums[l];
            return;
        }
        int mid = (l + r) >>> 1;
        build(node << 1, l, mid, nums);
        build(node << 1 | 1, mid + 1, r, nums);
        tree[node] = tree[node << 1] + tree[node << 1 | 1];
    }

    // 单点赋值：nums[pos] = val
    public void update(int pos, long val) {
        update(1, 0, n - 1, pos, val);
    }

    private void update(int node, int l, int r, int pos, long val) {
        if (l == r) {
            tree[node] = val;
            return;
        }
        int mid = (l + r) >>> 1;
        if (pos <= mid) update(node << 1, l, mid, pos, val);
        else update(node << 1 | 1, mid + 1, r, pos, val);
        tree[node] = tree[node << 1] + tree[node << 1 | 1];
    }

    // 区间和：sum(nums[ql..qr])
    public long query(int ql, int qr) {
        if (ql > qr) return 0;
        return query(1, 0, n - 1, ql, qr);
    }

    private long query(int node, int l, int r, int ql, int qr) {
        if (ql <= l && r <= qr) return tree[node];
        int mid = (l + r) >>> 1;
        long res = 0;
        if (ql <= mid) res += query(node << 1, l, mid, ql, qr);
        if (qr > mid) res += query(node << 1 | 1, mid + 1, r, ql, qr);
        return res;
    }
}
```

## 🎯 常见扩展

### 1) 区间最小值 / 最大值

把 `tree[node]` 的含义从“和”改为“min / max”，并把 `pushUp` 的合并从 `+` 改为 `Math.min/Math.max`。

### 2) 区间更新（Lazy Propagation）

如果题目需要：

- 对区间 `[l, r]` 全部加 `delta`
- 同时支持区间查询

则需要 `lazy[]`，在访问子节点前把标记下推（pushDown）。这块建议在需要时再单独写一个带懒标记版本，避免模板过重。

## ✅ 复杂度

- 建树：`O(n)`
- 单点更新：`O(log n)`
- 区间查询：`O(log n)`
- 空间：`O(n)`（通常用 `4n` 数组）

## 💡 常见坑

- 边界：`mid` 的计算与递归区间划分要严格保证不死循环（常用 `[l, mid]` 与 `[mid+1, r]`）。
- 空数组：`n = 0` 时要避免 build/query/update 访问越界。
- long 溢出：区间和场景建议用 `long`。
- 与 BIT 的选择：
  - BIT 更轻量，擅长前缀/可逆区间统计。
  - 线段树更通用，适合复杂区间信息与懒标记区间更新。
