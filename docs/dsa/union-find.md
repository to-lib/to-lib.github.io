---
sidebar_position: 24
title: 并查集
---

# 并查集

并查集（Union-Find / Disjoint Set Union, DSU）用于维护一组不相交集合，支持两类核心操作：

- `find(x)`：查询元素 x 所属集合的代表（根）
- `union(x, y)`：合并 x 与 y 所属集合

典型应用：连通性、连通分量数量、无向图判环、Kruskal 最小生成树等。

## 📌 核心优化

- **路径压缩**：`find` 时把路径上的节点直接挂到根上
- **按秩/按大小合并**：把小树挂到大树，降低树高

## ✅ Java 8 实现（路径压缩 + 按大小合并）

```java
class UnionFind {
    private int[] parent;
    private int[] size;
    private int count;

    public UnionFind(int n) {
        parent = new int[n];
        size = new int[n];
        count = n;
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            size[i] = 1;
        }
    }

    public int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }

    public boolean union(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        if (ra == rb) return false;

        if (size[ra] < size[rb]) {
            int tmp = ra;
            ra = rb;
            rb = tmp;
        }

        parent[rb] = ra;
        size[ra] += size[rb];
        count--;
        return true;
    }

    public boolean connected(int a, int b) {
        return find(a) == find(b);
    }

    public int count() {
        return count;
    }
}
```

## 🎯 经典应用 1：无向图判环

如果在遍历边 `(u, v)` 时，发现 `u` 和 `v` 已经连通，那么加入这条边会形成环。

```java
public boolean hasCycle(int n, int[][] edges) {
    UnionFind uf = new UnionFind(n);
    for (int[] e : edges) {
        int u = e[0], v = e[1];
        if (!uf.union(u, v)) return true;
    }
    return false;
}
```

## 🎯 经典应用 2：连通分量数量

```java
public int components(int n, int[][] edges) {
    UnionFind uf = new UnionFind(n);
    for (int[] e : edges) uf.union(e[0], e[1]);
    return uf.count();
}
```

## 💡 注意点

- 并查集适合处理“合并 + 连通性查询”，不擅长处理路径信息（如最短路径）。
- 带权并查集/可撤销并查集是更进阶的变体，可在需要时再扩展。
