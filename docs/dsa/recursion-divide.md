---
sidebar_position: 11
title: 递归与分治
---

# 递归与分治

## 📖 递归三要素

1. **终止条件**：何时停止递归
2. **递归调用**：问题规模缩小
3. **返回值**：小问题的解如何组合

## 🔧 经典递归

### 斐波那契

```java
// 带记忆化
public int fib(int n, int[] memo) {
    if (n <= 1) return n;
    if (memo[n] != 0) return memo[n];
    memo[n] = fib(n - 1, memo) + fib(n - 2, memo);
    return memo[n];
}
```

### 汉诺塔

```java
public void hanoi(int n, char from, char to, char aux) {
    if (n == 1) {
        System.out.println(from + " -> " + to);
        return;
    }
    hanoi(n - 1, from, aux, to);
    System.out.println(from + " -> " + to);
    hanoi(n - 1, aux, to, from);
}
```

## 🎯 分治策略

### 归并排序

```java
public void mergeSort(int[] arr, int l, int r) {
    if (l >= r) return;
    int mid = (l + r) / 2;
    mergeSort(arr, l, mid);       // 分
    mergeSort(arr, mid + 1, r);   // 分
    merge(arr, l, mid, r);        // 治
}
```

### 快速幂

```java
public long power(long base, int exp, int mod) {
    if (exp == 0) return 1;
    long half = power(base, exp / 2, mod);
    return exp % 2 == 0 ? half * half % mod : half * half % mod * base % mod;
}
```
