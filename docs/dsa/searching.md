---
sidebar_position: 10
title: 查找算法
---

# 查找算法

## 🔍 二分查找

```mermaid
graph TD
    Start([开始]) --> Init[left = 0, right = n-1]
    Init --> Check{left <= right?}
    Check -- No --> NotFound([返回 -1])
    Check -- Yes --> CalcMid[mid = left + (right-left)/2]
    CalcMid --> Compare{arr[mid] == target?}
    Compare -- Yes --> Found([返回 mid])
    Compare -- No --> CheckSize{arr[mid] < target?}
    CheckSize -- Yes --> MoveLeft[left = mid + 1] --> Check
    CheckSize -- No --> MoveRight[right = mid - 1] --> Check
```

```java
public int binarySearch(int[] arr, int target) {
    int left = 0, right = arr.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}
```

## 🎯 二分变体

### 查找左边界

```java
public int leftBound(int[] arr, int target) {
    int left = 0, right = arr.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] >= target) right = mid - 1;
        else left = mid + 1;
    }
    return left < arr.length && arr[left] == target ? left : -1;
}
```

### 查找右边界

```java
public int rightBound(int[] arr, int target) {
    int left = 0, right = arr.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] <= target) left = mid + 1;
        else right = mid - 1;
    }
    return right >= 0 && arr[right] == target ? right : -1;
}
```

## 📊 复杂度

| 算法     | 时间     | 空间 | 要求 |
| -------- | -------- | ---- | ---- |
| 顺序查找 | O(n)     | O(1) | 无   |
| 二分查找 | O(log n) | O(1) | 有序 |
| 哈希查找 | O(1)     | O(n) | 无   |
