---
sidebar_position: 2
title: 时间空间复杂度
---

# 时间空间复杂度

算法复杂度分析是评估算法效率的核心技能，帮助我们在编写代码前预判性能表现。

## 📖 大 O 表示法

大 O 表示法（Big O Notation）描述算法执行时间或空间随输入规模增长的**上界**。

### 常见复杂度等级

| 复杂度     | 名称     | 示例         |
| ---------- | -------- | ------------ |
| O(1)       | 常数     | 数组索引访问 |
| O(log n)   | 对数     | 二分查找     |
| O(n)       | 线性     | 遍历数组     |
| O(n log n) | 线性对数 | 归并排序     |
| O(n²)      | 平方     | 冒泡排序     |
| O(2ⁿ)      | 指数     | 递归斐波那契 |
| O(n!)      | 阶乘     | 全排列       |

### 增长趋势对比

```
n=10:     1 < 3 < 10 < 33 < 100 < 1024 < 3628800
n=100:    1 < 7 < 100 < 664 < 10000 < 10³⁰ < 10¹⁵⁸
          O(1) O(logn) O(n) O(nlogn) O(n²) O(2ⁿ) O(n!)
```

```mermaid
graph LR
    subgraph Complexity_Growth [复杂度增长趋势]
        direction LR
        O1[O(1)] --> Olog[O(log n)] --> On[O(n)] --> Onlog[O(n log n)] --> On2[O(n^2)] --> O2n[O(2^n)] --> Onfact[O(n!)]
        style O1 fill:#e6fffa,stroke:#00bcd4
        style Olog fill:#e6fffa,stroke:#00bcd4
        style On fill:#e8f5e9,stroke:#4caf50
        style Onlog fill:#fff3e0,stroke:#ff9800
        style On2 fill:#ffebee,stroke:#f44336
        style O2n fill:#ffebee,stroke:#b71c1c
        style Onfact fill:#ffebee,stroke:#b71c1c
    end
```

## ⏱️ 时间复杂度

### O(1) - 常数时间

```java
// 数组随机访问
public int getElement(int[] arr, int index) {
    return arr[index];  // O(1)
}

// 哈希表查找
public String getValue(Map<String, String> map, String key) {
    return map.get(key);  // O(1) 平均
}
```

### O(log n) - 对数时间

```java
// 二分查找
public int binarySearch(int[] arr, int target) {
    int left = 0, right = arr.length - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;  // O(log n)
}
```

### O(n) - 线性时间

```java
// 线性查找
public int linearSearch(int[] arr, int target) {
    for (int i = 0; i < arr.length; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;  // O(n)
}

// 求和
public int sum(int[] arr) {
    int total = 0;
    for (int num : arr) {
        total += num;
    }
    return total;  // O(n)
}
```

### O(n log n) - 线性对数时间

```java
// 归并排序
public void mergeSort(int[] arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);      // T(n/2)
        mergeSort(arr, mid + 1, right); // T(n/2)
        merge(arr, left, mid, right);   // O(n)
    }
}
// 总复杂度: O(n log n)
```

### O(n²) - 平方时间

```java
// 冒泡排序
public void bubbleSort(int[] arr) {
    int n = arr.length;
    for (int i = 0; i < n - 1; i++) {           // n 次
        for (int j = 0; j < n - 1 - i; j++) {   // n 次
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}  // O(n²)
```

## 💾 空间复杂度

### O(1) 原地算法

```java
// 原地反转数组
public void reverse(int[] arr) {
    int left = 0, right = arr.length - 1;
    while (left < right) {
        int temp = arr[left];
        arr[left] = arr[right];
        arr[right] = temp;
        left++;
        right--;
    }
}  // 空间 O(1)
```

### O(n) 线性空间

```java
// 复制数组
public int[] copyArray(int[] arr) {
    int[] copy = new int[arr.length];  // 额外 O(n) 空间
    for (int i = 0; i < arr.length; i++) {
        copy[i] = arr[i];
    }
    return copy;
}
```

### O(log n) 递归栈空间

```java
// 二分查找递归版本
public int binarySearchRecursive(int[] arr, int target, int left, int right) {
    if (left > right) return -1;

    int mid = left + (right - left) / 2;
    if (arr[mid] == target) return mid;

    if (arr[mid] < target) {
        return binarySearchRecursive(arr, target, mid + 1, right);
    } else {
        return binarySearchRecursive(arr, target, left, mid - 1);
    }
}  // 空间 O(log n) - 递归调用栈
```

## 📊 最好、最坏、平均复杂度

以**快速排序**为例：

| 情况 | 时间复杂度 | 说明         |
| ---- | ---------- | ------------ |
| 最好 | O(n log n) | 每次均匀分割 |
| 平均 | O(n log n) | 随机数据     |
| 最坏 | O(n²)      | 已排序或逆序 |

```java
// 快速排序
public void quickSort(int[] arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

private int partition(int[] arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr, i, j);
        }
    }
    swap(arr, i + 1, high);
    return i + 1;
}

private void swap(int[] arr, int i, int j) {
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}
```

## 🎯 复杂度分析技巧

### 1. 循环次数法

```java
// 单层循环: O(n)
for (int i = 0; i < n; i++) { ... }

// 嵌套循环: O(n²)
for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) { ... }
}

// 循环减半: O(log n)
for (int i = n; i > 0; i /= 2) { ... }
```

### 2. 递归主定理

对于形如 `T(n) = aT(n/b) + f(n)` 的递归：

- 归并排序: `T(n) = 2T(n/2) + O(n)` → O(n log n)
- 二分查找: `T(n) = T(n/2) + O(1)` → O(log n)

### 3. 摊销分析

```java
// ArrayList 动态扩容
// 单次添加可能是 O(n)（扩容时）
// 但平均摊销下来是 O(1)
List<Integer> list = new ArrayList<>();
for (int i = 0; i < n; i++) {
    list.add(i);  // 摊销 O(1)
}
```

## 📋 常见算法复杂度总结

| 算法     | 时间(平均) | 时间(最坏) | 空间     |
| -------- | ---------- | ---------- | -------- |
| 冒泡排序 | O(n²)      | O(n²)      | O(1)     |
| 选择排序 | O(n²)      | O(n²)      | O(1)     |
| 插入排序 | O(n²)      | O(n²)      | O(1)     |
| 归并排序 | O(n log n) | O(n log n) | O(n)     |
| 快速排序 | O(n log n) | O(n²)      | O(log n) |
| 堆排序   | O(n log n) | O(n log n) | O(1)     |
| 二分查找 | O(log n)   | O(log n)   | O(1)     |
| 哈希查找 | O(1)       | O(n)       | O(n)     |

## 💡 面试技巧

1. **先说复杂度，再写代码** - 展示算法思维
2. **考虑边界情况** - 空输入、单元素等
3. **权衡时空复杂度** - 空间换时间是常见优化
4. **了解常量因子** - O(100n) 实际可能比 O(n²) 慢

> [!TIP]
> 面试中，如果面试官问"能否优化"，通常意味着存在更低复杂度的解法。
