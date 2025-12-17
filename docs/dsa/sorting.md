---
sidebar_position: 9
title: 排序算法
---

# 排序算法

## 📊 复杂度对比

| 算法 | 平均       | 最坏       | 空间     | 稳定 |
| ---- | ---------- | ---------- | -------- | ---- |
| 冒泡 | O(n²)      | O(n²)      | O(1)     | ✅   |
| 选择 | O(n²)      | O(n²)      | O(1)     | ❌   |
| 插入 | O(n²)      | O(n²)      | O(1)     | ✅   |
| 归并 | O(n log n) | O(n log n) | O(n)     | ✅   |
| 快速 | O(n log n) | O(n²)      | O(log n) | ❌   |
| 堆排 | O(n log n) | O(n log n) | O(1)     | ❌   |

## 🔧 实现

### 冒泡排序

```mermaid
graph TD
    Start([开始]) --> LoopI{i < n-1?}
    LoopI -- Yes --> LoopJ{j < n-1-i?}
    LoopI -- No --> End([结束])
    LoopJ -- Yes --> Compare{arr[j] > arr[j+1]?}
    LoopJ -- No --> IncI[i++] --> LoopI
    Compare -- Yes --> Swap[交换 arr[j], arr[j+1]] --> IncJ[j++] --> LoopJ
    Compare -- No --> IncJ --> LoopJ
```

```java
public void bubbleSort(int[] arr) {
    for (int i = 0; i < arr.length - 1; i++) {
        for (int j = 0; j < arr.length - 1 - i; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}
```

### 快速排序

```mermaid
graph TD
    Start([Partition Start]) --> SetPivot[Pivot = arr[high]]
    SetPivot --> InitI[i = low - 1]
    InitI --> LoopJ{j < high?}
    LoopJ -- Yes --> CheckPivot{arr[j] < Pivot?}
    LoopJ -- No --> SwapPivot[Swap arr[i+1], arr[high]] --> End([Return i+1])
    CheckPivot -- Yes --> IncI[i++] --> Swap[Swap arr[i], arr[j]] --> IncJ[j++] --> LoopJ
    CheckPivot -- No --> IncJ --> LoopJ
```

```java
public void quickSort(int[] arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

private int partition(int[] arr, int low, int high) {
    int pivot = arr[high], i = low - 1;
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            int temp = arr[i]; arr[i] = arr[j]; arr[j] = temp;
        }
    }
    int temp = arr[i+1]; arr[i+1] = arr[high]; arr[high] = temp;
    return i + 1;
}
```

### 归并排序

```java
public void mergeSort(int[] arr, int left, int right) {
    if (left < right) {
        int mid = (left + right) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

private void merge(int[] arr, int left, int mid, int right) {
    int[] temp = new int[right - left + 1];
    int i = left, j = mid + 1, k = 0;
    while (i <= mid && j <= right) {
        temp[k++] = arr[i] <= arr[j] ? arr[i++] : arr[j++];
    }
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    System.arraycopy(temp, 0, arr, left, temp.length);
}
```
