---
sidebar_position: 7
title: 堆
---

# 堆

堆是一种完全二叉树，分为最大堆和最小堆。

## 📖 基本概念

```mermaid
graph TD
    subgraph Min_Heap [最小堆 (父 <= 子)]
        A((10)) --> B((20))
        A --> C((15))
        B --> D((30))
        B --> E((40))
        C --> F((50))
        C --> G((60))
    end
```

- **最大堆**：父节点 ≥ 子节点
- **最小堆**：父节点 ≤ 子节点

### 数组表示

对于索引 i 的节点：

- 父节点：(i - 1) / 2
- 左子节点：2 \* i + 1
- 右子节点：2 \* i + 2

## 🔧 最小堆实现

```java
public class MinHeap {
    private int[] heap;
    private int size;

    public MinHeap(int capacity) {
        heap = new int[capacity];
    }

    public void insert(int val) {
        heap[size] = val;
        siftUp(size++);
    }

    public int extractMin() {
        int min = heap[0];
        heap[0] = heap[--size];
        siftDown(0);
        return min;
    }

    private void siftUp(int i) {
        while (i > 0 && heap[(i-1)/2] > heap[i]) {
            swap(i, (i-1)/2);
            i = (i-1)/2;
        }
    }

    private void siftDown(int i) {
        while (2*i+1 < size) {
            int j = 2*i+1;
            if (j+1 < size && heap[j+1] < heap[j]) j++;
            if (heap[i] <= heap[j]) break;
            swap(i, j);
            i = j;
        }
    }

    private void swap(int i, int j) {
        int temp = heap[i];
        heap[i] = heap[j];
        heap[j] = temp;
    }
}
```

## 📚 Java PriorityQueue

```java
// 最小堆
PriorityQueue<Integer> minHeap = new PriorityQueue<>();

// 最大堆
PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a, b) -> b - a);

minHeap.offer(3);
minHeap.offer(1);
minHeap.poll();  // 返回 1
```

## 🎯 经典应用

### 堆排序

```java
public void heapSort(int[] arr) {
    int n = arr.length;
    // 建堆
    for (int i = n/2 - 1; i >= 0; i--) heapify(arr, n, i);
    // 排序
    for (int i = n - 1; i > 0; i--) {
        swap(arr, 0, i);
        heapify(arr, i, 0);
    }
}

private void heapify(int[] arr, int n, int i) {
    int largest = i, left = 2*i+1, right = 2*i+2;
    if (left < n && arr[left] > arr[largest]) largest = left;
    if (right < n && arr[right] > arr[largest]) largest = right;
    if (largest != i) {
        swap(arr, i, largest);
        heapify(arr, n, largest);
    }
}
```

### 前 K 个最大元素

```java
public int findKthLargest(int[] nums, int k) {
    PriorityQueue<Integer> minHeap = new PriorityQueue<>();
    for (int num : nums) {
        minHeap.offer(num);
        if (minHeap.size() > k) minHeap.poll();
    }
    return minHeap.peek();
}
```
