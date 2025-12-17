---
sidebar_position: 5
title: 哈希表
---

# 哈希表

哈希表通过哈希函数将键映射到数组索引，实现 O(1) 平均时间的查找、插入、删除。

## 📖 基本原理

### 哈希函数

```java
// 简单取模哈希
public int hash(int key, int capacity) {
    return key % capacity;
}

// 字符串哈希
public int hashString(String key, int capacity) {
    int hash = 0;
    for (char c : key.toCharArray()) {
        hash = (hash * 31 + c) % capacity;
    }
    return hash;
}
```

### 冲突解决

```mermaid
graph LR
    subgraph Hash_Table [哈希表 (链地址法)]
        direction LR
        idx0[Idx 0] --> A[Key: A, Val: 1] --> B[Key: B, Val: 2] --> Null0[Null]
        idx1[Idx 1] --> C[Key: C, Val: 3] --> Null1[Null]
        idx2[Idx 2] --> Null2[Null]
    end
```

1. **链地址法** - 每个位置是一个链表
2. **开放地址法** - 线性探测、二次探测

## 🔧 手动实现

```java
public class MyHashMap {
    private class Node {
        int key, value;
        Node next;
        Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }

    private Node[] buckets;
    private int capacity = 1024;

    public MyHashMap() {
        buckets = new Node[capacity];
    }

    private int hash(int key) {
        return key % capacity;
    }

    public void put(int key, int value) {
        int idx = hash(key);
        if (buckets[idx] == null) {
            buckets[idx] = new Node(key, value);
            return;
        }
        Node curr = buckets[idx];
        while (curr != null) {
            if (curr.key == key) {
                curr.value = value;
                return;
            }
            if (curr.next == null) break;
            curr = curr.next;
        }
        curr.next = new Node(key, value);
    }

    public int get(int key) {
        int idx = hash(key);
        Node curr = buckets[idx];
        while (curr != null) {
            if (curr.key == key) return curr.value;
            curr = curr.next;
        }
        return -1;
    }
}
```

## 📚 Java HashMap

```java
Map<String, Integer> map = new HashMap<>();
map.put("a", 1);
map.get("a");            // 1
map.getOrDefault("b", 0); // 0
map.containsKey("a");    // true
map.remove("a");

// 遍历
for (Map.Entry<String, Integer> entry : map.entrySet()) {
    System.out.println(entry.getKey() + ": " + entry.getValue());
}
```

## 🎯 经典应用

### 两数之和

```java
public int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < nums.length; i++) {
        int complement = target - nums[i];
        if (map.containsKey(complement)) {
            return new int[]{map.get(complement), i};
        }
        map.put(nums[i], i);
    }
    return new int[]{};
}
```

### 统计频率

```java
public Map<Integer, Integer> countFreq(int[] nums) {
    Map<Integer, Integer> freq = new HashMap<>();
    for (int num : nums) {
        freq.put(num, freq.getOrDefault(num, 0) + 1);
    }
    return freq;
}
```
