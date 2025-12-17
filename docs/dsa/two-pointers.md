---
sidebar_position: 20
title: 双指针
---

# 双指针

双指针（Two Pointers）是一类常用的解题技巧：用两个“指针”（通常是数组索引或链表指针）在同一个序列上协同移动，从而把一些看似需要嵌套循环的问题降为线性复杂度。

## 📌 什么时候用双指针

- **有序数组/有序链表**：经常可以用左右指针在 O(n) 内完成查找/计数。
- **原地修改**：如删除元素、去重、分区（partition）。
- **快慢指针**：链表环检测、找中点、滑动窗口中也会出现“不同速率的指针”。

## 🧠 常见类型

### 1) 左右指针（相向）

常见于“在有序数组中找两数之和”“反转数组”“回文判断”等。

```java
// 判断字符串是否回文（忽略大小写/非字母数字可自行扩展）
public boolean isPalindrome(String s) {
    int l = 0, r = s.length() - 1;
    while (l < r) {
        if (s.charAt(l) != s.charAt(r)) return false;
        l++; r--;
    }
    return true;
}
```

### 2) 快慢指针（同向不同速率）

- **用途**：找链表中点、判断链表是否有环、找环入口等。

```java
// 找数组中间位置（思想同“快慢指针”：fast 走两步，slow 走一步）
public int middleIndex(int[] nums) {
    int slow = 0, fast = 0;
    while (fast < nums.length && fast + 1 < nums.length) {
        slow++;
        fast += 2;
    }
    return slow;
}
```

### 3) 快慢指针（同向“写入/读取”）

常用于“原地删除/过滤”“去重”。

```java
// 删除有序数组重复项（返回新长度）
public int removeDuplicates(int[] nums) {
    if (nums.length == 0) return 0;
    int slow = 1;
    for (int fast = 1; fast < nums.length; fast++) {
        if (nums[fast] != nums[fast - 1]) {
            nums[slow++] = nums[fast];
        }
    }
    return slow;
}
```

## ✅ 模板：两数之和（有序数组）

```java
public int[] twoSumSorted(int[] nums, int target) {
    int l = 0, r = nums.length - 1;
    while (l < r) {
        int sum = nums[l] + nums[r];
        if (sum == target) return new int[] { l, r };
        if (sum < target) l++;
        else r--;
    }
    return new int[] { -1, -1 };
}
```

## 🎯 复杂度

- **时间复杂度**：通常 O(n)
- **空间复杂度**：通常 O(1)

## 💡 小技巧

- **先确认“单调性/有序性”**：相向指针能否移动，核心在于移动一端不会错过答案。
- **原地修改先想“写指针”**：`slow` 负责写入有效元素，`fast` 负责遍历。
