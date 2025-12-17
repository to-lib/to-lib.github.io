---
sidebar_position: 3
title: 数组与链表
---

# 数组与链表

数组和链表是最基础的线性数据结构，是学习其他数据结构的基石。

## 📖 数组 (Array)

数组是**连续内存**存储的相同类型元素集合，支持**随机访问**。

### 特点

- ✅ 随机访问 O(1)
- ❌ 插入/删除 O(n)
- ❌ 大小固定（静态数组）

### Java 数组操作

```java
// 声明和初始化
int[] arr = new int[5];
int[] arr2 = {1, 2, 3, 4, 5};
int[] arr3 = new int[]{1, 2, 3};

// 访问元素
int element = arr[0];  // O(1)

// 遍历
for (int i = 0; i < arr.length; i++) {
    System.out.println(arr[i]);
}

// 增强 for 循环
for (int num : arr) {
    System.out.println(num);
}
```

### 动态数组 ArrayList

```java
import java.util.ArrayList;
import java.util.List;

// 创建
List<Integer> list = new ArrayList<>();

// 添加元素 - 摊销 O(1)
list.add(1);
list.add(2);
list.add(0, 0);  // 指定位置插入 O(n)

// 访问 - O(1)
int val = list.get(0);

// 修改 - O(1)
list.set(0, 10);

// 删除 - O(n)
list.remove(0);          // 按索引
list.remove(Integer.valueOf(2));  // 按值

// 大小
int size = list.size();

// 是否包含 - O(n)
boolean contains = list.contains(1);
```

### 数组常见算法

#### 双指针技巧

```java
// 反转数组
public void reverse(int[] arr) {
    int left = 0, right = arr.length - 1;
    while (left < right) {
        int temp = arr[left];
        arr[left] = arr[right];
        arr[right] = temp;
        left++;
        right--;
    }
}

// 移除元素（原地）
public int removeElement(int[] nums, int val) {
    int slow = 0;
    for (int fast = 0; fast < nums.length; fast++) {
        if (nums[fast] != val) {
            nums[slow++] = nums[fast];
        }
    }
    return slow;
}
```

#### 滑动窗口

```java
// 最大子数组和（定长窗口）
public int maxSum(int[] arr, int k) {
    int windowSum = 0, maxSum = Integer.MIN_VALUE;

    for (int i = 0; i < arr.length; i++) {
        windowSum += arr[i];

        if (i >= k - 1) {
            maxSum = Math.max(maxSum, windowSum);
            windowSum -= arr[i - k + 1];
        }
    }
    return maxSum;
}
```

## 🔗 链表 (Linked List)

链表是**非连续内存**存储，通过指针连接的动态数据结构。

### 链表类型

| 类型     | 特点     | 适用场景     |
| -------- | -------- | ------------ |
| 单链表   | 单向遍历 | 简单场景     |
| 双链表   | 双向遍历 | 频繁前后操作 |
| 循环链表 | 首尾相连 | 环形结构     |

### 单链表实现

```java
// 节点定义
public class ListNode {
    int val;
    ListNode next;

    public ListNode(int val) {
        this.val = val;
        this.next = null;
    }
}

// 单链表类
public class LinkedList {
    private ListNode head;
    private int size;

    public LinkedList() {
        this.head = null;
        this.size = 0;
    }

    // 头部插入 O(1)
    public void addFirst(int val) {
        ListNode newNode = new ListNode(val);
        newNode.next = head;
        head = newNode;
        size++;
    }

    // 尾部插入 O(n)
    public void addLast(int val) {
        ListNode newNode = new ListNode(val);
        if (head == null) {
            head = newNode;
        } else {
            ListNode curr = head;
            while (curr.next != null) {
                curr = curr.next;
            }
            curr.next = newNode;
        }
        size++;
    }

    // 删除头节点 O(1)
    public int removeFirst() {
        if (head == null) throw new RuntimeException("链表为空");
        int val = head.val;
        head = head.next;
        size--;
        return val;
    }

    // 查找 O(n)
    public boolean contains(int val) {
        ListNode curr = head;
        while (curr != null) {
            if (curr.val == val) return true;
            curr = curr.next;
        }
        return false;
    }

    // 获取链表长度
    public int size() {
        return size;
    }
}
```

### 双链表实现

```java
public class DoublyListNode {
    int val;
    DoublyListNode prev;
    DoublyListNode next;

    public DoublyListNode(int val) {
        this.val = val;
    }
}

public class DoublyLinkedList {
    private DoublyListNode head;
    private DoublyListNode tail;
    private int size;

    public DoublyLinkedList() {
        // 使用哨兵节点简化操作
        head = new DoublyListNode(0);
        tail = new DoublyListNode(0);
        head.next = tail;
        tail.prev = head;
        size = 0;
    }

    // 头部插入 O(1)
    public void addFirst(int val) {
        DoublyListNode newNode = new DoublyListNode(val);
        newNode.next = head.next;
        newNode.prev = head;
        head.next.prev = newNode;
        head.next = newNode;
        size++;
    }

    // 尾部插入 O(1)
    public void addLast(int val) {
        DoublyListNode newNode = new DoublyListNode(val);
        newNode.prev = tail.prev;
        newNode.next = tail;
        tail.prev.next = newNode;
        tail.prev = newNode;
        size++;
    }

    // 删除指定节点 O(1)
    public void remove(DoublyListNode node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
        size--;
    }
}
```

### Java LinkedList

```java
import java.util.LinkedList;

LinkedList<Integer> list = new LinkedList<>();

// 添加操作
list.addFirst(1);   // 头部添加 O(1)
list.addLast(2);    // 尾部添加 O(1)
list.add(1, 3);     // 指定位置 O(n)

// 获取操作
int first = list.getFirst();  // O(1)
int last = list.getLast();    // O(1)
int val = list.get(1);        // O(n)

// 删除操作
list.removeFirst();  // O(1)
list.removeLast();   // O(1)
list.remove(1);      // O(n)

// 作为队列使用
list.offer(1);   // 入队
list.poll();     // 出队

// 作为栈使用
list.push(1);    // 入栈
list.pop();      // 出栈
```

## 🎯 经典链表算法

### 反转链表

```java
// 迭代法 O(n) 时间, O(1) 空间
public ListNode reverseList(ListNode head) {
    ListNode prev = null;
    ListNode curr = head;

    while (curr != null) {
        ListNode next = curr.next;
        curr.next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}

// 递归法 O(n) 时间, O(n) 空间
public ListNode reverseListRecursive(ListNode head) {
    if (head == null || head.next == null) {
        return head;
    }
    ListNode newHead = reverseListRecursive(head.next);
    head.next.next = head;
    head.next = null;
    return newHead;
}
```

### 检测环

```java
// 快慢指针法
public boolean hasCycle(ListNode head) {
    if (head == null || head.next == null) return false;

    ListNode slow = head;
    ListNode fast = head;

    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
        if (slow == fast) return true;
    }
    return false;
}

// 找环入口
public ListNode detectCycle(ListNode head) {
    ListNode slow = head, fast = head;

    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;

        if (slow == fast) {
            ListNode ptr = head;
            while (ptr != slow) {
                ptr = ptr.next;
                slow = slow.next;
            }
            return ptr;
        }
    }
    return null;
}
```

### 合并两个有序链表

```java
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    ListNode dummy = new ListNode(0);
    ListNode curr = dummy;

    while (l1 != null && l2 != null) {
        if (l1.val <= l2.val) {
            curr.next = l1;
            l1 = l1.next;
        } else {
            curr.next = l2;
            l2 = l2.next;
        }
        curr = curr.next;
    }

    curr.next = (l1 != null) ? l1 : l2;
    return dummy.next;
}
```

### 找中间节点

```java
public ListNode middleNode(ListNode head) {
    ListNode slow = head, fast = head;

    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
    }
    return slow;
}
```

## 📊 数组 vs 链表

| 操作     | 数组       | 链表             |
| -------- | ---------- | ---------------- |
| 随机访问 | O(1) ✅    | O(n)             |
| 头部插入 | O(n)       | O(1) ✅          |
| 尾部插入 | O(1)\*     | O(1)\*\*         |
| 中间插入 | O(n)       | O(1)\*\*\*       |
| 查找元素 | O(n)       | O(n)             |
| 内存使用 | 连续、紧凑 | 分散、有指针开销 |

> \*摊销复杂度  
> **需要尾指针  
> \***已知位置情况下

## 💡 选择建议

使用**数组**当：

- 需要频繁随机访问
- 数据量相对固定
- 内存空间紧张

使用**链表**当：

- 需要频繁插入/删除
- 不确定数据量
- 需要实现其他数据结构（栈、队列）
