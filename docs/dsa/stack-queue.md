---
sidebar_position: 4
title: 栈与队列
---

# 栈与队列

栈和队列是两种特殊的线性数据结构，限制了元素的访问方式。

## 📚 栈 (Stack)

栈是**后进先出 (LIFO)** 的数据结构，只能在栈顶进行操作。

### 基本操作

| 操作 | 描述     | 时间复杂度 |
| ---- | -------- | ---------- |
| push | 入栈     | O(1)       |
| pop  | 出栈     | O(1)       |
| peek | 查看栈顶 | O(1)       |

### 数组实现栈

```java
public class ArrayStack {
    private int[] data;
    private int top = -1;

    public ArrayStack(int capacity) {
        this.data = new int[capacity];
    }

    public void push(int val) {
        data[++top] = val;
    }

    public int pop() {
        return data[top--];
    }

    public int peek() {
        return data[top];
    }

    public boolean isEmpty() {
        return top == -1;
    }
}
```

### Java 栈的使用

```java
import java.util.Deque;
import java.util.ArrayDeque;

// 推荐使用 Deque
Deque<Integer> stack = new ArrayDeque<>();
stack.push(1);
stack.pop();
stack.peek();
```

### 栈的应用 - 括号匹配

```java
public boolean isValid(String s) {
    Deque<Character> stack = new ArrayDeque<>();
    for (char c : s.toCharArray()) {
        if (c == '(' || c == '[' || c == '{') {
            stack.push(c);
        } else {
            if (stack.isEmpty()) return false;
            char top = stack.pop();
            if (c == ')' && top != '(') return false;
            if (c == ']' && top != '[') return false;
            if (c == '}' && top != '{') return false;
        }
    }
    return stack.isEmpty();
}
```

### 单调栈 - 下一个更大元素

```java
public int[] nextGreaterElement(int[] nums) {
    int n = nums.length;
    int[] result = new int[n];
    Deque<Integer> stack = new ArrayDeque<>();

    for (int i = n - 1; i >= 0; i--) {
        while (!stack.isEmpty() && stack.peek() <= nums[i]) {
            stack.pop();
        }
        result[i] = stack.isEmpty() ? -1 : stack.peek();
        stack.push(nums[i]);
    }
    return result;
}
```

## 📬 队列 (Queue)

队列是**先进先出 (FIFO)** 的数据结构。

### Java 队列的使用

```java
import java.util.Queue;
import java.util.ArrayDeque;

Queue<Integer> queue = new ArrayDeque<>();
queue.offer(1);   // 入队
queue.poll();     // 出队
queue.peek();     // 查看队首
```

### 循环队列

```java
public class CircularQueue {
    private int[] data;
    private int front = 0, rear = 0, size = 0;
    private int capacity;

    public CircularQueue(int k) {
        capacity = k;
        data = new int[k];
    }

    public boolean enqueue(int val) {
        if (size == capacity) return false;
        data[rear] = val;
        rear = (rear + 1) % capacity;
        size++;
        return true;
    }

    public int dequeue() {
        int val = data[front];
        front = (front + 1) % capacity;
        size--;
        return val;
    }
}
```

## 🔄 双端队列 (Deque)

```java
Deque<Integer> deque = new ArrayDeque<>();
deque.addFirst(1);    // 头部插入
deque.addLast(2);     // 尾部插入
deque.removeFirst();  // 头部删除
deque.removeLast();   // 尾部删除
```

### 滑动窗口最大值

```java
public int[] maxSlidingWindow(int[] nums, int k) {
    int n = nums.length;
    int[] result = new int[n - k + 1];
    Deque<Integer> deque = new ArrayDeque<>();

    for (int i = 0; i < n; i++) {
        while (!deque.isEmpty() && deque.peekFirst() < i - k + 1)
            deque.pollFirst();
        while (!deque.isEmpty() && nums[deque.peekLast()] < nums[i])
            deque.pollLast();
        deque.offerLast(i);
        if (i >= k - 1) result[i - k + 1] = nums[deque.peekFirst()];
    }
    return result;
}
```

## ⚡ 优先队列

```java
// 最小堆
PriorityQueue<Integer> minHeap = new PriorityQueue<>();

// 最大堆
PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a, b) -> b - a);
```

## 🎯 用栈实现队列

```java
class MyQueue {
    private Deque<Integer> inStack = new ArrayDeque<>();
    private Deque<Integer> outStack = new ArrayDeque<>();

    public void push(int x) { inStack.push(x); }

    public int pop() {
        if (outStack.isEmpty()) transfer();
        return outStack.pop();
    }

    private void transfer() {
        while (!inStack.isEmpty()) outStack.push(inStack.pop());
    }
}
```
