---
sidebar_position: 17
title: 数据结构
---

# 前端常用数据结构

> [!TIP]
> 掌握基础数据结构能帮助你写出更高效的代码，也是面试必考内容。

## 📚 栈 (Stack)

后进先出 (LIFO)。

```javascript
class Stack {
  #items = [];

  push(item) {
    this.#items.push(item);
  }

  pop() {
    return this.#items.pop();
  }

  peek() {
    return this.#items[this.#items.length - 1];
  }

  isEmpty() {
    return this.#items.length === 0;
  }

  size() {
    return this.#items.length;
  }

  clear() {
    this.#items = [];
  }
}

// 使用
const stack = new Stack();
stack.push(1);
stack.push(2);
stack.pop(); // 2
stack.peek(); // 1
```

### 应用：括号匹配

```javascript
function isValidBrackets(str) {
  const stack = [];
  const pairs = { "(": ")", "[": "]", "{": "}" };

  for (const char of str) {
    if (char in pairs) {
      stack.push(char);
    } else if ([")", "]", "}"].includes(char)) {
      if (pairs[stack.pop()] !== char) {
        return false;
      }
    }
  }

  return stack.length === 0;
}

isValidBrackets("({[]})"); // true
isValidBrackets("([)]"); // false
```

## 📋 队列 (Queue)

先进先出 (FIFO)。

```javascript
class Queue {
  #items = [];

  enqueue(item) {
    this.#items.push(item);
  }

  dequeue() {
    return this.#items.shift();
  }

  front() {
    return this.#items[0];
  }

  isEmpty() {
    return this.#items.length === 0;
  }

  size() {
    return this.#items.length;
  }
}

// 使用
const queue = new Queue();
queue.enqueue("a");
queue.enqueue("b");
queue.dequeue(); // 'a'
queue.front(); // 'b'
```

### 应用：任务调度

```javascript
class TaskQueue {
  #queue = [];
  #running = false;

  add(task) {
    this.#queue.push(task);
    this.#run();
  }

  async #run() {
    if (this.#running) return;
    this.#running = true;

    while (this.#queue.length > 0) {
      const task = this.#queue.shift();
      await task();
    }

    this.#running = false;
  }
}
```

## 🔗 链表 (Linked List)

```javascript
class ListNode {
  constructor(value) {
    this.value = value;
    this.next = null;
  }
}

class LinkedList {
  constructor() {
    this.head = null;
    this.size = 0;
  }

  // 添加到末尾
  append(value) {
    const node = new ListNode(value);

    if (!this.head) {
      this.head = node;
    } else {
      let current = this.head;
      while (current.next) {
        current = current.next;
      }
      current.next = node;
    }

    this.size++;
  }

  // 插入到指定位置
  insert(index, value) {
    if (index < 0 || index > this.size) return false;

    const node = new ListNode(value);

    if (index === 0) {
      node.next = this.head;
      this.head = node;
    } else {
      let current = this.head;
      for (let i = 0; i < index - 1; i++) {
        current = current.next;
      }
      node.next = current.next;
      current.next = node;
    }

    this.size++;
    return true;
  }

  // 删除
  remove(index) {
    if (index < 0 || index >= this.size) return null;

    let removed;

    if (index === 0) {
      removed = this.head;
      this.head = this.head.next;
    } else {
      let current = this.head;
      for (let i = 0; i < index - 1; i++) {
        current = current.next;
      }
      removed = current.next;
      current.next = current.next.next;
    }

    this.size--;
    return removed.value;
  }

  // 转为数组
  toArray() {
    const result = [];
    let current = this.head;
    while (current) {
      result.push(current.value);
      current = current.next;
    }
    return result;
  }
}
```

### 链表反转

```javascript
function reverseList(head) {
  let prev = null;
  let current = head;

  while (current) {
    const next = current.next;
    current.next = prev;
    prev = current;
    current = next;
  }

  return prev;
}
```

## 🌳 二叉树 (Binary Tree)

```javascript
class TreeNode {
  constructor(value) {
    this.value = value;
    this.left = null;
    this.right = null;
  }
}

class BinaryTree {
  constructor() {
    this.root = null;
  }

  // 前序遍历（根-左-右）
  preorder(node = this.root, result = []) {
    if (node) {
      result.push(node.value);
      this.preorder(node.left, result);
      this.preorder(node.right, result);
    }
    return result;
  }

  // 中序遍历（左-根-右）
  inorder(node = this.root, result = []) {
    if (node) {
      this.inorder(node.left, result);
      result.push(node.value);
      this.inorder(node.right, result);
    }
    return result;
  }

  // 后序遍历（左-右-根）
  postorder(node = this.root, result = []) {
    if (node) {
      this.postorder(node.left, result);
      this.postorder(node.right, result);
      result.push(node.value);
    }
    return result;
  }

  // 层序遍历（BFS）
  levelOrder() {
    if (!this.root) return [];

    const result = [];
    const queue = [this.root];

    while (queue.length > 0) {
      const node = queue.shift();
      result.push(node.value);

      if (node.left) queue.push(node.left);
      if (node.right) queue.push(node.right);
    }

    return result;
  }

  // 最大深度
  maxDepth(node = this.root) {
    if (!node) return 0;
    return 1 + Math.max(this.maxDepth(node.left), this.maxDepth(node.right));
  }
}
```

## 🗺️ HashMap

```javascript
class HashMap {
  constructor(size = 53) {
    this.buckets = new Array(size);
    this.size = size;
  }

  #hash(key) {
    let total = 0;
    const PRIME = 31;

    for (let i = 0; i < Math.min(key.length, 100); i++) {
      const char = key[i];
      const value = char.charCodeAt(0) - 96;
      total = (total * PRIME + value) % this.size;
    }

    return total;
  }

  set(key, value) {
    const index = this.#hash(key);

    if (!this.buckets[index]) {
      this.buckets[index] = [];
    }

    const bucket = this.buckets[index];
    const existing = bucket.find(([k]) => k === key);

    if (existing) {
      existing[1] = value;
    } else {
      bucket.push([key, value]);
    }
  }

  get(key) {
    const index = this.#hash(key);
    const bucket = this.buckets[index];

    if (bucket) {
      const found = bucket.find(([k]) => k === key);
      if (found) return found[1];
    }

    return undefined;
  }

  has(key) {
    return this.get(key) !== undefined;
  }

  delete(key) {
    const index = this.#hash(key);
    const bucket = this.buckets[index];

    if (bucket) {
      const i = bucket.findIndex(([k]) => k === key);
      if (i !== -1) {
        bucket.splice(i, 1);
        return true;
      }
    }

    return false;
  }
}
```

## 📊 常见算法

### 二分查找

```javascript
function binarySearch(arr, target) {
  let left = 0;
  let right = arr.length - 1;

  while (left <= right) {
    const mid = Math.floor((left + right) / 2);

    if (arr[mid] === target) {
      return mid;
    } else if (arr[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  return -1;
}
```

### 快速排序

```javascript
function quickSort(arr) {
  if (arr.length <= 1) return arr;

  const pivot = arr[Math.floor(arr.length / 2)];
  const left = arr.filter((x) => x < pivot);
  const middle = arr.filter((x) => x === pivot);
  const right = arr.filter((x) => x > pivot);

  return [...quickSort(left), ...middle, ...quickSort(right)];
}
```

### 去重

```javascript
// 方法1: Set
const unique1 = (arr) => [...new Set(arr)];

// 方法2: filter
const unique2 = (arr) => arr.filter((v, i) => arr.indexOf(v) === i);

// 方法3: reduce
const unique3 = (arr) =>
  arr.reduce((acc, cur) => (acc.includes(cur) ? acc : [...acc, cur]), []);
```

## 💡 复杂度速查

| 结构    | 查找     | 插入     | 删除     |
| ------- | -------- | -------- | -------- |
| 数组    | O(n)     | O(n)     | O(n)     |
| 栈/队列 | O(n)     | O(1)     | O(1)     |
| 链表    | O(n)     | O(1)     | O(1)     |
| 哈希表  | O(1)     | O(1)     | O(1)     |
| 二叉树  | O(log n) | O(log n) | O(log n) |

## 🔗 相关资源

- [手写实现](/docs/frontend/javascript/implementations)
- [ES6+](/docs/frontend/javascript/es6)

---

**下一步**：学习 [移动端适配](/docs/frontend/css/mobile) 掌握移动端开发技巧。
