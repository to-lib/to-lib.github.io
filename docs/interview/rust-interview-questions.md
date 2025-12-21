---
sidebar_position: 17
title: 面试题集
---

# Rust 面试题集

本文整理了 Rust 技术面试中常见的问题，涵盖基础概念、进阶主题和实战编程题。

## 基础概念题

### 1. 解释 Rust 的所有权系统

**核心要点：**

- 每个值都有一个所有者
- 值在任何时刻只能有一个所有者
- 当所有者离开作用域，值将被丢弃

**意义：**

- 编译时保证内存安全
- 无需垃圾回收器
- 防止空指针、悬垂指针、数据竞争

**示例：**

```rust
let s1 = String::from("hello");
let s2 = s1;  // s1 所有权移动到 s2
// println!("{}", s1);  // 错误：s1 已失效
```

### 2. 借用规则是什么？

**规则：**

1. 可以有任意数量的不可变借用
2. 只能有一个可变借用
3. 可变借用和不可变借用不能同时存在

**原因：**
防止数据竞争和并发访问冲突。

```rust
let mut s = String::from("hello");

// 场景 1: 多个不可变借用
let r1 = &s;
let r2 = &s;  // OK

// 场景 2: 一个可变借用
let r1 = &mut s;
// let r2 = &mut s;  // 错误

// 场景 3: 不能混用
let r1 = &s;
// let r2 = &mut s;  // 错误
```

### 3. String 和 &str 的区别？

| 特性   | String | &str     |
| ------ | ------ | -------- |
| 所有权 | 拥有   | 借用     |
| 可变性 | 可变   | 不可变   |
| 内存   | 堆分配 | 栈或静态 |
| 大小   | 可变长 | 固定长   |

**使用建议：**

- 函数参数：`&str`（更灵活）
- 返回值：`String`（拥有所有权）
- 存储：`String`（可修改）

```rust
fn process(s: &str) -> String {
    s.to_uppercase()
}
```

### 4. 什么是生命周期？为什么需要标注？

**定义：** 引用有效的作用域范围。

**目的：** 确保引用始终有效，防止悬垂引用。

```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

**标注规则：**

- 返回值的生命周期必须与至少一个参数相关
- 编译器无法推断时需要显式标注

### 5. Copy 和 Clone 的区别？

**Copy:**

- 栈上的简单位复制
- 自动隐式复制
- 实现 Copy 的类型：整数、浮点、布尔、字符、元组（成员都是 Copy）

**Clone:**

- 显式深拷贝
- 需要调用 `.clone()`
- 可能涉及堆分配

```rust
// Copy
let x = 5;
let y = x;  // 隐式复制

// Clone
let s1 = String::from("hello");
let s2 = s1.clone();  // 显式克隆
```

## 进阶题目

### 6. 解释 Box、Rc、Arc 的使用场景

**`Box<T>`:**

- 单所有权，堆分配
- 递归类型、大对象

```rust
enum List {
    Cons(i32, Box<List>),
    Nil,
}
```

**`Rc<T>`:**

- 多所有权（单线程）
- 引用计数
- 只读共享

```rust
use std::rc::Rc;
let a = Rc::new(5);
let b = Rc::clone(&a);
```

**`Arc<T>`:**

- 多所有权（多线程）
- 原子引用计数
- 线程间共享

```rust
use std::sync::Arc;
use std::thread;

let data = Arc::new(vec![1, 2, 3]);
let data_clone = Arc::clone(&data);

thread::spawn(move || {
    println!("{:?}", data_clone);
});
```

### 7. RefCell 提供了什么？

**内部可变性：** 在不可变引用时修改值。

**运行时借用检查：** 违反借用规则会 panic。

```rust
use std::cell::RefCell;

let value = RefCell::new(5);
*value.borrow_mut() += 1;
```

**使用场景：**

- 与 Rc 结合实现多所有权可变数据
- Mock 对象
- 缓存

**Rc\u003cRefCell\u003cT\u003e\u003e 模式：**

```rust
use std::rc::Rc;
use std::cell::RefCell;

let shared = Rc::new(RefCell::new(5));
*shared.borrow_mut() += 1;
```

### 8. 解释闭包的三种 Trait

**FnOnce:**

- 消耗捕获的值
- 只能调用一次

**FnMut:**

- 可变借用捕获的值
- 可以多次调用

**Fn:**

- 不可变借用捕获的值
- 可以多次调用

```rust
// FnOnce
let s = String::from("hello");
let consume = || drop(s);

// FnMut
let mut count = 0;
let mut increment = || count += 1;

// Fn
let x = 5;
let print = || println!("{}", x);
```

**继承关系：** `Fn: FnMut: FnOnce`

### 9. Send 和 Sync 的区别？

**Send:**

- 类型可以安全地在线程间转移所有权
- 大多数类型都实现了 Send
- `Rc<T>` 没有实现 Send

**Sync:**

- 类型的引用可以安全地在线程间共享
- `&T` 是 Send，则 `T` 是 Sync
- `RefCell<T>` 没有实现 Sync

```rust
// Send
fn is_send<T: Send>() {}
is_send::<String>();

// Sync
fn is_sync<T: Sync>() {}
is_sync::<i32>();
```

### 10. 迭代器的优势是什么？

**零成本抽象：** 性能与手写循环相同。

**惰性求值：** 只在需要时计算。

**链式调用：** 代码更简洁。

```rust
let result: Vec<_> = vec![1, 2, 3, 4, 5]
    .iter()
    .filter(|&&x| x % 2 == 0)
    .map(|&x| x * 2)
    .collect();
```

## 实战编程题

### 11. 实现一个 LRU 缓存

```rust
use std::collections::HashMap;

struct LRUCache {
    capacity: usize,
    cache: HashMap<i32, i32>,
    order: Vec<i32>,
}

impl LRUCache {
    fn new(capacity: usize) -> Self {
        LRUCache {
            capacity,
            cache: HashMap::new(),
            order: Vec::new(),
        }
    }

    fn get(&mut self, key: i32) -> Option<i32> {
        if let Some(&value) = self.cache.get(&key) {
            // 更新访问顺序
            self.order.retain(|&k| k != key);
            self.order.push(key);
            Some(value)
        } else {
            None
        }
    }

    fn put(&mut self, key: i32, value: i32) {
        if self.cache.contains_key(&key) {
            self.order.retain(|&k| k != key);
        } else if self.cache.len() >= self.capacity {
            // 移除最久未使用
            if let Some(oldest) = self.order.first() {
                let oldest = *oldest;
                self.cache.remove(&oldest);
                self.order.remove(0);
            }
        }

        self.cache.insert(key, value);
        self.order.push(key);
    }
}
```

### 12. 实现一个线程安全的计数器

```rust
use std::sync::{Arc, Mutex};
use std::thread;

struct Counter {
    count: Arc<Mutex<i32>>,
}

impl Counter {
    fn new() -> Self {
        Counter {
            count: Arc::new(Mutex::new(0)),
        }
    }

    fn increment(&self) {
        let mut count = self.count.lock().unwrap();
        *count += 1;
    }

    fn value(&self) -> i32 {
        *self.count.lock().unwrap()
    }
}

impl Clone for Counter {
    fn clone(&self) -> Self {
        Counter {
            count: Arc::clone(&self.count),
        }
    }
}

fn main() {
    let counter = Counter::new();
    let mut handles = vec![];

    for _ in 0..10 {
        let counter_clone = counter.clone();
        let handle = thread::spawn(move || {
            for _ in 0..100 {
                counter_clone.increment();
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Counter: {}", counter.value());  // 1000
}
```

### 13. 实现一个简单的迭代器

```rust
struct Counter {
    count: u32,
    max: u32,
}

impl Counter {
    fn new(max: u32) -> Self {
        Counter { count: 0, max }
    }
}

impl Iterator for Counter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count < self.max {
            self.count += 1;
            Some(self.count)
        } else {
            None
        }
    }
}

fn main() {
    let counter = Counter::new(5);
    let sum: u32 = counter.sum();
    println!("{}", sum);  // 15
}
```

### 14. 实现一个递归类型（二叉树）

```rust
#[derive(Debug)]
enum TreeNode {
    Leaf(i32),
    Node(i32, Box<TreeNode>, Box<TreeNode>),
}

impl TreeNode {
    fn sum(&self) -> i32 {
        match self {
            TreeNode::Leaf(val) => *val,
            TreeNode::Node(val, left, right) => {
                val + left.sum() + right.sum()
            }
        }
    }
}

fn main() {
    let tree = TreeNode::Node(
        1,
        Box::new(TreeNode::Leaf(2)),
        Box::new(TreeNode::Leaf(3)),
    );

    println!("Sum: {}", tree.sum());  // 6
}
```

### 15. 异步函数调用

```rust
use tokio::time::{sleep, Duration};

async fn fetch_data(id: i32) -> String {
    sleep(Duration::from_secs(1)).await;
    format!("Data {}", id)
}

async fn process_data(data: String) -> String {
    sleep(Duration::from_secs(1)).await;
    format!("Processed: {}", data)
}

#[tokio::main]
async fn main() {
    let data = fetch_data(1).await;
    let result = process_data(data).await;
    println!("{}", result);
}
```

## 系统设计题

### 16. 设计一个 Web 服务器架构

**要点：**

1. 使用 Tokio 异步运行时
2. TCP 监听器接收连接
3. 为每个连接创建异步任务
4. 解析 HTTP 请求
5. 路由到处理函数
6. 返回 HTTP 响应

**关键考虑：**

- 并发连接数限制
- 错误处理
- 优雅关闭
- 日志记录

### 17. 设计一个任务调度系统

**要点：**

1. 使用优先队列存储任务
2. 定时器触发任务执行
3. 线程池执行任务
4. 任务状态跟踪

**关键考虑：**

- 任务优先级
- 任务重试机制
- 并发控制
- 资源限制

## 回答技巧

### 基础题回答框架

1. **定义概念** - 简洁准确
2. **解释原因** - 为什么需要这个特性
3. **举例说明** - 代码示例
4. **对比分析** - 与其他语言或方法对比
5. **应用场景** - 实际使用场景

### 编程题技巧

1. **理解需求** - 确认输入输出
2. **考虑边界** - 空输入、极端情况
3. **选择数据结构** - HashMap、Vec 等
4. **考虑性能** - 时间空间复杂度
5. **写测试** - 验证正确性

### 系统设计技巧

1. **需求分析** - 功能性和非功能性
2. **模块划分** - 职责清晰
3. **并发模型** - 线程、异步、通道
4. **错误处理** - 健壮性
5. **可扩展性** - 未来增长

## 常见追问

### "为什么选择 Rust？"

**要点：**

- 内存安全（无 GC）
- 并发安全（编译时检查）
- 零成本抽象
- 现代工具链（Cargo）
- 高性能

### "Rust 的劣势是什么？"

**诚实回答：**

- 学习曲线陡峭
- 编译时间较长
- 生态系统相对年轻
- 某些场景代码冗长

### "在项目中如何使用 Rust？"

**建议回答：**

- 具体项目经验
- 遇到的挑战和解决方案
- 性能提升或安全改进
- 团队协作经验

## 总结

面试准备建议：

1. **扎实基础** - 所有权、借用、生命周期
2. **熟悉标准库** - 集合、迭代器、错误处理
3. **实践经验** - 完成小项目
4. **阅读优秀代码** - 学习最佳实践
5. **模拟面试** - 练习表达能力

记住：面试不仅是技术考察，也是沟通能力的展示。清晰表达你的思路和权衡过程同样重要！
