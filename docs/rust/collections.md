---
sidebar_position: 5
title: 集合类型
---

# 集合类型

Rust 标准库提供了丰富的集合类型，用于存储多个值。本文介绍常用的集合类型及其使用场景。

## `Vec<T>` - 向量

Vec 是可变长度的数组，存储在堆上。

### 创建 Vec

```rust
fn main() {
    // 创建空 Vec
    let v1: Vec<i32> = Vec::new();

    // 使用 vec! 宏
    let v2 = vec![1, 2, 3];

    // 预分配容量
    let mut v3 = Vec::with_capacity(10);

    // 从迭代器创建
    let v4: Vec<_> = (0..5).collect();
}
```

### 添加元素

```rust
fn main() {
    let mut v = Vec::new();

    v.push(1);
    v.push(2);
    v.push(3);

    println!("{:?}", v);  // [1, 2, 3]
}
```

### 访问元素

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];

    // 索引访问（可能 panic）
    let third = v[2];

    // get 方法（返回 Option）
    match v.get(2) {
        Some(third) => println!("第三个元素: {}", third),
        None => println!("没有第三个元素"),
    }
}
```

### 遍历元素

```rust
fn main() {
    let v = vec![1, 2, 3];

    // 不可变引用
    for i in &v {
        println!("{}", i);
    }

    // 可变引用
    let mut v = vec![1, 2, 3];
    for i in &mut v {
        *i += 50;
    }

    // 获取所有权
    for i in v {
        println!("{}", i);
    }
    // v 已失效
}
```

### 常用方法

```rust
fn main() {
    let mut v = vec![1, 2, 3, 4, 5];

    // 长度和容量
    println!("长度: {}", v.len());
    println!("容量: {}", v.capacity());

    // 删除元素
    v.pop();  // 删除最后一个
    v.remove(0);  // 删除索引 0

    // 清空
    v.clear();

    // 判断是否为空
    println!("是否为空: {}", v.is_empty());

    // 插入元素
    let mut v = vec![1, 3, 4];
    v.insert(1, 2);  // 在索引 1 插入 2

    // 追加另一个 Vec
    let mut v1 = vec![1, 2];
    let mut v2 = vec![3, 4];
    v1.append(&mut v2);
}
```

### 存储不同类型

```rust
enum SpreadsheetCell {
    Int(i32),
    Float(f64),
    Text(String),
}

fn main() {
    let row = vec![
        SpreadsheetCell::Int(3),
        SpreadsheetCell::Text(String::from("blue")),
        SpreadsheetCell::Float(10.12),
    ];
}
```

## String - 字符串

### String vs &str

```rust
fn main() {
    // String：可变、堆分配、拥有所有权
    let mut s = String::from("hello");
    s.push_str(", world");

    // &str：不可变、引用、字符串切片
    let s: &str = "hello";
}
```

### 创建 String

```rust
fn main() {
    // 从字符串字面量
    let s1 = String::from("初始内容");
    let s2 = "初始内容".to_string();

    // 创建空字符串
    let s3 = String::new();

    // 预分配容量
    let s4 = String::with_capacity(10);
}
```

### 更新字符串

```rust
fn main() {
    let mut s = String::from("foo");

    // push_str 添加字符串切片
    s.push_str("bar");

    // push 添加单个字符
    s.push('!');

    println!("{}", s);  // "foobar!"
}
```

### 拼接字符串

```rust
fn main() {
    // + 运算符
    let s1 = String::from("Hello, ");
    let s2 = String::from("world!");
    let s3 = s1 + &s2;  // s1 被移动，不能再使用

    // format! 宏（推荐）
    let s1 = String::from("tic");
    let s2 = String::from("tac");
    let s3 = String::from("toe");
    let s = format!("{}-{}-{}", s1, s2, s3);
}
```

### 索引字符串

```rust
fn main() {
    let s = String::from("hello");

    // 错误：不能直接索引
    // let h = s[0];

    // 使用切片（按字节）
    let hello = "你好";
    let s = &hello[0..3];  // 一个中文字符是 3 字节

    // 遍历字符
    for c in "你好".chars() {
        println!("{}", c);
    }

    // 遍历字节
    for b in "你好".bytes() {
        println!("{}", b);
    }
}
```

### 常用方法

```rust
fn main() {
    let s = String::from("  hello world  ");

    // 去除空白
    let trimmed = s.trim();

    // 分割
    for word in s.split_whitespace() {
        println!("{}", word);
    }

    // 替换
    let s = s.replace("world", "Rust");

    // 判断
    println!("包含 'hello': {}", s.contains("hello"));
    println!("以 'hello' 开头: {}", s.starts_with("hello"));
    println!("以 'world' 结尾: {}", s.ends_with("world"));

    // 转换大小写
    let upper = s.to_uppercase();
    let lower = s.to_lowercase();
}
```

## `HashMap<K, V>`

HashMap 存储键值对。

### 创建 HashMap

```rust
use std::collections::HashMap;

fn main() {
    // 创建空 HashMap
    let mut scores = HashMap::new();
    scores.insert(String::from("Blue"), 10);
    scores.insert(String::from("Yellow"), 50);

    // 从元组向量创建
    let teams = vec![String::from("Blue"), String::from("Yellow")];
    let initial_scores = vec![10, 50];
    let scores: HashMap<_, _> = teams.iter()
        .zip(initial_scores.iter())
        .collect();
}
```

### 访问值

```rust
use std::collections::HashMap;

fn main() {
    let mut scores = HashMap::new();
    scores.insert(String::from("Blue"), 10);

    // get 返回 Option<&V>
    let team = String::from("Blue");
    let score = scores.get(&team);

    match score {
        Some(&score) => println!("分数: {}", score),
        None => println!("队伍不存在"),
    }
}
```

### 遍历

```rust
use std::collections::HashMap;

fn main() {
    let mut scores = HashMap::new();
    scores.insert(String::from("Blue"), 10);
    scores.insert(String::from("Yellow"), 50);

    for (key, value) in &scores {
        println!("{}: {}", key, value);
    }
}
```

### 更新值

```rust
use std::collections::HashMap;

fn main() {
    let mut scores = HashMap::new();

    // 覆盖值
    scores.insert(String::from("Blue"), 10);
    scores.insert(String::from("Blue"), 25);

    // 只在键不存在时插入
    scores.entry(String::from("Yellow")).or_insert(50);
    scores.entry(String::from("Blue")).or_insert(50);  // 不会覆盖

    // 根据旧值更新
    let text = "hello world wonderful world";
    let mut map = HashMap::new();

    for word in text.split_whitespace() {
        let count = map.entry(word).or_insert(0);
        *count += 1;
    }

    println!("{:?}", map);
}
```

### 所有权

```rust
use std::collections::HashMap;

fn main() {
    let field_name = String::from("Favorite color");
    let field_value = String::from("Blue");

    let mut map = HashMap::new();
    // 所有权被移动到 map
    map.insert(field_name, field_value);

    // field_name 和 field_value 不再有效
}
```

## `HashSet<T>`

HashSet 是值的集合，不允许重复。

### 基本使用

```rust
use std::collections::HashSet;

fn main() {
    let mut set = HashSet::new();

    set.insert(1);
    set.insert(2);
    set.insert(2);  // 重复值不会被插入

    println!("{:?}", set);  // {1, 2}

    // 判断是否包含
    println!("包含 1: {}", set.contains(&1));

    // 删除
    set.remove(&1);
}
```

### 集合运算

```rust
use std::collections::HashSet;

fn main() {
    let a: HashSet<_> = [1, 2, 3].iter().cloned().collect();
    let b: HashSet<_> = [2, 3, 4].iter().cloned().collect();

    // 并集
    let union: HashSet<_> = a.union(&b).collect();

    // 交集
    let intersection: HashSet<_> = a.intersection(&b).collect();

    // 差集
    let diff: HashSet<_> = a.difference(&b).collect();

    // 对称差集
    let sym_diff: HashSet<_> = a.symmetric_difference(&b).collect();
}
```

## `VecDeque<T>` - 双端队列

```rust
use std::collections::VecDeque;

fn main() {
    let mut deque = VecDeque::new();

    // 从前端添加
    deque.push_front(1);
    deque.push_front(2);

    // 从后端添加
    deque.push_back(3);
    deque.push_back(4);

    println!("{:?}", deque);  // [2, 1, 3, 4]

    // 从前端弹出
    let front = deque.pop_front();

    // 从后端弹出
    let back = deque.pop_back();
}
```

## BTreeMap / BTreeSet

BTreeMap 和 BTreeSet 是有序的集合类型。

```rust
use std::collections::BTreeMap;

fn main() {
    let mut map = BTreeMap::new();

    map.insert(3, "c");
    map.insert(1, "a");
    map.insert(2, "b");

    // 按键排序
    for (key, value) in &map {
        println!("{}: {}", key, value);
    }
    // 输出: 1: a, 2: b, 3: c
}
```

## `BinaryHeap<T>` - 二叉堆

```rust
use std::collections::BinaryHeap;

fn main() {
    let mut heap = BinaryHeap::new();

    heap.push(3);
    heap.push(1);
    heap.push(4);
    heap.push(1);
    heap.push(5);

    // 最大堆，弹出最大值
    while let Some(value) = heap.pop() {
        println!("{}", value);
    }
    // 输出: 5, 4, 3, 1, 1
}
```

## 集合选择指南

### 何时使用 Vec

- ✅ 需要按索引访问
- ✅ 主要在末尾添加/删除元素
- ✅ 需要保持插入顺序

### 何时使用 VecDeque

- ✅ 需要在两端添加/删除元素
- ✅ 实现队列或栈

### 何时使用 HashMap

- ✅ 需要键值对映射
- ✅ 快速查找
- ✅ 不需要排序

### 何时使用 BTreeMap

- ✅ 需要有序的键值对
- ✅ 需要范围查询

### 何时使用 HashSet

- ✅ 需要唯一值集合
- ✅ 快速查找
- ✅ 集合运算

### 何时使用 BTreeSet

- ✅ 需要有序的唯一值
- ✅ 范围操作

## 性能特点

| 集合     | 插入      | 删除      | 查找      | 排序 |
| -------- | --------- | --------- | --------- | ---- |
| Vec      | O(1) 末尾 | O(1) 末尾 | O(1) 索引 | ❌   |
| VecDeque | O(1) 两端 | O(1) 两端 | O(1) 索引 | ❌   |
| HashMap  | O(1) 平均 | O(1) 平均 | O(1) 平均 | ❌   |
| BTreeMap | O(log n)  | O(log n)  | O(log n)  | ✅   |
| HashSet  | O(1) 平均 | O(1) 平均 | O(1) 平均 | ❌   |
| BTreeSet | O(log n)  | O(log n)  | O(log n)  | ✅   |

## 最佳实践

### 1. 预分配容量

```rust
// 好：预分配
let mut v = Vec::with_capacity(1000);
for i in 0..1000 {
    v.push(i);
}

// 不够好：多次重新分配
let mut v = Vec::new();
for i in 0..1000 {
    v.push(i);
}
```

### 2. 使用 Entry API

```rust
use std::collections::HashMap;

// 好：使用 entry
let mut map = HashMap::new();
map.entry("key").or_insert(0);

// 不够简洁
let mut map = HashMap::new();
if !map.contains_key("key") {
    map.insert("key", 0);
}
```

### 3. 选择合适的集合类型

```rust
// 需要唯一值 -> HashSet
let unique: HashSet<_> = vec![1, 2, 2, 3].into_iter().collect();

// 需要计数 -> HashMap
let mut counts = HashMap::new();
for item in items {
    *counts.entry(item).or_insert(0) += 1;
}
```

### 4. 避免不必要的克隆

```rust
// 不好：克隆整个 Vec
fn bad(v: &Vec<i32>) -> Vec<i32> {
    v.clone()
}

// 好：返回引用或使用 Cow
fn good(v: &[i32]) -> &[i32] {
    v
}
```

## 总结

- ✅ Vec 是最常用的集合类型
- ✅ String 处理 UTF-8 文本
- ✅ HashMap 提供快速的键值查找
- ✅ HashSet 用于唯一值集合
- ✅ 根据需求选择合适的集合类型
- ✅ 预分配容量提升性能
- ✅ 使用迭代器处理集合数据

掌握集合类型，是编写高效 Rust 程序的基础！
