---
sidebar_position: 8
title: 闭包和迭代器
---

# 闭包和迭代器

闭包和迭代器是 Rust 函数式编程的核心特性，它们提供了强大而优雅的代码抽象能力。

## 闭包 (Closures)

闭包是可以捕获其环境的匿名函数。

### 闭包语法

```rust
fn main() {
    // 基本语法
    let add_one = |x| x + 1;
    println!("{}", add_one(5));  // 6

    // 带类型注解
    let add_one: fn(i32) -> i32 = |x: i32| -> i32 { x + 1 };

    // 多参数
    let add = |x, y| x + y;
    println!("{}", add(1, 2));  // 3

    // 多行闭包
    let complex = |x| {
        let y = x + 1;
        y * 2
    };
}
```

### 类型推断

```rust
fn main() {
    // Rust 会推断闭包的类型
    let example_closure = |x| x;

    let s = example_closure(String::from("hello"));
    // let n = example_closure(5);  // 错误：类型已确定为 String
}
```

### 捕获环境

#### 不可变借用

```rust
fn main() {
    let list = vec![1, 2, 3];

    // 闭包不可变借用 list
    let only_borrows = || println!("从闭包访问: {:?}", list);

    println!("调用闭包前: {:?}", list);
    only_borrows();
    println!("调用闭包后: {:?}", list);
}
```

#### 可变借用

```rust
fn main() {
    let mut list = vec![1, 2, 3];

    // 闭包可变借用 list
    let mut borrows_mutably = || list.push(7);

    // println!("{:?}", list);  // 错误：已有可变借用
    borrows_mutably();
    println!("调用后: {:?}", list);
}
```

#### 获取所有权 (move)

```rust
use std::thread;

fn main() {
    let list = vec![1, 2, 3];

    // move 关键字强制获取所有权
    thread::spawn(move || {
        println!("从线程: {:?}", list);
    }).join().unwrap();

    // println!("{:?}", list);  // 错误：所有权已移动
}
```

## 闭包 Trait

Rust 的闭包实现了以下 trait 之一：

### FnOnce

只能调用一次，会获取环境的所有权。

```rust
fn consume<F>(func: F)
where
    F: FnOnce() -> String,
{
    println!("{}", func());
    // func();  // 错误：FnOnce 只能调用一次
}

fn main() {
    let s = String::from("hello");

    let closure = || {
        s  // 获取所有权
    };

    consume(closure);
}
```

### FnMut

可以多次调用，可变借用环境。

```rust
fn do_twice<F>(mut func: F)
where
    F: FnMut(),
{
    func();
    func();
}

fn main() {
    let mut counter = 0;

    let mut increment = || {
        counter += 1;
        println!("计数: {}", counter);
    };

    do_twice(increment);
}
```

### Fn

可以多次调用，不可变借用环境。

```rust
fn call_multiple<F>(func: F)
where
    F: Fn(i32) -> i32,
{
    println!("{}", func(1));
    println!("{}", func(2));
}

fn main() {
    let multiplier = 2;

    let multiply = |x| x * multiplier;

    call_multiple(multiply);
}
```

### Trait 继承关系

```rust
// Fn: FnMut: FnOnce
// 所有闭包都实现 FnOnce
// 不会移动捕获变量的闭包实现 FnMut
// 不需要可变访问的闭包实现 Fn

fn main() {
    let s = String::from("hello");

    // Fn
    let f1 = || println!("{}", s);

    let mut count = 0;
    // FnMut
    let mut f2 = || count += 1;

    let s2 = String::from("world");
    // FnOnce
    let f3 = || {
        drop(s2);  // 消耗 s2
    };
}
```

## 迭代器 (Iterators)

迭代器允许对元素序列进行处理。

### Iterator Trait

```rust
pub trait Iterator {
    type Item;

    fn next(&mut self) -> Option<Self::Item>;

    // 提供了很多默认方法...
}
```

### 创建迭代器

```rust
fn main() {
    let v = vec![1, 2, 3];

    // iter(): 不可变引用
    let iter1 = v.iter();

    // iter_mut(): 可变引用
    let mut v2 = vec![1, 2, 3];
    let iter2 = v2.iter_mut();

    // into_iter(): 获取所有权
    let iter3 = v.into_iter();
}
```

### 使用迭代器

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];

    // for 循环自动调用 into_iter()
    for val in &v {
        println!("{}", val);
    }

    // 手动使用 next()
    let mut iter = v.iter();
    assert_eq!(iter.next(), Some(&1));
    assert_eq!(iter.next(), Some(&2));
    assert_eq!(iter.next(), Some(&3));
}
```

## 迭代器适配器

迭代器适配器是惰性的，只有在调用消费者方法时才会执行。

### map - 转换

```rust
fn main() {
    let v = vec![1, 2, 3];

    // map 是惰性的
    let iter = v.iter().map(|x| x + 1);

    // collect 触发执行
    let v2: Vec<_> = iter.collect();
    println!("{:?}", v2);  // [2, 3, 4]
}
```

### filter - 过滤

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5, 6];

    let evens: Vec<_> = v.iter()
        .filter(|x| *x % 2 == 0)
        .collect();

    println!("{:?}", evens);  // [2, 4, 6]
}
```

### 链式调用

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5, 6];

    let result: Vec<_> = v.iter()
        .filter(|x| *x % 2 == 0)  // 过滤偶数
        .map(|x| x * 2)            // 乘以 2
        .collect();

    println!("{:?}", result);  // [4, 8, 12]
}
```

### enumerate - 索引

```rust
fn main() {
    let v = vec!["a", "b", "c"];

    for (index, value) in v.iter().enumerate() {
        println!("{}: {}", index, value);
    }
}
```

### zip - 组合

```rust
fn main() {
    let names = vec!["Alice", "Bob", "Carol"];
    let ages = vec![25, 30, 35];

    for (name, age) in names.iter().zip(ages.iter()) {
        println!("{} is {} years old", name, age);
    }
}
```

### take - 获取前 n 个

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];

    let first_three: Vec<_> = v.iter()
        .take(3)
        .collect();

    println!("{:?}", first_three);  // [1, 2, 3]
}
```

### skip - 跳过前 n 个

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];

    let after_two: Vec<_> = v.iter()
        .skip(2)
        .collect();

    println!("{:?}", after_two);  // [3, 4, 5]
}
```

## 消费适配器

消费适配器会消耗迭代器。

### sum - 求和

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];

    let total: i32 = v.iter().sum();
    println!("{}", total);  // 15
}
```

### collect - 收集

```rust
fn main() {
    let v = vec![1, 2, 3];

    // 收集为 Vec
    let v2: Vec<_> = v.iter().map(|x| x * 2).collect();

    // 收集为 HashMap
    use std::collections::HashMap;
    let map: HashMap<_, _> = v.iter()
        .enumerate()
        .collect();
}
```

### fold - 折叠

```rust
fn main() {
    let v = vec![1, 2, 3, 4];

    // fold(初始值, 累积函数)
    let sum = v.iter().fold(0, |acc, x| acc + x);
    println!("{}", sum);  // 10

    // 计算乘积
    let product = v.iter().fold(1, |acc, x| acc * x);
    println!("{}", product);  // 24
}
```

### any / all - 判断

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];

    // 是否有偶数
    let has_even = v.iter().any(|x| x % 2 == 0);
    println!("{}", has_even);  // true

    // 是否全是正数
    let all_positive = v.iter().all(|x| *x > 0);
    println!("{}", all_positive);  // true
}
```

### find - 查找

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];

    let first_even = v.iter().find(|x| *x % 2 == 0);
    println!("{:?}", first_even);  // Some(2)
}
```

### position - 位置

```rust
fn main() {
    let v = vec![10, 20, 30, 40];

    let pos = v.iter().position(|x| *x == 30);
    println!("{:?}", pos);  // Some(2)
}
```

## 自定义迭代器

```rust
struct Counter {
    count: u32,
}

impl Counter {
    fn new() -> Counter {
        Counter { count: 0 }
    }
}

impl Iterator for Counter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count < 5 {
            self.count += 1;
            Some(self.count)
        } else {
            None
        }
    }
}

fn main() {
    let mut counter = Counter::new();

    assert_eq!(counter.next(), Some(1));
    assert_eq!(counter.next(), Some(2));

    // 使用迭代器方法
    let sum: u32 = Counter::new().sum();
    println!("{}", sum);  // 15
}
```

## 闭包与迭代器结合

### 实战示例 1: 数据处理

```rust
#[derive(Debug)]
struct Product {
    name: String,
    price: f64,
    stock: u32,
}

fn main() {
    let products = vec![
        Product { name: "笔记本".to_string(), price: 5000.0, stock: 10 },
        Product { name: "鼠标".to_string(), price: 100.0, stock: 50 },
        Product { name: "键盘".to_string(), price: 300.0, stock: 0 },
        Product { name: "显示器".to_string(), price: 2000.0, stock: 5 },
    ];

    // 找出有库存且价格超过 1000 的产品
    let expensive_in_stock: Vec<_> = products.iter()
        .filter(|p| p.stock > 0)
        .filter(|p| p.price > 1000.0)
        .collect();

    for product in expensive_in_stock {
        println!("{}: {} 元", product.name, product.price);
    }

    // 计算总库存价值
    let total_value: f64 = products.iter()
        .map(|p| p.price * p.stock as f64)
        .sum();

    println!("总库存价值: {} 元", total_value);
}
```

### 实战示例 2: 文本处理

```rust
fn main() {
    let text = "Hello world! This is Rust programming.";

    // 统计单词长度
    let word_lengths: Vec<_> = text
        .split_whitespace()
        .map(|word| (word, word.len()))
        .collect();

    println!("{:?}", word_lengths);

    // 找出最长的单词
    let longest = text
        .split_whitespace()
        .max_by_key(|word| word.len());

    println!("最长单词: {:?}", longest);
}
```

### 实战示例 3: 配置处理

```rust
use std::collections::HashMap;

fn main() {
    let config_pairs = vec![
        ("host", "localhost"),
        ("port", "8080"),
        ("timeout", "30"),
    ];

    // 转换为 HashMap
    let config: HashMap<_, _> = config_pairs
        .into_iter()
        .collect();

    println!("{:?}", config);

    // 过滤和转换
    let numeric_configs: Vec<_> = vec![
        ("max_connections", "100"),
        ("timeout", "30"),
        ("buffer_size", "1024"),
    ]
    .into_iter()
    .filter_map(|(key, value)| {
        value.parse::<u32>().ok().map(|v| (key, v))
    })
    .collect();

    println!("{:?}", numeric_configs);
}
```

## 性能对比

### 迭代器 vs for 循环

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];

    // for 循环
    let mut sum1 = 0;
    for &x in &v {
        sum1 += x;
    }

    // 迭代器（零成本抽象）
    let sum2: i32 = v.iter().sum();

    // 性能相同！
    assert_eq!(sum1, sum2);
}
```

### 惰性求值的优势

```rust
fn main() {
    let v: Vec<_> = (1..1000000).collect();

    // 只处理前 10 个元素
    let result: Vec<_> = v.iter()
        .map(|x| x * 2)           // 惰性，不会立即执行
        .filter(|x| *x > 100)     // 惰性
        .take(10)                  // 惰性
        .collect();                // 触发执行

    // 效率高：只处理了需要的元素
}
```

## 最佳实践

### 1. 优先使用迭代器

```rust
// 好：使用迭代器
fn sum_positives(numbers: &[i32]) -> i32 {
    numbers.iter()
        .filter(|&&x| x > 0)
        .sum()
}

// 可以，但不够优雅
fn sum_positives_loop(numbers: &[i32]) -> i32 {
    let mut sum = 0;
    for &num in numbers {
        if num > 0 {
            sum += num;
        }
    }
    sum
}
```

### 2. 使用 filter_map 简化代码

```rust
fn main() {
    let strings = vec!["1", "two", "3", "four", "5"];

    // 好：使用 filter_map
    let numbers: Vec<_> = strings.iter()
        .filter_map(|s| s.parse::<i32>().ok())
        .collect();

    // 不够简洁
    let numbers2: Vec<_> = strings.iter()
        .map(|s| s.parse::<i32>())
        .filter(|r| r.is_ok())
        .map(|r| r.unwrap())
        .collect();
}
```

### 3. 避免不必要的 collect

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];

    // 不好：不必要的中间集合
    let temp: Vec<_> = v.iter().map(|x| x * 2).collect();
    let sum: i32 = temp.iter().sum();

    // 好：链式调用
    let sum: i32 = v.iter().map(|x| x * 2).sum();
}
```

### 4. 选择合适的闭包 trait

```rust
// 如果只需不可变访问，使用 Fn
fn apply<F>(f: F, x: i32) -> i32
where
    F: Fn(i32) -> i32,
{
    f(x)
}

// 如果需要可变访问，使用 FnMut
fn apply_mut<F>(mut f: F, x: i32)
where
    F: FnMut(i32),
{
    f(x);
}

// 如果会消耗闭包，使用 FnOnce
fn apply_once<F>(f: F, x: i32)
where
    F: FnOnce(i32),
{
    f(x);
}
```

## 常见陷阱

### 1. 忘记消费迭代器

```rust
fn main() {
    let v = vec![1, 2, 3];

    // 错误：map 不会执行（编译警告）
    v.iter().map(|x| println!("{}", x));

    // 正确：使用 for_each 或 collect
    v.iter().for_each(|x| println!("{}", x));
}
```

### 2. 过早 collect

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];

    // 不够高效
    let temp: Vec<_> = v.iter().filter(|&&x| x > 2).collect();
    let result: Vec<_> = temp.iter().map(|&x| x * 2).collect();

    // 更好：一次性处理
    let result: Vec<_> = v.iter()
        .filter(|&&x| x > 2)
        .map(|x| x * 2)
        .collect();
}
```

### 3. 闭包捕获过多

```rust
fn main() {
    let data = vec![1, 2, 3];
    let extra = String::from("extra");

    // 不好：捕获了整个 data
    let closure = || {
        println!("{}", data.len());
        println!("{}", extra);
    };

    // 好：只捕获需要的
    let len = data.len();
    let closure = || {
        println!("{}", len);
        println!("{}", extra);
    };
}
```

## 总结

- ✅ 闭包提供了灵活的函数抽象
- ✅ 三种闭包 trait：Fn、FnMut、FnOnce
- ✅ 迭代器是零成本抽象
- ✅ 链式调用提高代码可读性
- ✅ 惰性求值提升性能
- ✅ 优先使用迭代器而非 for 循环

掌握闭包和迭代器，能让你的 Rust 代码更加优雅和高效！
