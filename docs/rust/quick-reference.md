---
sidebar_position: 15
title: 快速参考
---

# Rust 快速参考

快速查找 Rust 常用语法、模式和标准库功能。

## 基础语法

### 变量声明

```rust
let x = 5;              // 不可变
let mut y = 10;         // 可变
const MAX: u32 = 100;   // 常量
static GLOBAL: i32 = 0; // 静态变量
```

### 数据类型

```rust
// 整数
let a: i32 = 42;
let b: u64 = 100;

// 浮点数
let f: f64 = 3.14;

// 布尔
let t: bool = true;

// 字符
let c: char = 'R';

// 元组
let tup: (i32, f64, u8) = (500, 6.4, 1);

// 数组
let arr: [i32; 5] = [1, 2, 3, 4, 5];
```

### 函数

```rust
fn add(x: i32, y: i32) -> i32 {
    x + y  // 表达式返回
}

// 泛型函数
fn max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}
```

### 控制流

```rust
// if
if x > 0 {
    println!("positive");
} else if x < 0 {
    println!("negative");
} else {
    println!("zero");
}

// match
match value {
    0 => println!("zero"),
    1..=5 => println!("one to five"),
    _ => println!("other"),
}

// loop
loop {
    break;
}

// while
while condition {
    // ...
}

// for
for i in 0..10 {
    println!("{}", i);
}
```

## 所有权规则

### 三大规则

1. 每个值都有一个**所有者**
2. 值在任何时刻只能有**一个所有者**
3. 当所有者离开作用域，值将被**丢弃**

### 移动 vs 复制

```rust
// 移动（堆分配）
let s1 = String::from("hello");
let s2 = s1;  // s1 失效

// 复制（栈分配）
let x = 5;
let y = x;  // x 仍有效
```

### 借用规则

```rust
// 不可变借用（可多个）
let r1 = &s;
let r2 = &s;

// 可变借用（只能一个）
let r = &mut s;

// 不能同时存在可变和不可变借用
```

## 常用 Trait

### Display 和 Debug

```rust
use std::fmt;

impl fmt::Display for MyType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

#[derive(Debug)]
struct Point { x: i32, y: i32 }
```

### Clone 和 Copy

```rust
#[derive(Clone)]
struct MyStruct { /* ... */ }

#[derive(Copy, Clone)]
struct Point { x: i32, y: i32 }
```

### Iterator

```rust
impl Iterator for Counter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        // ...
    }
}
```

### From 和 Into

```rust
impl From<i32> for MyType {
    fn from(num: i32) -> Self {
        MyType { value: num }
    }
}

// 自动获得 Into
let t: MyType = 5.into();
```

## 标准库常用模块

### std::collections

```rust
use std::collections::{HashMap, HashSet, VecDeque, BTreeMap};

let mut map = HashMap::new();
let mut set = HashSet::new();
let mut deque = VecDeque::new();
let mut btree = BTreeMap::new();
```

### std::fs - 文件操作

```rust
use std::fs;

// 读取文件
let content = fs::read_to_string("file.txt")?;

// 写入文件
fs::write("file.txt", "content")?;

// 复制文件
fs::copy("src.txt", "dst.txt")?;

// 删除文件
fs::remove_file("file.txt")?;

// 创建目录
fs::create_dir("my_dir")?;
```

### std::io - 输入输出

```rust
use std::io::{self, Read, Write};

// 标准输入
let mut input = String::new();
io::stdin().read_line(&mut input)?;

// 标准输出
println!("Hello, {}!", name);
eprintln!("Error: {}", error);

// 文件读写
let mut file = File::open("file.txt")?;
let mut contents = String::new();
file.read_to_string(&mut contents)?;
```

### std::path - 路径操作

```rust
use std::path::{Path, PathBuf};

let path = Path::new("/tmp/file.txt");
let parent = path.parent();
let file_name = path.file_name();
let extension = path.extension();

let mut path_buf = PathBuf::from("/tmp");
path_buf.push("file.txt");
```

### std::env - 环境变量

```rust
use std::env;

// 获取环境变量
let home = env::var("HOME")?;

// 设置环境变量
env::set_var("MY_VAR", "value");

// 获取命令行参数
let args: Vec<String> = env::args().collect();

// 当前目录
let current_dir = env::current_dir()?;
```

### std::thread - 线程

```rust
use std::thread;
use std::time::Duration;

// 创建线程
let handle = thread::spawn(|| {
    println!("Hello from thread");
});

// 等待线程
handle.join().unwrap();

// 睡眠
thread::sleep(Duration::from_secs(1));
```

## 错误处理

### Result 和 Option

```rust
// Result
fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("division by zero".to_string())
    } else {
        Ok(a / b)
    }
}

// Option
fn find_item(items: &[i32], target: i32) -> Option<usize> {
    items.iter().position(|&x| x == target)
}
```

### ? 运算符

```rust
fn read_file() -> Result<String, io::Error> {
    let mut file = File::open("file.txt")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}
```

### unwrap 和 expect

```rust
let x = Some(5);
let value = x.unwrap();  // panic if None

let value = x.expect("x should have a value");
```

### 自定义错误

```rust
use std::fmt;
use std::error::Error;

#[derive(Debug)]
struct MyError {
    details: String
}

impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.details)
    }
}

impl Error for MyError {}
```

## 闭包和迭代器

### 闭包语法

```rust
// 最小语法
let add_one = |x| x + 1;

// 带类型注解
let add_one = |x: i32| -> i32 { x + 1 };

// 捕获环境
let y = 10;
let add_y = |x| x + y;

// move 关键字
let s = String::from("hello");
let print = move || println!("{}", s);
```

### 迭代器链

```rust
let result: Vec<_> = vec![1, 2, 3, 4, 5]
    .iter()
    .filter(|&&x| x % 2 == 0)
    .map(|&x| x * 2)
    .collect();

// 其他常用方法
.take(n)           // 前 n 个
.skip(n)           // 跳过 n 个
.enumerate()       // 添加索引
.zip(other)        // 组合
.fold(init, f)     // 折叠
.any(predicate)    // 是否有满足
.all(predicate)    // 是否全满足
.find(predicate)   // 查找
```

## 并发编程

### 线程

```rust
use std::thread;

let handle = thread::spawn(|| {
    // 线程代码
});

handle.join().unwrap();
```

### 通道

```rust
use std::sync::mpsc;

let (tx, rx) = mpsc::channel();

thread::spawn(move || {
    tx.send("hello").unwrap();
});

let msg = rx.recv().unwrap();
```

### Mutex 和 Arc

```rust
use std::sync::{Arc, Mutex};

let counter = Arc::new(Mutex::new(0));

let counter_clone = Arc::clone(&counter);
thread::spawn(move || {
    let mut num = counter_clone.lock().unwrap();
    *num += 1;
});
```

## 异步编程

### async/await

```rust
async fn fetch_data() -> String {
    "data".to_string()
}

async fn process() {
    let data = fetch_data().await;
    println!("{}", data);
}
```

### Tokio

```rust
#[tokio::main]
async fn main() {
    let handle = tokio::spawn(async {
        // 异步任务
    });

    handle.await.unwrap();
}
```

## 宏

### 声明宏

```rust
macro_rules! my_macro {
    ($x:expr) => {
        println!("Value: {}", $x);
    };
}

my_macro!(42);
```

### 常用宏

```rust
println!()       // 打印
format!()        // 格式化
vec![]           // 创建 Vec
panic!()         // panic
assert!()        // 断言
assert_eq!()     // 相等断言
dbg!()           // 调试打印
todo!()          // 待实现
unimplemented!() // 未实现
```

## 测试

### 单元测试

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 2), 4);
    }

    #[test]
    #[should_panic]
    fn test_panic() {
        panic!("expected panic");
    }
}
```

### 集成测试

在 `tests/` 目录下创建测试文件。

## 常用命令

```bash
# Cargo
cargo new <name>        # 新建项目
cargo build             # 构建
cargo build --release   # 发布构建
cargo run               # 运行
cargo test              # 测试
cargo doc --open        # 生成文档
cargo check             # 检查
cargo clean             # 清理
cargo update            # 更新依赖

# Rustup
rustup update           # 更新 Rust
rustup default stable   # 设置默认版本
rustup target add <target> # 添加目标平台

# 工具
rustfmt                 # 格式化代码
clippy                  # 代码检查
```

## 属性

```rust
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
#[warn(missing_docs)]
#[cfg(target_os = "linux")]
#[inline]
#[test]
#[should_panic]
#[ignore]
```

## 模式匹配

```rust
// 字面值
match x {
    1 => println!("one"),
    2 => println!("two"),
    _ => println!("other"),
}

// 范围
match x {
    1..=5 => println!("one to five"),
    _ => println!("other"),
}

// 解构
match point {
    Point { x: 0, y } => println!("y axis: {}", y),
    Point { x, y: 0 } => println!("x axis: {}", x),
    Point { x, y } => println!("({}, {})", x, y),
}

// 守卫
match num {
    Some(x) if x < 5 => println!("less than five: {}", x),
    Some(x) => println!("{}", x),
    None => (),
}
```

## 生命周期

```rust
// 函数签名
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

// 结构体
struct ImportantExcerpt<'a> {
    part: &'a str,
}

// 方法
impl<'a> ImportantExcerpt<'a> {
    fn level(&self) -> i32 {
        3
    }
}
```

## 智能指针

```rust
use std::rc::Rc;
use std::cell::RefCell;
use std::sync::Arc;

// Box
let b = Box::new(5);

// Rc (引用计数)
let a = Rc::new(5);
let b = Rc::clone(&a);

// RefCell (内部可变性)
let value = RefCell::new(5);
*value.borrow_mut() += 1;

// Arc (原子引用计数)
let data = Arc::new(Mutex::new(0));
```

## 类型转换

```rust
// as
let x = 5i32;
let y = x as f64;

// From/Into
let s = String::from("hello");
let s: String = "hello".into();

// TryFrom/TryInto
let n: u8 = "42".parse().unwrap();
```

## 总结

本速查表涵盖了 Rust 最常用的语法和模式，可以作为日常开发的快速参考。

需要详细了解某个主题时，请参阅对应的详细文档章节。
