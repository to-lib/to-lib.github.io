---
sidebar_position: 4
title: 结构体和枚举
---

# 结构体和枚举

结构体和枚举是 Rust 中创建自定义数据类型的方式。

## 结构体 (Struct)

### 定义结构体

```rust
struct User {
    username: String,
    email: String,
    sign_in_count: u64,
    active: bool,
}

fn main() {
    // 创建实例
    let user1 = User {
        email: String::from("someone@example.com"),
        username: String::from("someusername123"),
        active: true,
        sign_in_count: 1,
    };
    
    // 访问字段
    println!("用户名: {}", user1.username);
}
```

### 可变结构体

```rust
fn main() {
    let mut user1 = User {
        email: String::from("someone@example.com"),
        username: String::from("someusername123"),
        active: true,
        sign_in_count: 1,
    };
    
    // 修改字段
    user1.email = String::from("anotheremail@example.com");
}
```

### 字段初始化简写

```rust
fn build_user(email: String, username: String) -> User {
    User {
        email,      // 简写：email: email
        username,   // 简写：username: username
        active: true,
        sign_in_count: 1,
    }
}
```

### 结构体更新语法

```rust
fn main() {
    let user1 = User {
        email: String::from("someone@example.com"),
        username: String::from("someusername123"),
        active: true,
        sign_in_count: 1,
    };
    
    // 使用 user1 的部分字段创建 user2
    let user2 = User {
        email: String::from("another@example.com"),
        ..user1  // 其余字段从 user1 复制
    };
    
    // 注意：user1 的 username 被移动到 user2
    // println!("{}", user1.username);  // 错误
}
```

### 元组结构体

```rust
struct Color(i32, i32, i32);
struct Point(i32, i32, i32);

fn main() {
    let black = Color(0, 0, 0);
    let origin = Point(0, 0, 0);
    
    // 访问元组结构体的字段
    println!("Black: ({}, {}, {})", black.0, black.1, black.2);
}
```

### 单元结构体

```rust
struct AlwaysEqual;

fn main() {
    let subject = AlwaysEqual;
}
```

## 结构体方法

### 方法定义

```rust
#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    // 方法：第一个参数是 &self
    fn area(&self) -> u32 {
        self.width * self.height
    }
    
    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.height > other.height
    }
    
    // 可变方法
    fn set_width(&mut self, width: u32) {
        self.width = width;
    }
}

fn main() {
    let rect1 = Rectangle {
        width: 30,
        height: 50,
    };
    
    println!("面积: {}", rect1.area());
    
    let mut rect2 = Rectangle {
        width: 10,
        height: 20,
    };
    
    rect2.set_width(15);
}
```

### 关联函数

```rust
impl Rectangle {
    // 关联函数（类似静态方法）
    fn square(size: u32) -> Rectangle {
        Rectangle {
            width: size,
            height: size,
        }
    }
}

fn main() {
    let sq = Rectangle::square(10);
}
```

### 多个 impl 块

```rust
impl Rectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }
}

impl Rectangle {
    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.height > other.height
    }
}
```

## 枚举 (Enum)

### 定义枚举

```rust
enum IpAddrKind {
    V4,
    V6,
}

fn main() {
    let four = IpAddrKind::V4;
    let six = IpAddrKind::V6;
    
    route(IpAddrKind::V4);
}

fn route(ip_kind: IpAddrKind) {}
```

### 枚举值

```rust
enum IpAddr {
    V4(u8, u8, u8, u8),
    V6(String),
}

fn main() {
    let home = IpAddr::V4(127, 0, 0, 1);
    let loopback = IpAddr::V6(String::from("::1"));
}
```

### 不同类型的枚举变体

```rust
enum Message {
    Quit,                       // 无数据
    Move { x: i32, y: i32 },   // 具名字段
    Write(String),              // 单个 String
    ChangeColor(i32, i32, i32), // 三个 i32
}

impl Message {
    fn call(&self) {
        // 方法体
    }
}

fn main() {
    let m = Message::Write(String::from("hello"));
    m.call();
}
```

## Option 枚举

### Option<T> 定义

```rust
enum Option<T> {
    Some(T),
    None,
}

fn main() {
    let some_number = Some(5);
    let some_string = Some("a string");
    
    let absent_number: Option<i32> = None;
}
```

### 使用 Option

```rust
fn main() {
    let x: i8 = 5;
    let y: Option<i8> = Some(5);
    
    // let sum = x + y;  // 错误：不能直接相加
    
    // 需要先处理 Option
    match y {
        Some(value) => println!("sum = {}", x + value),
        None => println!("y is None"),
    }
}
```

## match 表达式

### 基本 match

```rust
enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter,
}

fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => 1,
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter => 25,
    }
}
```

### 绑定值的模式

```rust
#[derive(Debug)]
enum UsState {
    Alabama,
    Alaska,
    // ...
}

enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter(UsState),
}

fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => 1,
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter(state) => {
            println!("州硬币来自 {:?}!", state);
            25
        }
    }
}
```

### 匹配 Option<T>

```rust
fn plus_one(x: Option<i32>) -> Option<i32> {
    match x {
        None => None,
        Some(i) => Some(i + 1),
    }
}

fn main() {
    let five = Some(5);
    let six = plus_one(five);
    let none = plus_one(None);
}
```

### 通配模式

```rust
fn main() {
    let some_value = 0u8;
    
    match some_value {
        1 => println!("one"),
        3 => println!("three"),
        5 => println!("five"),
        7 => println!("seven"),
        _ => (),  // 匹配所有其他情况
    }
}
```

## if let 简洁控制流

### if let 语法

```rust
fn main() {
    let some_value = Some(3);
    
    // 使用 match
    match some_value {
        Some(3) => println!("three"),
        _ => (),
    }
    
    // 使用 if let（更简洁）
    if let Some(3) = some_value {
        println!("three");
    }
}
```

### if let 与 else

```rust
fn main() {
    let mut count = 0;
    let coin = Coin::Quarter(UsState::Alaska);
    
    if let Coin::Quarter(state) = coin {
        println!("州硬币来自 {:?}!", state);
    } else {
        count += 1;
    }
}
```

## 实用示例

### 自定义 Result 类型

```rust
enum Result<T, E> {
    Ok(T),
    Err(E),
}

fn divide(numerator: f64, denominator: f64) -> Result<f64, String> {
    if denominator == 0.0 {
        Err(String::from("除数不能为0"))
    } else {
        Ok(numerator / denominator)
    }
}

fn main() {
    match divide(10.0, 2.0) {
        Ok(result) => println!("结果: {}", result),
        Err(e) => println!("错误: {}", e),
    }
}
```

### 链表节点

```rust
#[derive(Debug)]
enum List {
    Cons(i32, Box<List>),
    Nil,
}

use List::{Cons, Nil};

fn main() {
    let list = Cons(1, Box::new(Cons(2, Box::new(Cons(3, Box::new(Nil))))));
    println!("{:?}", list);
}
```

## 打印结构体

```rust
#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}

fn main() {
    let rect = Rectangle {
        width: 30,
        height: 50,
    };
    
    // Debug 输出
    println!("{:?}", rect);
    
    // 美化输出
    println!("{:#?}", rect);
    
    // dbg! 宏
    dbg!(&rect);
}
```

## 最佳实践

### 1. 使用 derive 自动实现 trait

```rust
#[derive(Debug, Clone, PartialEq)]
struct Point {
    x: i32,
    y: i32,
}
```

### 2. 优先使用枚举而非多个布尔值

```rust
// 不好
struct Config {
    is_active: bool,
    is_premium: bool,
    is_verified: bool,
}

// 好
enum AccountStatus {
    Active,
    Premium,
    Verified,
    Inactive,
}
```

### 3. 使用 Option 而非空值

```rust
// 不好（在其他语言中）
// fn find_user(id: i32) -> User {
//     // 返回 null 如果未找到
// }

// 好
fn find_user(id: i32) -> Option<User> {
    // 返回 Some(user) 或 None
    None
}
```

## 总结

本文介绍了结构体和枚举的使用：

- ✅ 结构体定义和方法
- ✅ 关联函数
- ✅ 枚举定义和变体
- ✅ Option 枚举
- ✅ match 表达式
- ✅ if let 语法

掌握这些后，继续学习 [模块和包管理](./modules-packages)。
