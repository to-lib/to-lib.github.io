---
sidebar_position: 6
title: 泛型和 Trait
---

# 泛型和 Trait

泛型和 Trait 是 Rust 实现代码复用和抽象的重要特性。

## 泛型

### 泛型函数

```rust
//泛型函数
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    
    for item in list {
        if item > largest {
            largest = item;
        }
    }
    
    largest
}

fn main() {
    let number_list = vec![34, 50, 25, 100, 65];
    let result = largest(&number_list);
    println!("最大的数字是 {}", result);
    
    let char_list = vec!['y', 'm', 'a', 'q'];
    let result = largest(&char_list);
    println!("最大的字符是 {}", result);
}
```

### 泛型结构体

```rust
struct Point<T> {
    x: T,
    y: T,
}

fn main() {
    let integer = Point { x: 5, y: 10 };
    let float = Point { x: 1.0, y: 4.0 };
}
```

### 多个泛型参数

```rust
struct Point<T, U> {
    x: T,
    y: U,
}

fn main() {
    let both_integer = Point { x: 5, y: 10 };
    let both_float = Point { x: 1.0, y: 4.0 };
    let integer_and_float = Point { x: 5, y: 4.0 };
}
```

### 泛型枚举

```rust
enum Option<T> {
    Some(T),
    None,
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

### 泛型方法

```rust
struct Point<T> {
    x: T,
    y: T,
}

impl<T> Point<T> {
    fn x(&self) -> &T {
        &self.x
    }
}

// 只为特定类型实现方法
impl Point<f32> {
    fn distance_from_origin(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}

fn main() {
    let p = Point { x: 5, y: 10 };
    println!("p.x = {}", p.x());
    
    let p = Point { x: 3.0, y: 4.0 };
    println!("距离原点: {}", p.distance_from_origin());
}
```

## Trait

Trait 定义共享的行为。

### 定义 Trait

```rust
pub trait Summary {
    fn summarize(&self) -> String;
}
```

### 实现 Trait

```rust
pub struct NewsArticle {
    pub headline: String,
    pub location: String,
    pub author: String,
    pub content: String,
}

impl Summary for NewsArticle {
    fn summarize(&self) -> String {
        format!("{}, by {} ({})", self.headline, self.author, self.location)
    }
}

pub struct Tweet {
    pub username: String,
    pub content: String,
    pub reply: bool,
    pub retweet: bool,
}

impl Summary for Tweet {
    fn summarize(&self) -> String {
        format!("{}: {}", self.username, self.content)
    }
}
```

### 默认实现

```rust
pub trait Summary {
    fn summarize(&self) -> String {
        String::from("(阅读更多...)")
    }
}

// 使用默认实现
impl Summary for NewsArticle {}

fn main() {
    let article = NewsArticle {
        headline: String::from("大新闻！"),
        location: String::from("北京"),
        author: String::from("张三"),
        content: String::from("新闻内容..."),
    };
    
    println!("新文章！{}", article.summarize());
}
```

### 默认实现调用其他方法

```rust
pub trait Summary {
    fn summarize_author(&self) -> String;
    
    fn summarize(&self) -> String {
        format!("(更多来自 {}...)", self.summarize_author())
    }
}

impl Summary for Tweet {
    fn summarize_author(&self) -> String {
        format!("@{}", self.username)
    }
}
```

## Trait 作为参数

### impl Trait 语法

```rust
pub fn notify(item: &impl Summary) {
    println!("突发新闻！{}", item.summarize());
}
```

### Trait Bound 语法

```rust
pub fn notify<T: Summary>(item: &T) {
    println!("突发新闻！{}", item.summarize());
}

// 多个参数
pub fn notify<T: Summary>(item1: &T, item2: &T) {
    // ...
}
```

### 多个 Trait Bound

```rust
use std::fmt::Display;

pub fn notify(item: &(impl Summary + Display)) {
    // ...
}

// 或
pub fn notify<T: Summary + Display>(item: &T) {
    // ...
}
```

### where 子句

```rust
// 不使用 where
fn some_function<T: Display + Clone, U: Clone + Debug>(t: &T, u: &U) -> i32 {
    // ...
    0
}

// 使用 where（更清晰）
fn some_function<T, U>(t: &T, u: &U) -> i32
where
    T: Display + Clone,
    U: Clone + Debug,
{
    // ...
    0
}
```

## 返回实现 Trait 的类型

```rust
fn returns_summarizable() -> impl Summary {
    Tweet {
        username: String::from("horse_ebooks"),
        content: String::from("当然，正如你可能已经知道的那样"),
        reply: false,
        retweet: false,
    }
}
```

### 限制：只能返回单一类型

```rust
// 错误：不能返回不同的类型
// fn returns_summarizable(switch: bool) -> impl Summary {
//     if switch {
//         NewsArticle { ... }
//     } else {
//         Tweet { ... }
//     }
// }
```

## 使用 Trait Bound 有条件地实现方法

```rust
use std::fmt::Display;

struct Pair<T> {
    x: T,
    y: T,
}

impl<T> Pair<T> {
    fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

// 只为实现了 Display 和 PartialOrd 的类型实现此方法
impl<T: Display + PartialOrd> Pair<T> {
    fn cmp_display(&self) {
        if self.x >= self.y {
            println!("最大的是 x = {}", self.x);
        } else {
            println!("最大的是 y = {}", self.y);
        }
    }
}
```

## 常用 Trait

### Clone

```rust
#[derive(Clone)]
struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let p1 = Point { x: 1, y: 2 };
    let p2 = p1.clone();
}
```

### Copy

```rust
#[derive(Copy, Clone)]
struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let p1 = Point { x: 1, y: 2 };
    let p2 = p1;  // 自动复制
    println!("p1: ({}, {})", p1.x, p1.y);  // p1 仍然有效
}
```

### Debug

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
    
    println!("{:?}", rect);
    println!("{:#?}", rect);
}
```

### Display

```rust
use std::fmt;

struct Point {
    x: i32,
    y: i32,
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

fn main() {
    let p = Point { x: 3, y: 4 };
    println!("点的坐标: {}", p);
}
```

### PartialEq 和 Eq

```rust
#[derive(PartialEq, Eq)]
struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let p1 = Point { x: 1, y: 2 };
    let p2 = Point { x: 1, y: 2 };
    
    assert_eq!(p1, p2);
}
```

### PartialOrd 和 Ord

```rust
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let p1 = Point { x: 1, y: 2 };
    let p2 = Point { x: 2, y: 3 };
    
    assert!(p1 < p2);
}
```

## 运算符重载

```rust
use std::ops::Add;

#[derive(Debug, PartialEq)]
struct Point {
    x: i32,
    y: i32,
}

impl Add for Point {
    type Output = Point;
    
    fn add(self, other: Point) -> Point {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

fn main() {
    let p1 = Point { x: 1, y: 2 };
    let p2 = Point { x: 3, y: 4 };
    let p3 = p1 + p2;
    
    assert_eq!(p3, Point { x: 4, y: 6 });
}
```

## 关联类型

```rust
pub trait Iterator {
    type Item;  // 关联类型
    
    fn next(&mut self) -> Option<Self::Item>;
}

struct Counter {
    count: u32,
}

impl Iterator for Counter {
    type Item = u32;
    
    fn next(&mut self) -> Option<Self::Item> {
        self.count += 1;
        
        if self.count < 6 {
            Some(self.count)
        } else {
            None
        }
    }
}
```

## 完全限定语法

```rust
trait Pilot {
    fn fly(&self);
}

trait Wizard {
    fn fly(&self);
}

struct Human;

impl Pilot for Human {
    fn fly(&self) {
        println!("机长讲话");
    }
}

impl Wizard for Human {
    fn fly(&self) {
        println!("飞起来！");
    }
}

impl Human {
    fn fly(&self) {
        println!("挥动手臂");
    }
}

fn main() {
    let person = Human;
    Pilot::fly(&person);   // 机长讲话
    Wizard::fly(&person);  // 飞起来！
    person.fly();          // 挥动手臂
}
```

## Trait 对象

### 动态分发

```rust
pub trait Draw {
    fn draw(&self);
}

pub struct Screen {
    pub components: Vec<Box<dyn Draw>>,
}

impl Screen {
    pub fn run(&self) {
        for component in self.components.iter() {
            component.draw();
        }
    }
}

pub struct Button {
    pub width: u32,
    pub height: u32,
}

impl Draw for Button {
    fn draw(&self) {
        println!("绘制按钮");
    }
}

pub struct TextField {
    pub width: u32,
}

impl Draw for TextField {
    fn draw(&self) {
        println!("绘制文本框");
    }
}

fn main() {
    let screen = Screen {
        components: vec![
            Box::new(Button {
                width: 50,
                height: 10,
            }),
            Box::new(TextField {
                width: 100,
            }),
        ],
    };
    
    screen.run();
}
```

## 最佳实践

### 1. 优先使用泛型而非重复代码

```rust
// 不好：重复代码
fn largest_i32(list: &[i32]) -> i32 { /* ... */ 0 }
fn largest_char(list: &[char]) -> char { /* ... */ 'a' }

// 好：使用泛型
fn largest<T: PartialOrd + Copy>(list: &[T]) -> T { /* ... */ list[0] }
```

### 2. 使用 derive 宏

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
struct Point {
    x: i32,
    y: i32,
}
```

### 3. 为外部类型实现本地 Trait

```rust
// 可以为 Vec<T> 实现我们自己的 trait
trait MyTrait {
    fn my_method(&self);
}

impl<T> MyTrait for Vec<T> {
    fn my_method(&self) {
        println!("自定义方法");
    }
}
```

## 高级 Trait 使用

### 关联类型

```rust
trait Iterator {
    type Item;  // 关联类型
    
    fn next(&mut self) -> Option<Self::Item>;
}

struct Counter {
    count: u32,
}

impl Iterator for Counter {
    type Item = u32;  // 指定关联类型
    
    fn next(&mut self) -> Option<Self::Item> {
        self.count += 1;
        if self.count < 6 {
            Some(self.count)
        } else {
            None
        }
    }
}
```

### Trait 对象详解

```rust
trait Draw {
    fn draw(&self);
}

struct Circle {
    radius: f64,
}

struct Rectangle {
    width: f64,
    height: f64,
}

impl Draw for Circle {
    fn draw(&self) {
        println!("绘制圆形,半径: {}", self.radius);
    }
}

impl Draw for Rectangle {
    fn draw(&self) {
        println!("绘制矩形,宽: {}, 高: {}", self.width, self.height);
    }
}

// 使用 trait 对象
fn draw_all(shapes: &[Box<dyn Draw>]) {
    for shape in shapes {
        shape.draw();
    }
}

fn main() {
    let shapes: Vec<Box<dyn Draw>> = vec![
        Box::new(Circle { radius: 5.0 }),
        Box::new(Rectangle { width: 10.0, height: 20.0 }),
    ];
    
    draw_all(&shapes);
}
```

### Trait 对象 vs 泛型

```rust
// 静态分发(泛型):编译时确定类型,性能更好
fn draw_generic<T: Draw>(shape: &T) {
    shape.draw();
}

// 动态分发(trait对象):运行时确定类型,更灵活
fn draw_dynamic(shape: &dyn Draw) {
    shape.draw();
}
```

### 超 Trait (Supertrait)

```rust
use std::fmt::Display;

// OutlinePrint 依赖 Display trait
trait OutlinePrint: Display {
    fn outline_print(&self) {
        let output = self.to_string();
        let len = output.len();
        println!("{}", "*".repeat(len + 4));
        println!("*{}*", " ".repeat(len + 2));
        println!("* {} *", output);
        println!("*{}*", " ".repeat(len + 2));
        println!("{}", "*".repeat(len + 4));
    }
}

struct Point {
    x: i32,
    y: i32,
}

impl Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl OutlinePrint for Point {}

fn main() {
    let point = Point { x: 1, y: 3 };
    point.outline_print();
}
```

### Newtype 模式绕过孤儿规则

```rust
use std::fmt;

// 为外部类型Vec<String>实现外部trait Display
struct Wrapper(Vec<String>);

impl fmt::Display for Wrapper {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}]", self.0.join(", "))
    }
}

fn main() {
    let w = Wrapper(vec![
        String::from("hello"),
        String::from("world"),
    ]);
    println!("w = {}", w);
}
```

## Trait 边界高级技巧

### Where 子句

```rust
// 复杂的 trait 边界使用 where
fn some_function<T, U>(t: &T, u: &U) -> i32
where
    T: Display + Clone,
    U: Clone + Debug,
{
    // 函数体
    0
}
```

### 多个 Trait 边界

```rust
use std::fmt::{Debug, Display};

fn compare_prints<T: Debug + Display>(t: &T) {
    println!("Debug: {:?}", t);
    println!("Display: {}", t);
}
```

### 返回实现 Trait 的类型

```rust
fn returns_summarizable() -> impl Summary {
    Tweet {
        username: String::from("horse_ebooks"),
        content: String::from("of course"),
        reply: false,
        retweet: false,
    }
}
```

### 有条件地实现方法

```rust
use std::fmt::Display;

struct Pair<T> {
    x: T,
    y: T,
}

impl<T> Pair<T> {
    fn new(x: T, y: T) -> Self {
        Pair { x, y }
    }
}

// 只为实现了 Display + PartialOrd 的类型实现 cmp_display
impl<T: Display + PartialOrd> Pair<T> {
    fn cmp_display(&self) {
        if self.x >= self.y {
            println!("最大值是 x = {}", self.x);
        } else {
            println!("最大值是 y = {}", self.y);
        }
    }
}
```

### Blanket Implementation

```rust
// 为所有实现了 Display 的类型实现 ToString
trait ToString {
    fn to_string(&self) -> String;
}

impl<T: Display> ToString for T {
    fn to_string(&self) -> String {
        format!("{}", self)
    }
}
```

## 运算符重载

```rust
use std::ops::Add;

#[derive(Debug, Copy, Clone, PartialEq)]
struct Point {
    x: i32,
    y: i32,
}

impl Add for Point {
    type Output = Point;

    fn add(self, other: Point) -> Point {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

fn main() {
    let p1 = Point { x: 1, y: 0 };
    let p2 = Point { x: 2, y: 3 };
    let p3 = p1 + p2;
    
    println!("{:?} + {:?} = {:?}", p1, p2, p3);
}
```

## 完全限定语法

```rust
trait Pilot {
    fn fly(&self);
}

trait Wizard {
    fn fly(&self);
}

struct Human;

impl Pilot for Human {
    fn fly(&self) {
        println!("这是你的机长在讲话");
    }
}

impl Wizard for Human {
    fn fly(&self) {
        println!("飞起来!");
    }
}

impl Human {
    fn fly(&self) {
        println!("*疯狂挥动手臂*");
    }
}

fn main() {
    let person = Human;
    
    person.fly();                // 调用 Human::fly
    Pilot::fly(&person);         // 调用 Pilot::fly
    Wizard::fly(&person);        // 调用 Wizard::fly
    <Human as Pilot>::fly(&person);  // 完全限定语法
}
```

## 总结

本文介绍了泛型和 Trait：

- ✅ 泛型函数、结构体和枚举
- ✅ Trait 定义和实现
- ✅ Trait Bound
- ✅ 默认实现
- ✅ 常用 Trait
- ✅ Trait 对象

掌握泛型和 Trait 后，继续学习 [并发编程](/docs/rust/concurrency)。
