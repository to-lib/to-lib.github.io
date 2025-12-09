---
sidebar_position: 2
title: Rust 基础语法
---

# Rust 基础语法

掌握 Rust 基础语法是学习 Rust 的第一步。本文涵盖变量、数据类型、函数和控制流等核心概念。

## 变量和可变性

### 变量声明

```rust
fn main() {
    // 不可变变量（默认）
    let x = 5;
    println!("x 的值是: {}", x);
    
    // x = 6;  // 错误：不能对不可变变量二次赋值
    
    // 可变变量
    let mut y = 5;
    println!("y 的值是: {}", y);
    y = 6;
    println!("y 的值变为: {}", y);
}
```

### 常量

```rust
// 常量必须标注类型，使用大写下划线命名
const MAX_POINTS: u32 = 100_000;
const PI: f64 = 3.14159;

fn main() {
    println!("最大分数: {}", MAX_POINTS);
}
```

### 变量遮蔽 (Shadowing)

```rust
fn main() {
    let x = 5;
    
    // 遮蔽之前的 x
    let x = x + 1;
    
    {
        // 内部作用域的遮蔽
        let x = x * 2;
        println!("内部作用域 x 的值是: {}", x);  // 12
    }
    
    println!("外部作用域 x 的值是: {}", x);  // 6
    
    // 遮蔽可以改变类型
    let spaces = "   ";
    let spaces = spaces.len();
}
```

## 数据类型

### 标量类型

#### 整数类型

| 长度 | 有符号 | 无符号 |
|------|--------|--------|
| 8-bit | i8 | u8 |
| 16-bit | i16 | u16 |
| 32-bit | i32 | u32 |
| 64-bit | i64 | u64 |
| 128-bit | i128 | u128 |
| arch | isize | usize |

```rust
fn main() {
    let a: i32 = 42;
    let b: u8 = 255;
    let c = 98_222;  // 使用下划线分隔，提高可读性
    let d = 0xff;    // 十六进制
    let e = 0o77;    // 八进制
    let f = 0b1111_0000;  // 二进制
    let g = b'A';    // 字节（u8）
}
```

#### 浮点类型

```rust
fn main() {
    let x = 2.0;      // f64（默认）
    let y: f32 = 3.0; // f32
}
```

#### 布尔类型

```rust
fn main() {
    let t = true;
    let f: bool = false;
}
```

#### 字符类型

```rust
fn main() {
    let c = 'z';
    let z = 'ℤ';
    let heart_eyed_cat = '😻';
    
    // char 是 4 字节的 Unicode 标量值
}
```

### 复合类型

#### 元组

```rust
fn main() {
    // 元组可以包含不同类型
    let tup: (i32, f64, u8) = (500, 6.4, 1);
    
    // 解构
    let (x, y, z) = tup;
    println!("y 的值是: {}", y);
    
    // 索引访问
    let five_hundred = tup.0;
    let six_point_four = tup.1;
    let one = tup.2;
}
```

#### 数组

```rust
fn main() {
    // 数组长度固定，元素类型相同
    let a = [1, 2, 3, 4, 5];
    
    // 指定类型和长度
    let b: [i32; 5] = [1, 2, 3, 4, 5];
    
    // 初始化相同值
    let c = [3; 5];  // [3, 3, 3, 3, 3]
    
    // 访问元素
    let first = a[0];
    let second = a[1];
    
    // 数组长度
    println!("数组长度: {}", a.len());
}
```

## 函数

### 函数定义

```rust
fn main() {
    println!("Hello, world!");
    
    another_function();
    function_with_params(5, 'h');
    
    let result = add(5, 3);
    println!("5 + 3 = {}", result);
}

fn another_function() {
    println!("另一个函数");
}

fn function_with_params(value: i32, label: char) {
    println!("参数值: {}{}", value, label);
}

fn add(x: i32, y: i32) -> i32 {
    x + y  // 表达式作为返回值（无分号）
}
```

### 语句和表达式

```rust
fn main() {
    // 语句：执行操作但不返回值
    let y = 6;
    
    // 表达式：求值并返回值
    let x = {
        let z = 3;
        z + 1  // 注意：没有分号
    };
    println!("x 的值是: {}", x);  // 4
}
```

## 控制流

### if 表达式

```rust
fn main() {
    let number = 6;
    
    // 简单 if
    if number < 5 {
        println!("条件为真");
    } else {
        println!("条件为假");
    }
    
    // if-else if-else
    if number % 4 == 0 {
        println!("数字可被 4 整除");
    } else if number % 3 == 0 {
        println!("数字可被 3 整除");
    } else if number % 2 == 0 {
        println!("数字可被 2 整除");
    } else {
        println!("数字不能被 4、3 或 2 整除");
    }
    
    // if 是表达式，可以赋值
    let condition = true;
    let value = if condition { 5 } else { 6 };
    println!("value 的值是: {}", value);
}
```

### loop 循环

```rust
fn main() {
    // 无限循环
    let mut counter = 0;
    
    let result = loop {
        counter += 1;
        
        if counter == 10 {
            break counter * 2;  // loop 可以返回值
        }
    };
    
    println!("结果是: {}", result);  // 20
}
```

### while 循环

```rust
fn main() {
    let mut number = 3;
    
    while number != 0 {
        println!("{}!", number);
        number -= 1;
    }
    
    println!("发射！");
}
```

### for 循环

```rust
fn main() {
    // 遍历数组
    let a = [10, 20, 30, 40, 50];
    
    for element in a.iter() {
        println!("值是: {}", element);
    }
    
    // 使用范围
    for number in 1..4 {
        println!("{}", number);  // 1, 2, 3
    }
    
    // 包含结束值
    for number in 1..=4 {
        println!("{}", number);  // 1, 2, 3, 4
    }
    
    // 倒序
    for number in (1..4).rev() {
        println!("{}", number);  // 3, 2, 1
    }
}
```

## 字符串

### String vs &str

```rust
fn main() {
    // String：可变、堆分配
    let mut s = String::from("hello");
    s.push_str(", world");
    println!("{}", s);
    
    // &str：字符串切片、不可变
    let s = "hello, world";
    
    // 字符串方法
    let s = String::from("hello");
    println!("长度: {}", s.len());
    println!("是否为空: {}", s.is_empty());
    println!("包含 'ell': {}", s.contains("ell"));
    
    // 字符串切片
    let s = String::from("hello world");
    let hello = &s[0..5];
    let world = &s[6..11];
}
```

## 注释

```rust
fn main() {
    // 这是单行注释
    
    /*
     * 这是
     * 多行注释
     */
    
    /// 这是文档注释
    /// 用于生成文档
    
    //! 这是模块级文档注释
}
```

## 打印输出

```rust
fn main() {
    // println! 宏
    println!("Hello, world!");
    
    // 格式化输出
    let x = 5;
    let y = 10;
    println!("x = {} and y = {}", x, y);
    
    // 位置参数
    println!("{0}, {1}, {0}", "Alice", "Bob");
    
    // 命名参数
    println!("{name} is {age} years old", name="张三", age=25);
    
    // 调试输出
    println!("{:?}", (1, 2, 3));
    
    // 美化调试输出
    #[derive(Debug)]
    struct Point {
        x: i32,
        y: i32,
    }
    let origin = Point { x: 0, y: 0 };
    println!("{:#?}", origin);
}
```

## 类型转换

```rust
fn main() {
    // as 关键字
    let a = 3.14;
    let b = a as i32;  // 3
    
    // 整数间转换
    let x = 255u8;
    let y = x as i32;
    
    // 字符串转换
    let num: i32 = "42".parse().expect("不是一个数字！");
    let num: i32 = "42".parse().unwrap();
    
    // to_string
    let s = 42.to_string();
}
```

## 运算符

```rust
fn main() {
    // 算术运算符
    let sum = 5 + 10;
    let difference = 95.5 - 4.3;
    let product = 4 * 30;
    let quotient = 56.7 / 32.2;
    let remainder = 43 % 5;
    
    // 比较运算符
    let is_greater = 5 > 3;
    let is_equal = 5 == 5;
    let is_not_equal = 5 != 3;
    
    // 逻辑运算符
    let and = true && false;
    let or = true || false;
    let not = !true;
    
    // 位运算符
    let bitwise_and = 0b1010 & 0b1100;  // 0b1000
    let bitwise_or = 0b1010 | 0b1100;   // 0b1110
    let bitwise_xor = 0b1010 ^ 0b1100;  // 0b0110
    let left_shift = 1 << 2;             // 4
    let right_shift = 8 >> 2;            // 2
}
```

## 所有权预览

```rust
fn main() {
    // 所有权转移
    let s1 = String::from("hello");
    let s2 = s1;  // s1 的所有权移动到 s2
    // println!("{}", s1);  // 错误：s1 已失效
    
    // 克隆
    let s1 = String::from("hello");
    let s2 = s1.clone();
    println!("s1 = {}, s2 = {}", s1, s2);  // 都有效
    
    // 栈上的数据（实现了 Copy trait）
    let x = 5;
    let y = x;
    println!("x = {}, y = {}", x, y);  // 都有效
}
```

## 最佳实践

### 命名规范

```rust
// 变量和函数：snake_case
let my_variable = 5;
fn my_function() {}

// 类型和 trait：PascalCase
struct MyStruct {}
trait MyTrait {}

// 常量：SCREAMING_SNAKE_CASE
const MAX_VALUE: u32 = 100;

// 生命周期：小写单字母
fn foo<'a>(x: &'a str) {}
```

### 代码风格

```rust
fn main() {
    // 使用 cargo fmt 自动格式化代码
    
    // 优先使用不可变变量
    let x = 5;  // 好
    // let mut x = 5;  // 仅在需要时使用
    
    // 使用类型推断
    let x = 5;           // 好
    let x: i32 = 5;      // 仅在需要明确类型时
    
    // 使用表达式而非语句
    let max = if a > b { a } else { b };  // 好
}
```

## 常见错误

### 可变性错误

```rust
fn main() {
    let x = 5;
    // x = 6;  // 错误：不能对不可变变量赋值
    
    let mut x = 5;
    x = 6;  // 正确
}
```

### 类型不匹配

```rust
fn main() {
    let condition = true;
    // let number = if condition { 5 } else { "six" };  // 错误：类型不匹配
    
    let number = if condition { 5 } else { 6 };  // 正确
}
```

### 数组越界

```rust
fn main() {
    let a = [1, 2, 3];
    // let element = a[10];  // 运行时panic
    
    // 使用 get 方法安全访问
    match a.get(10) {
        Some(value) => println!("值: {}", value),
        None => println!("索引越界"),
    }
}
```

## 迭代器

### 迭代器基础

迭代器模式允许你对一个序列的项进行某些处理。

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];
    
    // 创建迭代器
    let v_iter = v.iter();
    
    // 使用迭代器
    for val in v_iter {
        println!("{}", val);
    }
}
```

### Iterator Trait

```rust
pub trait Iterator {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
    
    // 其他方法有默认实现...
}
```

### 三种迭代方式

```rust
fn main() {
    let v = vec![1, 2, 3];
    
    // iter(): 不可变引用
    for val in v.iter() {
        println!("{}", val);  // &i32
    }
    
    // iter_mut(): 可变引用
    let mut v = vec![1, 2, 3];
    for val in v.iter_mut() {
        *val += 1;  // &mut i32
    }
    
    // into_iter(): 获取所有权
    for val in v.into_iter() {
        println!("{}", val);  // i32
    }
    // println!("{:?}", v);  // 错误: v 已被移动
}
```

### 消费适配器

```rust
fn main() {
    let v = vec![1, 2, 3];
    
    // sum: 消费迭代器
    let total: i32 = v.iter().sum();
    println!("总和: {}", total);
    
    // collect: 收集到集合
    let v2: Vec<_> = v.iter().collect();
    
    // count: 计数
    let count = v.iter().count();
}
```

### 迭代器适配器

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];
    
    // map: 转换每个元素
    let v2: Vec<_> = v.iter()
        .map(|x| x + 1)
        .collect();
    println!("{:?}", v2);  // [2, 3, 4, 5, 6]
    
    // filter: 过滤元素
    let v3: Vec<_> = v.iter()
        .filter(|&x| x % 2 == 0)
        .collect();
    println!("{:?}", v3);  // [2, 4]
    
    // 链式调用
    let result: Vec<_> = v.iter()
        .filter(|&&x| x % 2 == 0)
        .map(|&x| x * 2)
        .collect();
    println!("{:?}", result);  // [4, 8]
}
```

### 常用迭代器方法

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];
    
    // take: 取前 n 个
    let first_three: Vec<_> = v.iter().take(3).collect();
    
    // skip: 跳过前 n 个
    let after_two: Vec<_> = v.iter().skip(2).collect();
    
    // enumerate: 带索引
    for (i, val) in v.iter().enumerate() {
        println!("索引 {}: 值 {}", i, val);
    }
    
    // zip: 组合两个迭代器
    let v2 = vec!["a", "b", "c"];
    for (num, letter) in v.iter().zip(v2.iter()) {
        println!("{}: {}", num, letter);
    }
    
    // fold: 折叠/累积
    let sum = v.iter().fold(0, |acc, &x| acc + x);
    println!("总和: {}", sum);
    
    // any: 任意一个满足
    let has_even = v.iter().any(|&x| x % 2 == 0);
    
    // all: 全部满足
    let all_positive = v.iter().all(|&x| x > 0);
    
    // find: 查找第一个
    let first_even = v.iter().find(|&&x| x % 2 == 0);
    
    // position: 查找位置
    let pos = v.iter().position(|&x| x == 3);
}
```

### 自定义迭代器

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
        self.count += 1;
        
        if self.count < 6 {
            Some(self.count)
        } else {
            None
        }
    }
}

fn main() {
    let counter = Counter::new();
    
    for num in counter {
        println!("{}", num);  // 1, 2, 3, 4, 5
    }
    
    // 使用迭代器方法
    let sum: u32 = Counter::new()
        .zip(Counter::new().skip(1))
        .map(|(a, b)| a * b)
        .filter(|x| x % 3 == 0)
        .sum();
    println!("结果: {}", sum);  // 18
}
```

### 迭代器性能

```rust
// 迭代器是零成本抽象
fn main() {
    let v = vec![1, 2, 3, 4, 5];
    
    // 使用迭代器(推荐)
    let sum: i32 = v.iter().sum();
    
    // 等价的 for 循环
    let mut sum = 0;
    for &x in &v {
        sum += x;
    }
    
    // 编译后性能相同!
}
```

## 闭包

### 闭包基础

闭包是可以捕获环境的匿名函数。

```rust
fn main() {
    // 闭包语法
    let add_one = |x: i32| -> i32 { x + 1 };
    
    // 类型推断
    let add_one = |x| x + 1;
    
    // 调用闭包
    let result = add_one(5);
    println!("{}", result);  // 6
    
    // 多行闭包
    let complex = |x| {
        let y = x + 1;
        y * 2
    };
}
```

### 捕获环境

```rust
fn main() {
    let x = 4;
    
    // 闭包捕获环境变量
    let equal_to_x = |z| z == x;
    
    let y = 4;
    println!("{}", equal_to_x(y));  // true
}
```

### Fn Trait

Rust 有三种闭包 trait:

1. **FnOnce** - 消费捕获的变量,只能调用一次
2. **FnMut** - 可变借用,可调用多次
3. **Fn** - 不可变借用,可调用多次

```rust
fn main() {
    // FnOnce: 获取所有权
    let s = String::from("hello");
    let consume = || {
        println!("{}", s);
        drop(s);  // 消费 s
    };
    consume();
    // consume();  // 错误:只能调用一次
    
    // FnMut: 可变借用
    let mut count = 0;
    let mut increment = || {
        count += 1;
        println!("{}", count);
    };
    increment();  // 1
    increment();  // 2
    
    // Fn: 不可变借用
    let value = String::from("hello");
    let print = || {
        println!("{}", value);
    };
    print();
    print();  // 可以多次调用
}
```

### move 关键字

```rust
use std::thread;

fn main() {
    let s = String::from("hello");
    
    // move 强制闭包获取所有权
    let handle = thread::spawn(move || {
        println!("{}", s);
    });
    
    // println!("{}", s);  // 错误: s 已被移动
    
    handle.join().unwrap();
}
```

### 闭包作为参数

```rust
fn apply<F>(f: F, x: i32) -> i32
where
    F: Fn(i32) -> i32,
{
    f(x)
}

fn main() {
    let double = |x| x * 2;
    let result = apply(double, 5);
    println!("{}", result);  // 10
}
```

### 闭包作为返回值

```rust
fn make_adder(x: i32) -> impl Fn(i32) -> i32 {
    move |y| x + y
}

fn main() {
    let add_5 = make_adder(5);
    println!("{}", add_5(10));  // 15
}
```

### 闭包与迭代器

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];
    
    // filter + map
    let result: Vec<_> = v.iter()
        .filter(|&x| x % 2 == 0)
        .map(|x| x * 2)
        .collect();
    println!("{:?}", result);  // [4, 8]
    
    // 复杂闭包
    let threshold = 3;
    let result: Vec<_> = v.iter()
        .filter(|&&x| x > threshold)
        .map(|&x| {
            if x % 2 == 0 {
                x * 2
            } else {
                x * 3
            }
        })
        .collect();
    println!("{:?}", result);  // [12, 15]
}
```

### 缓存/记忆化

```rust
use std::collections::HashMap;

struct Cacher<T>
where
    T: Fn(u32) -> u32,
{
    calculation: T,
    value: HashMap<u32, u32>,
}

impl<T> Cacher<T>
where
    T: Fn(u32) -> u32,
{
    fn new(calculation: T) -> Cacher<T> {
        Cacher {
            calculation,
            value: HashMap::new(),
        }
    }
    
    fn value(&mut self, arg: u32) -> u32 {
        match self.value.get(&arg) {
            Some(&v) => v,
            None => {
                let v = (self.calculation)(arg);
                self.value.insert(arg, v);
                v
            }
        }
    }
}

fn main() {
    let mut expensive_result = Cacher::new(|num| {
        println!("计算中...");
        std::thread::sleep(std::time::Duration::from_secs(1));
        num
    });
    
    println!("{}", expensive_result.value(1));  // 计算
    println!("{}", expensive_result.value(1));  // 使用缓存
}
```

### 实用示例

```rust
fn main() {
    // 示例1: 排序
    let mut v = vec![5, 2, 8, 1, 9];
    v.sort_by(|a, b| a.cmp(b));
    println!("{:?}", v);
    
    // 示例2: 自定义迭代处理
    let numbers = vec![1, 2, 3, 4, 5];
    let sum_of_squares: i32 = numbers
        .iter()
        .map(|&x| x * x)
        .sum();
    println!("平方和: {}", sum_of_squares);
    
    // 示例3: Option 和 Result 处理
    let maybe_number = Some(5);
    let doubled = maybe_number.map(|x| x * 2);
    
    // 示例4: 链式处理
    let text = "hello world";
    let result: String = text
        .split_whitespace()
        .map(|word| word.chars().rev().collect::<String>())
        .collect::<Vec<_>>()
        .join(" ");
    println!("{}", result);  // "olleh dlrow"
}
```

## 总结

本文介绍了 Rust 的基础语法：

- ✅ 变量和可变性
- ✅ 标量类型和复合类型
- ✅ 函数定义和调用
- ✅ 控制流语句
- ✅ 字符串基础
- ✅ 类型转换和运算符
- ✅ 迭代器:iter、map、filter、collect
- ✅ 闭包:Fn/FnMut/FnOnce、捕获环境

掌握这些基础知识后，继续学习 [所有权系统](./ownership)，这是 Rust 最重要的特性。
