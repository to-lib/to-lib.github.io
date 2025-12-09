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

## 总结

本文介绍了 Rust 的基础语法：

- ✅ 变量和可变性
- ✅ 标量类型和复合类型
- ✅ 函数定义和调用
- ✅ 控制流语句
- ✅ 字符串基础
- ✅ 类型转换和运算符

掌握这些基础知识后，继续学习 [所有权系统](./ownership)，这是 Rust 最重要的特性。
