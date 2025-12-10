---
sidebar_position: 9
title: 生命周期
---

# 生命周期

生命周期是 Rust 用于防止悬垂引用的机制,确保引用始终有效。每个引用都有生命周期,大多数情况下是隐式的。

## 生命周期基础

### 什么是生命周期

生命周期是引用保持有效的作用域。

```rust
fn main() {
    let r;                // ---------+-- 'a
                          //          |
    {                     //          |
        let x = 5;        // -+-- 'b  |
        r = &x;           //  |       |
    }                     // -+       |
                          //          |
    // println!("{}", r); // 错误: x已离开作用域 |
}                        // ---------+
```

### 编译器如何检查

Rust 编译器有一个**借用检查器**,比较作用域来确保所有借用都有效:

```rust
fn main() {
    let x = 5;            // ----------+-- 'b
                          //           |
    let r = &x;           // --+-- 'a  |
                          //   |       |
    println!("r: {}", r); //   |       |
                          // --+       |
}                         // ----------+
```

## 函数中的生命周期

### 为什么需要生命周期注解

```rust
// 编译器无法推断返回值的生命周期
// fn longest(x: &str, y: &str) -> &str {
//     if x.len() > y.len() {
//         x
//     } else {
//         y
//     }
// }
```

### 生命周期注解语法

```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

fn main() {
    let string1 = String::from("long string is long");
    
    {
        let string2 = String::from("xyz");
        let result = longest(string1.as_str(), string2.as_str());
        println!("最长的字符串是 {}", result);
    }
}
```

### 生命周期注解语法规则

- 生命周期参数名称必须以撇号(`'`)开头
- 通常使用小写字母,如 `'a`, `'b`, `'c`
- 生命周期注解位于引用的 `&` 之后

```rust
&i32        // 引用
&'a i32     // 带有显式生命周期的引用
&'a mut i32 // 带有显式生命周期的可变引用
```

### 理解生命周期注解

```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    // 'a 的实际生命周期是 x 和 y 生命周期中较小的那一个
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

fn main() {
    let string1 = String::from("long string");
    let result;
    
    {
        let string2 = String::from("xyz");
        result = longest(string1.as_str(), string2.as_str());
        println!("{}", result);  // 正确: result 在 string2 离开作用域前使用
    }
    
    // println!("{}", result);  // 错误: result 引用了 string2
}
```

## 深入理解生命周期

### 不需要返回引用的情况

```rust
// 直接返回参数,不需要生命周期注解
fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    
    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }
    
    &s[..]
}
```

### 不同生命周期参数

```rust
// x 和 y 可以有不同的生命周期
fn longest<'a, 'b>(x: &'a str, y: &'b str) -> &'a str {
    x  // 只返回 x,所以只需要 'a
}

fn main() {
    let string1 = String::from("long string");
    
    {
        let string2 = String::from("xyz");
        let result = longest(string1.as_str(), string2.as_str());
显示字节
        // result 的生命周期只依赖于 string1
    }
}
```

### 生命周期约束

```rust
// 指定 'b 至少要活得和 'a 一样久
fn example<'a, 'b: 'a>(x: &'a i32, y: &'b i32) -> &'a i32 {
    if x > y { x } else { y }
}
```

## 结构体中的生命周期

### 定义包含引用的结构体

```rust
struct ImportantExcerpt<'a> {
    part: &'a str,
}

fn main() {
    let novel = String::from("Call me Ishmael. Some years ago...");
    let first_sentence = novel.split('.').next().expect("Could not find a '.'");
    
    let i = ImportantExcerpt {
        part: first_sentence,
    };
    
    println!("重要摘录: {}", i.part);
}
```

### 结构体方法中的生命周期

```rust
struct ImportantExcerpt<'a> {
    part: &'a str,
}

impl<'a> ImportantExcerpt<'a> {
    // 方法的生命周期注解
    fn level(&self) -> i32 {
        3
    }
    
    fn announce_and_return_part(&self, announcement: &str) -> &str {
        println!("注意: {}", announcement);
        self.part
    }
}

fn main() {
    let novel = String::from("Call me Ishmael.");
    let first_sentence = novel.split('.').next().unwrap();
    
    let excerpt = ImportantExcerpt {
        part: first_sentence,
    };
    
    println!("级别: {}", excerpt.level());
    excerpt.announce_and_return_part("重要!");
}
```

## 生命周期省略规则

编译器使用三条规则自动推断生命周期:

### 规则 1: 每个引用参数都有自己的生命周期

```rust
// 编译器自动推断
fn first_word(s: &str) -> &str { ... }

// 等价于
fn first_word<'a>(s: &'a str) -> &'a str { ... }
```

### 规则 2: 如果只有一个输入生命周期参数,它会被赋予所有输出生命周期参数

```rust
fn example(s: &str) -> &str { ... }

// 等价于
fn example<'a>(s: &'a str) -> &'a str { ... }
```

### 规则 3: 如果有多个输入生命周期参数,且其中一个是 `&self` 或 `&mut self`,则 `self` 的生命周期被赋予所有输出生命周期参数

```rust
impl<'a> ImportantExcerpt<'a> {
    fn announce_and_return_part(&self, announcement: &str) -> &str {
        self.part
    }
}

// 编译器推断为
impl<'a> ImportantExcerpt<'a> {
    fn announce_and_return_part<'b>(&'a self, announcement: &'b str) -> &'a str {
        self.part
    }
}
```

## 静态生命周期

### 'static 生命周期

`'static` 表示整个程序期间都有效:

```rust
// 字符串字面量都有 'static 生命周期
let s: &'static str = "I have a static lifetime.";

fn main() {
    println!("{}", s);
}
```

### 谨慎使用 'static

```rust
// 不好: 滥用 'static
fn bad_example() -> &'static str {
    let s = String::from("hello");
    // &s  // 错误: s 不是 'static
    
    // 被迫使用 Box::leak
    Box::leak(s.into_boxed_str())
}

// 好: 使用适当的生命周期
fn good_example<'a>(s: &'a str) -> &'a str {
    s
}
```

## 生命周期边界

### Trait 对象的生命周期

```rust
use std::fmt::Display;

fn longest_with_an_announcement<'a, T>(
    x: &'a str,
    y: &'a str,
    ann: T,
) -> &'a str
where
    T: Display,
{
    println!("公告! {}", ann);
    if x.len() > y.len() {
        x
    } else {
        y
    }
}
```

### 生命周期 + Trait Bound

```rust
use std::fmt::Display;

fn example<'a, T: Display + 'a>(x: &'a T) -> &'a T {
    println!("{}", x);
    x
}
```

## 高级生命周期

### 生命周期子类型

```rust
// 'b 比 'a 活得更久
fn parse<'a, 'b: 'a>(context: &'b str, s: &'a str) -> &'a str {
    s
}
```

### 多个生命周期参数

```rust
struct Context<'a>(&'a str);

struct Parser<'a, 'b> {
    context: &'a Context<'b>,
}

impl<'a, 'b> Parser<'a, 'b> {
    fn parse(&self) -> &'b str {
        self.context.0
    }
}
```

### Higher-Ranked Trait Bounds (HRTB)

```rust
// for<'a> 表示对所有生命周期 'a
fn apply<F>(f: F)
where
    F: for<'a> Fn(&'a i32),
{
    let value = 42;
    f(&value);
}

fn main() {
    apply(|x| println!("{}", x));
}
```

## 实战示例

### 示例1: 字符串分割器

```rust
struct StrSplit<'a, 'b> {
    remainder: Option<&'a str>,
    delimiter: &'b str,
}

impl<'a, 'b> StrSplit<'a, 'b> {
    fn new(haystack: &'a str, delimiter: &'b str) -> Self {
        Self {
            remainder: Some(haystack),
            delimiter,
        }
    }
}

impl<'a> Iterator for StrSplit<'a, '_> {
    type Item = &'a str;
    
    fn next(&mut self) -> Option<Self::Item> {
        let remainder = self.remainder.as_mut()?;
        
        if let Some(next_delim) =remainder.find(self.delimiter) {
            let until_delimiter = &remainder[..next_delim];
            *remainder = &remainder[next_delim + self.delimiter.len()..];
            Some(until_delimiter)
        } else {
            self.remainder.take()
        }
    }
}

fn main() {
    let haystack = "a b c d e";
    let letters: Vec<_> = StrSplit::new(haystack, " ").collect();
    println!("{:?}", letters);
}
```

### 示例2: 配置解析器

```rust
struct Config<'a> {
    host: &'a str,
    port: u16,
}

impl<'a> Config<'a> {
    fn new(host: &'a str, port: u16) -> Self {
        Config { host, port }
    }
    
    fn url(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

fn main() {
    let host = String::from("localhost");
    let config = Config::new(&host, 8080);
    println!("URL: {}", config.url());
}
```

## 常见模式

### 模式1: 返回输入引用

```rust
fn first<'a>(items: &'a [i32]) -> Option<&'a i32> {
    items.first()
}
```

### 模式2: 结构体持有引用

```rust
struct Buffer<'a> {
    data: &'a [u8],
}

impl<'a> Buffer<'a> {
    fn new(data: &'a [u8]) -> Self {
        Buffer { data }
    }
}
```

### 模式3: 迭代器

```rust
struct Iter<'a, T> {
    items: &'a [T],
    index: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.items.len() {
            let item = &self.items[self.index];
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }
}
```

## 最佳实践

### 1. 优先让编译器推断

```rust
// 好: 让编译器推断
fn example(s: &str) -> &str {
    s
}

// 避免: 不必要的显式注解
fn example<'a>(s: &'a str) -> &'a str {
    s
}
```

### 2. 使用有意义的生命周期名称

```rust
// 好: 描述性名称(在复杂情况下)
struct Parser<'input, 'config> {
    input: &'input str,
    config: &'config Config,
}

// 避免: 过多的泛型名称
struct Parser<'a, 'b, 'c, 'd> {
    // ...
}
```

### 3. 避免不必要的 'static

```rust
// 不好
fn bad(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())  // 内存泄漏!
}

// 好
fn good(s: &str) -> &str {
    s
}
```

## 常见错误

### 错误1: 返回局部变量的引用

```rust
// 错误: 返回悬垂引用
// fn dangle() -> &str {
//     let s = String::from("hello");
//     &s
// }

// 正确: 返回所有权
fn no_dangle() -> String {
    let s = String::from("hello");
    s
}
```

### 错误2: 生命周期不匹配

```rust
// 错误
// fn example<'a>(x: &'a str) -> &'a str {
//     let s = String::from("hello");
//     &s  // s 的生命周期不够长
// }
```

### 错误3: 过度指定生命周期

```rust
// 过度指定
fn longest<'a, 'b>(x: &'a str, y: &'b str) -> &'a str
where
    'b: 'a,
{
    x
}

// 更简单
fn longest<'a>(x: &'a str, y: &str) -> &'a str {
    x
}
```

## 总结

本文详细介绍了 Rust 的生命周期:

- ✅ 生命周期基础概念
- ✅ 函数和结构体中的生命周期注解
- ✅ 生命周期省略规则
- ✅ 静态生命周期 'static
- ✅ 生命周期边界
- ✅ 高级生命周期特性
- ✅ 实战示例和常见模式
- ✅ 最佳实践和常见错误

**关键要点:**

1. 生命周期确保引用始终有效
2. 大多数情况下编译器可以推断生命周期
3. 生命周期注解描述引用之间的关系
4. 'static 表示整个程序期间有效
5. 避免不必要的显式生命周期注解

掌握生命周期后,继续学习 [模块和包管理](/docs/rust/modules-packages)。
