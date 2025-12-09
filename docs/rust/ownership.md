---
sidebar_position: 3
title: 所有权系统
---

# 所有权系统

所有权（Ownership）是 Rust 最独特和最重要的特性，它使 Rust 无需垃圾回收器就能保证内存安全。

## 所有权规则

Rust 的所有权有三条核心规则：

1. **每个值都有一个所有者**
2. **值在任何时刻只能有一个所有者**
3. **当所有者离开作用域时，值将被丢弃**

## 作用域

```rust
fn main() {
    {                      // s 无效，尚未声明
        let s = "hello";   // s 从此处开始有效
        
        // 使用 s
        println!("{}", s);
    }                      // 作用域结束，s 不再有效
    
    // println!("{}", s);  // 错误：s 已经超出作用域
}
```

## String 类型

### String vs &str

```rust
fn main() {
    // 字符串字面量：不可变，存储在栈上
    let s1 = "hello";
    
    // String 类型：可变，存储在堆上
    let mut s2 = String::from("hello");
    s2.push_str(", world");
    println!("{}", s2);
}
```

### 内存分配

```rust
fn main() {
    let s = String::from("hello");
    
    // String 由三部分组成（存储在栈上）：
    // - ptr: 指向堆上内容的指针
    // - len: 当前长度
    // - capacity: 容量
}
```

## 移动 (Move)

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1;  // s1 的所有权移动到 s2
    
    // println!("{}", s1);  // 错误：s1 已失效
    println!("{}", s2);     // 正确
}
```

### 为什么移动？

```rust
// 如果允许 s1 和 s2 同时有效：
// 当它们离开作用域时，会尝试两次释放相同的内存
// 这就是"双重释放"错误，可能导致内存损坏

// Rust 的解决方案：移动后使前一个变量无效
```

## 克隆 (Clone)

```rust
fn main() {
    // 深拷贝：复制堆上的数据
    let s1 = String::from("hello");
    let s2 = s1.clone();
    
    println!("s1 = {}, s2 = {}", s1, s2);  // 都有效
}
```

## 复制 (Copy)

```rust
fn main() {
    // 栈上的数据：自动复制
    let x = 5;
    let y = x;
    
    println!("x = {}, y = {}", x, y);  // 都有效
}
```

### 实现了 Copy trait 的类型

- 所有整数类型（i32, u32 等）
- 布尔类型 bool
- 浮点类型（f32, f64）
- 字符类型 char
- 元组（如果元素都实现了 Copy）

```rust
fn main() {
    // 实现了 Copy
    let x = (1, 2);
    let y = x;
    println!("{:?}, {:?}", x, y);  // 都有效
    
    // 未实现 Copy（包含 String）
    let x = (String::from("hello"), 5);
    let y = x;
    // println!("{:?}", x);  // 错误：x 已失效
}
```

## 所有权与函数

### 传递值给函数

```rust
fn main() {
    let s = String::from("hello");
    takes_ownership(s);  // s 的所有权移动到函数
    // println!("{}", s);  // 错误：s 已失效
    
    let x = 5;
    makes_copy(x);   // x 被复制
    println!("{}", x);  // 正确：x 仍然有效
}

fn takes_ownership(some_string: String) {
    println!("{}", some_string);
}  // some_string 离开作用域，内存被释放

fn makes_copy(some_integer: i32) {
    println!("{}", some_integer);
}  // some_integer 离开作用域，没有特殊操作
```

### 返回值与所有权

```rust
fn main() {
    let s1 = gives_ownership();  // 获得所有权
    
    let s2 = String::from("hello");
    let s3 = takes_and_gives_back(s2);  // s2 移动，s3 获得新所有权
    
    // println!("{}", s2);  // 错误：s2 已失效
    println!("{}", s3);     // 正确
}

fn gives_ownership() -> String {
    let some_string = String::from("hello");
    some_string  // 返回，所有权移出
}

fn takes_and_gives_back(a_string: String) -> String {
    a_string  // 返回，所有权移出
}
```

## 引用和借用

引用允许使用值但不获取所有权。

### 不可变引用

```rust
fn main() {
    let s1 = String::from("hello");
    
    let len = calculate_length(&s1);  // 借用 s1
    
    println!("'{}' 的长度是 {}", s1, len);  // s1 仍然有效
}

fn calculate_length(s: &String) -> usize {
    s.len()
}  // s 离开作用域，但不会释放内存（因为没有所有权）
```

### 可变引用

```rust
fn main() {
    let mut s = String::from("hello");
    
    change(&mut s);  // 可变借用
    
    println!("{}", s);  // "hello, world"
}

fn change(some_string: &mut String) {
    some_string.push_str(", world");
}
```

### 借用规则

```rust
fn main() {
    let mut s = String::from("hello");
    
    // 规则 1：同一作用域内，只能有一个可变引用
    let r1 = &mut s;
    // let r2 = &mut s;  // 错误：不能同时有两个可变引用
    println!("{}", r1);
    
    // 规则 2：不能同时有可变引用和不可变引用
    let r1 = &s;     // 可以
    let r2 = &s;     // 可以
    // let r3 = &mut s;  // 错误：已有不可变引用
    println!("{} and {}", r1, r2);
    
    // 注意：引用的作用域从声明到最后一次使用
    let r1 = &s;
    println!("{}", r1);  // r1 的作用域到此结束
    
    let r2 = &mut s;  // 正确：r1 已不再使用
    println!("{}", r2);
}
```

### 悬垂引用

```rust
fn main() {
    // let reference_to_nothing = dangle();  // 错误：悬垂引用
    let s = no_dangle();  // 正确：返回所有权
}

// 错误示例：返回悬垂引用
// fn dangle() -> &String {
//     let s = String::from("hello");
//     &s  // 返回 s 的引用，但 s 将被释放
// }

// 正确示例：返回所有权
fn no_dangle() -> String {
    let s = String::from("hello");
    s  // 返回 s 本身，所有权移出
}
```

## 切片 (Slice)

切片是对集合中连续序列的引用。

### 字符串切片

```rust
fn main() {
    let s = String::from("hello world");
    
    let hello = &s[0..5];   // "hello"
    let world = &s[6..11];  // "world"
    
    // 语法糖
    let hello = &s[..5];    // 从开头
    let world = &s[6..];    // 到结尾
    let whole = &s[..];     // 整个字符串
    
    println!("{} {}", hello, world);
}
```

### 字符串切片作为参数

```rust
fn main() {
    let my_string = String::from("hello world");
    
    // 接受 String 的切片
    let word = first_word(&my_string[..]);
    
    let my_string_literal = "hello world";
    
    // 接受字符串字面量的切片
    let word = first_word(&my_string_literal[..]);
    
    // 字符串字面量本身就是切片
    let word = first_word(my_string_literal);
}

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

### 数组切片

```rust
fn main() {
    let a = [1, 2, 3, 4, 5];
    
    let slice = &a[1..3];  // [2, 3]
    
    assert_eq!(slice, &[2, 3]);
}
```

## 所有权模式

### 模式 1：使用引用避免所有权转移

```rust
fn main() {
    let s = String::from("hello");
    
    // 不好：所有权转移
    let len = calculate_length_bad(s);
    // println!("{}", s);  // 错误：s 已失效
    
    let s = String::from("hello");
    
    // 好：使用引用
    let len = calculate_length_good(&s);
    println!("{}", s);  // 正确：s 仍然有效
}

fn calculate_length_bad(s: String) -> usize {
    s.len()
}

fn calculate_length_good(s: &String) -> usize {
    s.len()
}
```

### 模式 2：返回多个值

```rust
fn main() {
    let s1 = String::from("hello");
    
    // 不好：返回元组
    let (s2, len) = calculate_length_tuple(s1);
    
    // 好：使用引用
    let s1 = String::from("hello");
    let len = calculate_length(&s1);
    println!("'{}' 的长度是 {}", s1, len);
}

fn calculate_length_tuple(s: String) -> (String, usize) {
    let length = s.len();
    (s, length)
}

fn calculate_length(s: &String) -> usize {
    s.len()
}
```

### 模式 3：修改数据

```rust
fn main() {
    let mut s = String::from("hello");
    
    // 使用可变引用修改数据
    append_world(&mut s);
    
    println!("{}", s);  // "hello, world"
}

fn append_world(s: &mut String) {
    s.push_str(", world");
}
```

## 最佳实践

### 1. 优先使用引用

```rust
// 不好：unnecessary ownership transfer
fn process(data: String) {
    println!("{}", data);
}

// 好：使用引用
fn process(data: &str) {
    println!("{}", data);
}
```

### 2. 保持借用作用域最小

```rust
fn main() {
    let mut s = String::from("hello");
    
    {
        let r = &mut s;
        r.push_str(", world");
    }  // r 离开作用域
    
    // 现在可以再次借用
    let r2 = &s;
    println!("{}", r2);
}
```

### 3. 使用 &str 而非 &String

```rust
// 不好
fn first_word(s: &String) -> &str {
    // ...
    &s[..]
}

// 好：更灵活
fn first_word(s: &str) -> &str {
    // 可以接受 &String 和 &str
    &s[..]
}
```

## 常见错误

### 错误 1：使用已移动的值

```rust
fn main() {
    let s = String::from("hello");
    let s2 = s;
    // println!("{}", s);  // 错误：s 已失效
}
```

### 错误 2：悬垂引用

```rust
// fn dangle() -> &String {
//     let s = String::from("hello");
//     &s  // 错误：返回悬垂引用
// }
```

### 错误 3：多个可变引用

```rust
fn main() {
    let mut s = String::from("hello");
    
    let r1 = &mut s;
    // let r2 = &mut s;  // 错误：已有可变引用
    println!("{}", r1);
}
```

## 内存布局详解

### 栈和堆

```rust
fn main() {
    // 栈上分配:固定大小,快速访问
    let x = 5;              // i32: 4字节,栈上
    let y = true;           // bool: 1字节,栈上
    let z = 3.14;           // f64: 8字节,栈上
    
    // 堆上分配:动态大小,较慢但灵活
    let s = String::from("hello");  // 指针在栈,数据在堆
    let v = vec![1, 2, 3];          // 指针在栈,数据在堆
}
```

### String 的内存布局

```rust
fn main() {
    let s = String::from("hello");
    
    // String 在栈上存储三个值(共24字节):
    // - ptr: 指向堆数据的指针 (8字节)
    // - len: 当前长度 (8字节)
    // - capacity: 容量 (8字节)
    
    // 堆上存储实际字符串数据: "hello"
}
```

### 移动的内存效果

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1;  // 浅拷贝栈数据,s1失效
    
    // 内存布局:
    // 栈:
    //   s1: [失效]
    //   s2: [ptr | len | cap] -> 堆数据
    // 堆:
    //   "hello"
    
    // 只有一个所有者,避免双重释放!
}
```

### 克隆的内存效果

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1.clone();  // 深拷贝
    
    // 内存布局:
    // 栈:
    //   s1: [ptr1 | len | cap] -> 堆数据1
    //   s2: [ptr2 | len | cap] -> 堆数据2
    // 堆:
    //   "hello" (s1的数据)
    //   "hello" (s2的数据,独立拷贝)
}
```

### Copy 类型的内存

```rust
fn main() {
    let x = 5;
    let y = x;  // 按位复制,都有效
    
    // 内存布局(栈):
    //   x: 5
    //   y: 5
    // 简单的按位复制,无堆分配
}
```

## 生命周期预览

### 编译器如何跟踪生命周期

```rust
fn main() {
    let r;                    // -------+-- 'a
                              //        |
    {                         //        |
        let x = 5;            // -+-- 'b|
        r = &x;               //  |     |
    }                         // -+     |
                              //        |
    // println!("{}", r);     // 错误   |
}                            // -------+

// 'b < 'a: x的生命周期小于r的期望生命周期
```

### 函数返回引用

```rust
// 需要生命周期注解
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

fn main() {
    let string1 = String::from("long string");
    let string2 = String::from("xyz");
    
    let result = longest(string1.as_str(), string2.as_str());
    println!("{}", result);
}
```

### 结构体中的引用

```rust
// 结构体持有引用需要生命周期注解
struct ImportantExcerpt<'a> {
    part: &'a str,
}

fn main() {
    let novel = String::from("Call me Ishmael.");
    let first_sentence = novel.split('.').next().unwrap();
    
    let excerpt = ImportantExcerpt {
        part: first_sentence,
    };
    
    println!("{}", excerpt.part);
}
```

## 实战场景

### 场景1:解析器状态

```rust
struct Parser<'a> {
    input: &'a str,
    position: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Parser { input, position: 0 }
    }
    
    fn current_char(&self) -> Option<char> {
        self.input.chars().nth(self.position)
    }
    
    fn advance(&mut self) {
        self.position += 1;
    }
    
    fn parse_word(&mut self) -> Option<&'a str> {
        let start = self.position;
        
        while let Some(c) = self.current_char() {
            if c.is_whitespace() {
                break;
            }
            self.advance();
        }
        
        if start < self.position {
            Some(&self.input[start..self.position])
        } else {
            None
        }
    }
}

fn main() {
    let text = "hello world rust";
    let mut parser = Parser::new(text);
    
    while let Some(word) = parser.parse_word() {
        println!("{}", word);
        parser.advance(); // 跳过空格
    }
}
```

### 场景2:缓存系统

```rust
use std::collections::HashMap;

struct Cache {
    data: HashMap<String, String>,
}

impl Cache {
    fn new() -> Self {
        Cache {
            data: HashMap::new(),
        }
    }
    
    // 返回引用避免克隆
    fn get(&self, key: &str) -> Option<&String> {
        self.data.get(key)
    }
    
    fn insert(&mut self, key: String, value: String) {
        self.data.insert(key, value);
    }
}

fn main() {
    let mut cache = Cache::new();
    cache.insert("name".to_string(), "Alice".to_string());
    
    // 借用而非克隆
    if let Some(value) = cache.get("name") {
        println!("Found: {}", value);
    }
}
```

### 场景3:数据流处理

```rust
fn process_data(data: &[i32]) -> Vec<i32> {
    data.iter()
        .filter(|&&x| x > 0)
        .map(|&x| x * 2)
        .collect()
}

fn main() {
    let numbers = vec![1, -2, 3, -4, 5];
    
    // 传递引用,避免所有权转移
    let result = process_data(&numbers);
    
    println!("原始: {:?}", numbers);
    println!("结果: {:?}", result);
}
```

## 性能优化建议

### 1. 避免不必要的克隆

```rust
// 不好:频繁克隆
fn bad_example(data: Vec<String>) -> Vec<String> {
    let mut result = Vec::new();
    for item in data {
        let cloned = item.clone();
        result.push(cloned);
    }
    result
}

// 好:直接使用所有权
fn good_example(data: Vec<String>) -> Vec<String> {
    data
}

// 或者使用引用
fn reference_example(data: &[String]) -> Vec<String> {
    data.to_vec()  // 只在需要时克隆
}
```

### 2. 使用 Cow 优化写时复制

```rust
use std::borrow::Cow;

fn process<'a>(input: &'a str) -> Cow<'a, str> {
    if input.contains("bad") {
        // 需要修改:返回拥有的
        Cow::Owned(input.replace("bad", "good"))
    } else {
        // 不需要修改:返回借用
        Cow::Borrowed(input)
    }
}

fn main() {
    let s1 = "hello bad world";
    let s2 = "hello world";
    
    println!("{}", process(s1));  // Owned
    println!("{}", process(s2));  // Borrowed (无分配)
}
```

### 3. 使用切片而非完整集合

```rust
// 不好:传递整个Vec
fn sum_bad(data: Vec<i32>) -> i32 {
    data.iter().sum()
}

// 好:传递切片
fn sum_good(data: &[i32]) -> i32 {
    data.iter().sum()
}

fn main() {
    let numbers = vec![1, 2, 3, 4, 5];
    
    // 可以传递Vec、数组、切片等
    println!("{}", sum_good(&numbers));
    println!("{}", sum_good(&[1, 2, 3]));
}
```

### 4. 预分配容量

```rust
fn main() {
    // 不好:多次重新分配
    let mut v = Vec::new();
    for i in 0..1000 {
        v.push(i);
    }
    
    // 好:预分配容量
    let mut v = Vec::with_capacity(1000);
    for i in 0..1000 {
        v.push(i);
    }
}
```

### 5. 使用 &str 而非 String

```rust
// 不好:不必要的分配
fn greet_bad(name: String) {
    println!("Hello, {}!", name);
}

// 好:使用字符串切片
fn greet_good(name: &str) {
    println!("Hello, {}!", name);
}

fn main() {
    let name = String::from("Alice");
    greet_good(&name);  // 更灵活
    greet_good("Bob");  // 也可以接受字面量
}
```

### 6. 避免运行时借用检查

```rust
use std::cell::RefCell;

// 不好:运行时检查(有开销)
fn bad_pattern() {
    let x = RefCell::new(5);
    *x.borrow_mut() += 1;
}

// 好:编译时检查(零开销)
fn good_pattern() {
    let mut x = 5;
    x += 1;
}
```

### 7. 返回迭代器而非Vec

```rust
// 不好:立即分配
fn get_evens_bad(data: &[i32]) -> Vec<i32> {
    data.iter()
        .filter(|&&x| x % 2 == 0)
        .copied()
        .collect()
}

// 好:惰性求值
fn get_evens_good(data: &[i32]) -> impl Iterator<Item = i32> + '_ {
    data.iter()
        .filter(|&&x| x % 2 == 0)
        .copied()
}

fn main() {
    let numbers = vec![1, 2, 3, 4, 5, 6];
    
    // 只在需要时分配
    let evens: Vec<_> = get_evens_good(&numbers).collect();
}
```

### 8. 使用引用优化循环

```rust
fn main() {
    let strings = vec![
        String::from("hello"),
        String::from("world"),
    ];
    
    // 不好:每次迭代都克隆
    for s in strings.clone() {
        println!("{}", s);
    }
    
    // 好:使用引用
    for s in &strings {
        println!("{}", s);
    }
    
    // strings 仍然有效
    println!("{:?}", strings);
}
```

## 总结

本文介绍了 Rust 的所有权系统：

- ✅ 所有权三大规则
- ✅ 移动、克隆和复制
- ✅ 所有权与函数
- ✅ 引用和借用
- ✅ 可变引用和不可变引用
- ✅ 切片类型
- ✅ 内存布局详解(栈/堆、String布局)
- ✅ 生命周期预览(编译器跟踪、函数返回引用)
- ✅ 实战场景(解析器、缓存系统、数据流)
- ✅ 性能优化建议(Cow、切片、迭代器等8条建议)

掌握所有权后，继续学习 [结构体和枚举](./structs-enums)。
