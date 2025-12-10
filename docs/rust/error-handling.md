---
sidebar_position: 5
title: 错误处理
---

# 错误处理

Rust 将错误分为两大类：可恢复错误和不可恢复错误。本文介绍 `Result<T, E>` 和 `panic!` 的使用。

## 不可恢复错误 panic

### 触发 panic

```rust
fn main() {
    panic!("程序崩溃了！");
}
```

### 数组越界导致 panic

```rust
fn main() {
    let v = vec![1, 2, 3];
    
    v[99];  // panic：索引越界
}
```

### RUST_BACKTRACE 环境变量

```bash
# 查看完整的错误回溯
RUST_BACKTRACE=1 cargo run
RUST_BACKTRACE=full cargo run
```

## 可恢复错误 `Result<T, E>`

### Result 枚举

```rust
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

### 处理 Result

```rust
use std::fs::File;

fn main() {
    let f = File::open("hello.txt");
    
    let f = match f {
        Ok(file) => file,
        Err(error) => {
            panic!("打开文件失败: {:?}", error);
        }
    };
}
```

### 匹配不同的错误类型

```rust
use std::fs::File;
use std::io::ErrorKind;

fn main() {
    let f = File::open("hello.txt");
    
    let f = match f {
        Ok(file) => file,
        Err(error) => match error.kind() {
            ErrorKind::NotFound => match File::create("hello.txt") {
                Ok(fc) => fc,
                Err(e) => panic!("创建文件失败: {:?}", e),
            },
            other_error => {
                panic!("打开文件失败: {:?}", other_error);
            }
        },
    };
}
```

### 使用闭包简化

```rust
use std::fs::File;
use std::io::ErrorKind;

fn main() {
    let f = File::open("hello.txt").unwrap_or_else(|error| {
        if error.kind() == ErrorKind::NotFound {
            File::create("hello.txt").unwrap_or_else(|error| {
                panic!("创建文件失败: {:?}", error);
            })
        } else {
            panic!("打开文件失败: {:?}", error);
        }
    });
}
```

## 简化错误处理

### unwrap

```rust
use std::fs::File;

fn main() {
    // 如果 Result 是 Ok，返回值
    // 如果是 Err，调用 panic!
    let f = File::open("hello.txt").unwrap();
}
```

### expect

```rust
use std::fs::File;

fn main() {
    // 类似 unwrap，但可以自定义 panic! 消息
    let f = File::open("hello.txt")
        .expect("无法打开 hello.txt");
}
```

## 传播错误

### 手动传播

```rust
use std::fs::File;
use std::io::{self, Read};

fn read_username_from_file() -> Result<String, io::Error> {
    let f = File::open("hello.txt");
    
    let mut f = match f {
        Ok(file) => file,
        Err(e) => return Err(e),  // 提前返回错误
    };
    
    let mut s = String::new();
    
    match f.read_to_string(&mut s) {
        Ok(_) => Ok(s),
        Err(e) => Err(e),
    }
}
```

### ? 运算符

```rust
use std::fs::File;
use std::io::{self, Read};

fn read_username_from_file() -> Result<String, io::Error> {
    let mut f = File::open("hello.txt")?;  // 如果出错，提前返回
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    Ok(s)
}
```

### 链式调用 ?

```rust
use std::fs::File;
use std::io::{self, Read};

fn read_username_from_file() -> Result<String, io::Error> {
    let mut s = String::new();
    File::open("hello.txt")?.read_to_string(&mut s)?;
    Ok(s)
}
```

### 更简洁的方式

```rust
use std::fs;
use std::io;

fn read_username_from_file() -> Result<String, io::Error> {
    fs::read_to_string("hello.txt")
}
```

## ? 运算符的规则

### 只能用于返回 Result 的函数

```rust
use std::fs::File;

fn main() {
    // 错误：main 不返回 Result
    // let f = File::open("hello.txt")?;
}

// 正确：返回 Result
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let f = File::open("hello.txt")?;
    Ok(())
}
```

### Option 也可以使用 ?

```rust
fn last_char_of_first_line(text: &str) -> Option<char> {
    text.lines().next()?.chars().last()
}

fn main() {
    assert_eq!(
        last_char_of_first_line("Hello, world\nHow are you today?"),
        Some('d')
    );
    
    assert_eq!(last_char_of_first_line(""), None);
}
```

## 自定义错误类型

```rust
use std::fmt;

#[derive(Debug)]
enum MyError {
    IoError(std::io::Error),
    ParseError(std::num::ParseIntError),
    CustomError(String),
}

impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MyError::IoError(e) => write!(f, "IO错误: {}", e),
            MyError::ParseError(e) => write!(f, "解析错误: {}", e),
            MyError::CustomError(msg) => write!(f, "自定义错误: {}", msg),
        }
    }
}

impl std::error::Error for MyError {}

// 实现 From trait 以支持 ? 运算符
impl From<std::io::Error> for MyError {
    fn from(error: std::io::Error) -> Self {
        MyError::IoError(error)
    }
}

impl From<std::num::ParseIntError> for MyError {
    fn from(error: std::num::ParseIntError) -> Self {
        MyError::ParseError(error)
    }
}

fn process_file(filename: &str) -> Result<i32, MyError> {
    let content = std::fs::read_to_string(filename)?;  // 自动转换为 MyError
    let number: i32 = content.trim().parse()?;         // 自动转换为 MyError
    Ok(number)
}
```

## panic! 还是 Result?

### 何时使用 panic

```rust
// 示例代码和原型代码
fn demo() {
    let v = vec![1, 2, 3];
    v[0];  // 可以使用 unwrap
}

// 你比编译器知道更多信息
use std::net::IpAddr;

fn main() {
    let home: IpAddr = "127.0.0.1".parse().unwrap();  // 确定不会失败
}
```

### 何时使用 Result

```rust
// 可恢复的错误
fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err(String::from("除数不能为0"))
    } else {
        Ok(a / b)
    }
}

// 库代码（不应该 panic）
pub fn parse_config(input: &str) -> Result<Config, ConfigError> {
    // 返回 Result 让调用者决定如何处理
    Ok(Config {})
}
```

## 创建验证类型

```rust
pub struct Guess {
    value: i32,
}

impl Guess {
    pub fn new(value: i32) -> Guess {
        if value < 1 || value > 100 {
            panic!("Guess value must be between 1 and 100, got {}.", value);
        }
        
        Guess { value }
    }
    
    pub fn value(&self) -> i32 {
        self.value
    }
}

fn main() {
    let guess = Guess::new(50);
    println!("猜测的数字: {}", guess.value());
}
```

## 实用模式

### 使用 unwrap_or

```rust
fn main() {
    let value: Result<i32, _> = "42".parse();
    let number = value.unwrap_or(0);  // 失败时返回默认值
    
    let value: Option<i32> = None;
    let number = value.unwrap_or(0);
}
```

### 使用 unwrap_or_else

```rust
fn main() {
    let value: Result<i32, _> = "not a number".parse();
    let number = value.unwrap_or_else(|_| {
        println!("解析失败，使用默认值");
        0
    });
}
```

### 使用 and_then

```rust
fn parse_and_double(s: &str) -> Result<i32, std::num::ParseIntError> {
    s.parse::<i32>()
        .and_then(|n| Ok(n * 2))
}

fn main() {
    assert_eq!(parse_and_double("10"), Ok(20));
}
```

### 使用 or_else

```rust
fn main() {
    let result1: Result<i32, &str> = Err("first error");
    let result2 = result1.or_else(|_| Ok(42));
    
    assert_eq!(result2, Ok(42));
}
```

### 使用 map 和 map_err

```rust
fn main() {
    // map：转换 Ok 值
    let result: Result<i32, &str> = Ok(5);
    let doubled = result.map(|x| x * 2);
    assert_eq!(doubled, Ok(10));
    
    // map_err：转换 Err 值
    let result: Result<i32, &str> = Err("error");
    let custom_error = result.map_err(|e| format!("Error: {}", e));
}
```

## 最佳实践

### 1. 在函数签名中明确错误类型

```rust
// 好
fn process_data(input: &str) -> Result<Data, ProcessError> {
    // ...
    Ok(Data {})
}

// 不够好：使用 Box<dyn Error>
fn process_data(input: &str) -> Result<Data, Box<dyn std::error::Error>> {
    // ...
    Ok(Data {})
}
```

### 2. 使用 ? 运算符简化代码

```rust
// 不好
fn read_file() -> Result<String, std::io::Error> {
    let f = File::open("file.txt");
    let mut f = match f {
        Ok(file) => file,
        Err(e) => return Err(e),
    };
    // ...
    Ok(String::new())
}

// 好
fn read_file() -> Result<String, std::io::Error> {
    let mut f = File::open("file.txt")?;
    // ...
    Ok(String::new())
}
```

### 3. 为错误提供上下文

```rust
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let f = File::open("config.json")
        .expect("无法打开配置文件 config.json");
    
    // 或使用 context（需要额外的 crate）
    Ok(())
}
```

## 常用错误处理 Crate

### anyhow

```rust
use anyhow::{Context, Result};

fn main() -> Result<()> {
    let content = std::fs::read_to_string("config.json")
        .context("无法读取配置文件")?;
    
    Ok(())
}
```

### thiserror

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DataStoreError {
    #[error("数据存储断开连接")]
    Disconnect(#[from] std::io::Error),
    
    #[error("无效的数据头 (expected {expected:?}, found {found:?})")]
    InvalidHeader {
        expected: String,
        found: String,
    },
    
    #[error("未知错误")]
    Unknown,
}
```

## 错误处理设计模式

### 错误链(Error Chain)

```rust
use std::error::Error;
use std::fmt;

#[derive(Debug)]
struct DatabaseError {
    message: String,
    source: Option<Box<dyn Error>>,
}

impl fmt::Display for DatabaseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "数据库错误: {}", self.message)
    }
}

impl Error for DatabaseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_ref().map(|e| e.as_ref())
    }
}

impl From<std::io::Error> for DatabaseError {
    fn from(error: std::io::Error) -> Self {
        DatabaseError {
            message: "IO操作失败".to_string(),
            source: Some(Box::new(error)),
        }
    }
}
```

### Context 模式

```rust
trait Context<T, E> {
    fn context<C>(self, context: C) -> Result<T, String>
    where
        C: fmt::Display;
}

impl<T, E: fmt::Display> Context<T, E> for Result<T, E> {
    fn context<C>(self, context: C) -> Result<T, String>
    where
        C: fmt::Display,
    {
        self.map_err(|e| format!("{}: {}", context, e))
    }
}

fn read_config() -> Result<String, String> {
    std::fs::read_to_string("config.toml")
        .map_err(|e| e.to_string())
        .context("无法读取配置文件")
}
```

### Early Return 模式

```rust
fn process_data() -> Result<(), Box<dyn Error>> {
    let file = std::fs::File::open("data.txt")?;
    let content = std::fs::read_to_string("data.txt")?;
    let number: i32 = content.trim().parse()?;
    
    println!("数字: {}", number);
    Ok(())
}
```

## 错误上下文处理

### 使用 anyhow crate

```rust
use anyhow::{Context, Result};

fn read_username() -> Result<String> {
    let path = "username.txt";
    std::fs::read_to_string(path)
        .with_context(|| format!("无法读取文件 {}", path))
}

fn main() -> Result<()> {
    let username = read_username()?;
    println!("用户名: {}", username);
    Ok(())
}
```

### 自定义上下文

```rust
#[derive(Debug)]
struct AppError {
    context: String,
    source: Box<dyn std::error::Error>,
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}\n原因: {}", self.context, self.source)
    }
}

impl std::error::Error for AppError {}

fn with_context<T, E>(result: Result<T, E>, context: &str) -> Result<T, AppError>
where
    E: std::error::Error + 'static,
{
    result.map_err(|e| AppError {
        context: context.to_string(),
        source: Box::new(e),
    })
}
```

## 错误恢复策略

### 重试模式

```rust
fn retry<F, T, E>(mut f: F, max_attempts: u32) -> Result<T, E>
where
    F: FnMut() -> Result<T, E>,
{
    let mut attempts = 0;
    loop {
        match f() {
            Ok(result) => return Ok(result),
            Err(e) => {
                attempts += 1;
                if attempts >= max_attempts {
                    return Err(e);
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    }
}

fn main() {
    let result = retry(
        || {
            // 模拟不稳定的操作
            if rand::random::<bool>() {
                Ok("成功")
            } else {
                Err("失败")
            }
        },
        3,
    );
}
```

### 降级模式

```rust
fn get_user_from_cache(id: u32) -> Option<User> {
    // 尝试从缓存获取
    None
}

fn get_user_from_database(id: u32) -> Result<User, DbError> {
    // 从数据库获取
    Ok(User { id, name: "Alice".to_string() })
}

fn get_user(id: u32) -> User {
    // 优先缓存,失败则数据库,再失败则默认值
    get_user_from_cache(id)
        .or_else(|| get_user_from_database(id).ok())
        .unwrap_or_else(|| User::default())
}
```

## 总结

本文介绍了 Rust 的错误处理机制：

- ✅ panic! 用于不可恢复错误
- ✅ `Result<T, E>` 用于可恢复错误
- ✅ ? 运算符简化错误传播
- ✅ unwrap 和 expect
- ✅ 自定义错误类型
- ✅ 错误处理最佳实践
- ✅ 错误处理设计模式:错误链、Context模式、Early Return
- ✅ 错误上下文:anyhow crate、自定义上下文
- ✅ 错误恢复策略:重试模式、降级模式

掌握错误处理后，继续学习 [泛型和 Trait](/docs/rust/generics-traits)。
