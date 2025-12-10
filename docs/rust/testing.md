---
sidebar_position: 11
title: 测试
---

# 测试

Rust 内置了测试框架,支持单元测试、集成测试、文档测试和基准测试。编写测试是保证代码质量的重要手段。

## 测试基础

### 第一个测试

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
```

### 运行测试

```bash
# 运行所有测试
cargo test

# 运行特定测试
cargo test it_works

# 显示打印输出
cargo test -- --show-output

# 并行/串行运行
cargo test -- --test-threads=1
```

## 断言宏

### assert

```rust
#[test]
fn test_assert() {
    let value = true;
    assert!(value, "value should be true");
}
```

### assert_eq! 和 assert_ne

```rust
#[test]
fn test_equality() {
    assert_eq!(2 + 2, 4);
    assert_ne!(2 + 2, 5);
}

#[derive(Debug, PartialEq)]
struct Point {
    x: i32,
    y: i32,
}

#[test]
fn test_struct_equality() {
    let p1 = Point { x: 1, y: 2 };
    let p2 = Point { x: 1, y: 2 };
    assert_eq!(p1, p2);
}
```

### 自定义错误消息

```rust
#[test]
fn test_with_message() {
    let expected = 5;
    let actual = 2 + 2;
    
    assert_eq!(
        expected,
        actual,
        "期望值 {} 不等于实际值 {}",
        expected,
        actual
    );
}
```

## 单元测试

### 测试私有函数

```rust
pub fn add_two(a: i32) -> i32 {
    internal_adder(a, 2)
}

fn internal_adder(a: i32, b: i32) -> i32 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn internal() {
        // 可以测试私有函数
        assert_eq!(4, internal_adder(2, 2));
    }
}
```

### 测试模式

```rust
pub struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    pub fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.height > other.height
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn larger_can_hold_smaller() {
        let larger = Rectangle { width: 8, height: 7 };
        let smaller = Rectangle { width: 5, height: 1 };
        
        assert!(larger.can_hold(&smaller));
    }
    
    #[test]
    fn smaller_cannot_hold_larger() {
        let larger = Rectangle { width: 8, height: 7 };
        let smaller = Rectangle { width: 5, height: 1 };
        
        assert!(!smaller.can_hold(&larger));
    }
}
```

## should_panic

### 基本用法

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
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[should_panic]
    fn greater_than_100() {
        Guess::new(200);
    }
}
```

### 检查 panic 消息

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[should_panic(expected = "between 1 and 100")]
    fn greater_than_100() {
        Guess::new(200);
    }
}
```

## Result 测试

### 使用 `Result<T, E>`

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() -> Result<(), String> {
        if 2 + 2 == 4 {
            Ok(())
        } else {
            Err(String::from("two plus two does not equal four"))
        }
    }
}
```

### 测试错误情况

```rust
fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err(String::from("division by zero"))
    } else {
        Ok(a / b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_divide_success() -> Result<(), String> {
        let result = divide(10.0, 2.0)?;
        assert_eq!(result, 5.0);
        Ok(())
    }
    
    #[test]
    fn test_divide_by_zero() {
        let result = divide(10.0, 0.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "division by zero");
    }
}
```

## 集成测试

### tests 目录

```
my_project/
├── Cargo.toml
├── src/
│   └── lib.rs
└── tests/
    ├── integration_test.rs
    └── common/
        └── mod.rs
```

### tests/integration_test.rs

```rust
use my_project;

#[test]
fn it_adds_two() {
    assert_eq!(4, my_project::add_two(2));
}
```

### 共享代码

```rust
// tests/common/mod.rs
pub fn setup() {
    // 测试设置代码
}

// tests/integration_test.rs
mod common;

#[test]
fn test_with_setup() {
    common::setup();
    // 测试代码
}
```

## 文档测试

### 编写文档测试

```rust
/// 将两个数字相加
///
/// # Examples
///
/// ```
/// let result = my_crate::add(2, 2);
/// assert_eq!(result, 4);
/// ```
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

### 隐藏部分代码

```rust
/// ```
/// # fn expensive_function() -> i32 { 42 }
/// let result = expensive_function();
/// assert_eq!(result, 42);
/// ```
pub fn example() {}
```

### 标记失败的测试

```rust
/// ```should_panic
/// panic!("this will panic");
/// ```
pub fn will_panic() {}

/// ```no_run
/// loop {
///     // 无限循环,但不运行
/// }
/// ```
pub fn infinite_loop() {}

/// ```ignore
/// // 暂时忽略这个测试
/// ```
pub fn ignored() {}
```

## 组织测试

### 按模块组织

```rust
#[cfg(test)]
mod unit_tests {
    use super::*;
    
    #[test]
    fn test_1() {}
    
    #[test]
    fn test_2() {}
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_3() {}
}
```

### 测试过滤

```bash
# 运行名称包含 "add" 的测试
cargo test add

# 运行特定模块的测试
cargo test unit_tests::

# 忽略的测试
cargo test -- --ignored

# 运行所有测试(包括忽略的)
cargo test -- --include-ignored
```

## 测试辅助工具

### Setup 和 Teardown

```rust
struct TestContext {
    value: i32,
}

impl TestContext {
    fn new() -> Self {
        println!("Setting up test");
        TestContext { value: 42 }
    }
}

impl Drop for TestContext {
    fn drop(&mut self) {
        println!("Tearing down test");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_with_context() {
        let ctx = TestContext::new();
        assert_eq!(ctx.value, 42);
        // Drop 会在这里自动调用
    }
}
```

### 测试fixture

```rust
#[cfg(test)]
mod tests {
    fn setup() -> Vec<i32> {
        vec![1, 2, 3, 4, 5]
    }
    
    #[test]
    fn test_with_fixture() {
        let data = setup();
        assert_eq!(data.len(), 5);
    }
}
```

## Mock 和 Stub

### 使用 trait 进行 Mock

```rust
trait Database {
    fn get_user(&self, id: u32) -> Option<String>;
}

struct RealDatabase;

impl Database for RealDatabase {
    fn get_user(&self, id: u32) -> Option<String> {
        // 真实的数据库查询
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    struct MockDatabase;
    
    impl Database for MockDatabase {
        fn get_user(&self, id: u32) -> Option<String> {
            if id == 1 {
                Some(String::from("Alice"))
            } else {
                None
            }
        }
    }
    
    #[test]
    fn test_with_mock() {
        let db = MockDatabase;
        assert_eq!(db.get_user(1), Some(String::from("Alice")));
        assert_eq!(db.get_user(2), None);
    }
}
```

## 基准测试

### 使用 Criterion

```toml
[dev-dependencies]
criterion = "0.5"

[[bench]]
name = "my_benchmark"
harness = false
```

### benches/my_benchmark.rs

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
```

### 运行基准测试

```bash
cargo bench
```

## 代码覆盖率

### 使用 tarpaulin

```bash
# 安装
cargo install cargo-tarpaulin

# 生成覆盖率报告
cargo tarpaulin --out Html
```

### 使用 llvm-cov

```bash
# 安装
rustup component add llvm-tools-preview
cargo install cargo-llvm-cov

# 运行
cargo llvm-cov --html
```

## 最佳实践

### 1. AAA 模式(Arrange-Act-Assert)

```rust
#[test]
fn test_with_aaa() {
    // Arrange: 准备测试数据
    let mut vec = Vec::new();
    
    // Act: 执行操作
    vec.push(1);
    vec.push(2);
    
    // Assert: 验证结果
    assert_eq!(vec.len(), 2);
    assert_eq!(vec[0], 1);
}
```

### 2. 一个测试一个断言

```rust
// 好:每个测试关注一个方面
#[test]
fn test_push_increases_length() {
    let mut vec = Vec::new();
    vec.push(1);
    assert_eq!(vec.len(), 1);
}

#[test]
fn test_push_adds_element() {
    let mut vec = Vec::new();
    vec.push(42);
    assert_eq!(vec[0], 42);
}
```

### 3. 描述性测试名称

```rust
// 好:清晰描述测试内容
#[test]
fn empty_vector_has_zero_length() {
    let vec: Vec<i32> = Vec::new();
    assert_eq!(vec.len(), 0);
}

// 避免:模糊的名称
#[test]
fn test1() {
    // ...
}
```

### 4. 测试边界情况

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_empty_input() {
        // 测试空输入
    }
    
    #[test]
    fn test_single_element() {
        // 测试单个元素
    }
    
    #[test]
    fn test_maximum_size() {
        // 测试最大值
    }
}
```

## 持续集成

### GitHub Actions 配置

```yaml
name: Rust

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    - name: Build
      run: cargo build --verbose
    - name: Run tests
      run: cargo test --verbose
    - name: Run clippy
      run: cargo clippy -- -D warnings
    - name: Check formatting
      run: cargo fmt -- --check
```

## 总结

本文全面介绍了 Rust 的测试:

- ✅ 单元测试、集成测试、文档测试
- ✅ 断言宏和错误处理
- ✅ should_panic 和 Result 测试
- ✅ Mock 和 Stub 技术
- ✅ 基准测试和代码覆盖率
- ✅ 测试最佳实践

**关键要点:**

1. 使用 `#[test]` 标记测试函数
2. 集成测试放在 `tests/` 目录
3. 文档测试确保示例代码正确
4. 遵循 AAA 模式组织测试
5. 使用 CI/CD 自动化测试

掌握测试后,继续学习 [宏编程](/docs/rust/macros) 或 [最佳实践](/docs/rust/best-practices)。
