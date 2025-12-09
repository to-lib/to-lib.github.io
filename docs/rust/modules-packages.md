---
sidebar_position: 10
title: 模块和包管理
---

# 模块和包管理

Rust 的模块系统帮助组织代码,管理作用域和隐私。包管理器 Cargo 简化了依赖管理和项目构建。

## 模块系统概述

Rust 的模块系统包括:

- **包(Package)** - Cargo 的功能,用于构建、测试和分享 crate
- **Crate** - 模块的树形结构,形成库或可执行文件
- **模块(Module)** - 控制作用域和隐私的代码组织单元
- **路径(Path)** - 命名项的方式

## 包和 Crate

### 创建包

```bash
# 创建二进制包
cargo new my_project

# 创建库包
cargo new my_lib --lib
```

### 包的结构

```
my_project/
├── Cargo.toml
├── src/
│   ├── main.rs      # 二进制 crate 根
│   └── lib.rs       # 库 crate 根
```

### Crate 根

- `src/main.rs` - 二进制 crate 的 crate 根
- `src/lib.rs` - 库 crate 的 crate 根

```rust
// src/main.rs
fn main() {
    println!("Hello, world!");
}

// src/lib.rs
pub fn hello() {
    println!("Hello from lib!");
}
```

## 定义模块

### 使用 mod 关键字

```rust
// src/lib.rs
mod front_of_house {
    mod hosting {
        fn add_to_waitlist() {}
        fn seat_at_table() {}
    }
    
    mod serving {
        fn take_order() {}
        fn serve_order() {}
        fn take_payment() {}
    }
}
```

### 模块树

```
crate
 └── front_of_house
     ├── hosting
     │   ├── add_to_waitlist
     │   └── seat_at_table
     └── serving
         ├── take_order
         ├── serve_order
         └── take_payment
```

## 路径

### 绝对路径和相对路径

```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

pub fn eat_at_restaurant() {
    // 绝对路径
    crate::front_of_house::hosting::add_to_waitlist();
    
    // 相对路径
    front_of_house::hosting::add_to_waitlist();
}
```

### super 关键字

```rust
fn serve_order() {}

mod back_of_house {
    fn fix_incorrect_order() {
        cook_order();
        super::serve_order();  // 使用 super 访问父模块
    }
    
    fn cook_order() {}
}
```

## 可见性

### pub 关键字

```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
        
        fn internal_function() {}  // 私有函数
    }
}

pub fn eat_at_restaurant() {
    front_of_house::hosting::add_to_waitlist();  // 可以访问
    // front_of_house::hosting::internal_function();  // 错误:私有
}
```

### 结构体和枚举的可见性

```rust
mod back_of_house {
    pub struct Breakfast {
        pub toast: String,        // 公有字段
        seasonal_fruit: String,   // 私有字段
    }
    
    impl Breakfast {
        pub fn summer(toast: &str) -> Breakfast {
            Breakfast {
                toast: String::from(toast),
                seasonal_fruit: String::from("peaches"),
            }
        }
    }
    
    // 枚举:pub 使所有成员公有
    pub enum Appetizer {
        Soup,
        Salad,
    }
}

pub fn eat_at_restaurant() {
    let mut meal = back_of_house::Breakfast::summer("Rye");
    meal.toast = String::from("Wheat");
    // meal.seasonal_fruit = String::from("blueberries");  // 错误:私有
    
    let order1 = back_of_house::Appetizer::Soup;
    let order2 = back_of_house::Appetizer::Salad;
}
```

## use 关键字

### 引入路径

```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

// 使用 use 引入
use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
}
```

### use 的惯用模式

```rust
// 函数:引入模块
use crate::front_of_house::hosting;
hosting::add_to_waitlist();

// 结构体、枚举等:引入完整路径
use std::collections::HashMap;
let mut map = HashMap::new();
```

### as 关键字

```rust
use std::fmt::Result;
use std::io::Result as IoResult;

fn function1() -> Result {
    // fmt::Result
    Ok(())
}

fn function2() -> IoResult<()> {
    // io::Result
    Ok(())
}
```

### pub use 重导出

```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

// 重导出:外部代码可以使用这个路径
pub use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
}
```

### 嵌套路径

```rust
// 不使用嵌套
use std::io;
use std::io::Write;

// 使用嵌套
use std::io::{self, Write};

// 多个项
use std::collections::{HashMap, BTreeMap, HashSet};

// glob 运算符
use std::collections::*;
```

## 将模块拆分到多个文件

### 文件结构

```
restaurant/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── front_of_house.rs
    └── front_of_house/
        ├── hosting.rs
        └── serving.rs
```

### src/lib.rs

```rust
mod front_of_house;

pub use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
}
```

### src/front_of_house.rs

```rust
pub mod hosting;
pub mod serving;
```

### src/front_of_house/hosting.rs

```rust
pub fn add_to_waitlist() {
    println!("添加到等待列表");
}

pub fn seat_at_table() {
    println!("安排座位");
}
```

## Cargo 和 Crates.io

### Cargo.toml 配置

```toml
[package]
name = "my_project"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <you@example.com>"]
description = "A short description"
license = "MIT"

[dependencies]
serde = "1.0"
serde_json = "1.0"
tokio = { version = "1.0", features = ["full"] }

[dev-dependencies]
criterion = "0.5"

[build-dependencies]
cc = "1.0"
```

### 添加依赖

```bash
# 添加依赖
cargo add serde

# 添加特定版本
cargo add tokio@1.0

# 添加带features的依赖
cargo add tokio --features full
```

### 版本要求

```toml
[dependencies]
# 精确版本
serde = "=1.0.152"

# 兼容版本(默认)
serde = "1.0"          # >= 1.0.0, < 2.0.0
serde = "1.0.152"      # >= 1.0.152, < 1.1.0

# 通配符
serde = "1.*"          # >= 1.0.0, < 2.0.0

# 范围
serde = ">= 1.0, < 2.0"
```

### 本地依赖

```toml
[dependencies]
my_lib = { path = "../my_lib" }
```

### Git 依赖

```toml
[dependencies]
regex = { git = "https://github.com/rust-lang/regex" }
regex = { git = "https://github.com/rust-lang/regex", branch = "master" }
regex = { git = "https://github.com/rust-lang/regex", tag = "1.5.4" }
```

## Cargo 工作空间

### 创建工作空间

```
workspace/
├── Cargo.toml
├── adder/
│   ├── Cargo.toml
│   └── src/
│       └── main.rs
└── add_one/
    ├── Cargo.toml
    └── src/
        └── lib.rs
```

### workspace/Cargo.toml

```toml
[workspace]
members = [
    "adder",
    "add_one",
]

[workspace.dependencies]
serde = "1.0"
```

### 在工作空间中运行命令

```bash
# 构建所有包
cargo build

# 构建特定包
cargo build -p add_one

# 运行特定包
cargo run -p adder

# 测试所有包
cargo test

# 测试特定包
cargo test -p add_one
```

### 共享依赖

```toml
# workspace/Cargo.toml
[workspace.dependencies]
serde = "1.0"

# add_one/Cargo.toml
[dependencies]
serde = { workspace = true }
```

## 发布到 Crates.io

### 准备发布

```rust
//! # My Crate
//!
//! `my_crate` 是一个演示包文档的集合

/// 将一加到数字上
///
/// # Examples
///
/// ```
/// let five = 5;
///
/// assert_eq!(6, my_crate::add_one(five));
/// ```
pub fn add_one(x: i32) -> i32 {
    x + 1
}
```

### 发布命令

```bash
# 登录
cargo login <token>

# 打包
cargo package

# 发布
cargo publish

# 撤回版本
cargo yank --vers 1.0.1
cargo yank --vers 1.0.1 --undo
```

## 最佳实践

### 1. 模块组织

```rust
// 好:按功能组织
mod database {
    pub mod user;
    pub mod post;
}

mod api {
    pub mod routes;
    pub mod handlers;
}

// 避免:所有代码在一个文件
```

### 2. 公有 API 设计

```rust
// src/lib.rs
mod internal;

// 重导出公有API
pub use internal::PublicStruct;
pub use internal::public_function;

// 保持内部实现私有
```

### 3. 使用 prelude 模块

```rust
// src/prelude.rs
pub use crate::Error;
pub use crate::Result;
pub use crate::Config;

// src/lib.rs
pub mod prelude;

// 用户代码
use my_crate::prelude::*;
```

### 4. 文档规范

```rust
/// 简短描述
///
/// # 详细说明
///
/// 更详细的说明...
///
/// # Examples
///
/// ```
/// use my_crate::example;
/// let result = example(5);
/// assert_eq!(result, 10);
/// ```
///
/// # Panics
///
/// 何时会 panic
///
/// # Errors
///
/// 何时返回错误
///
/// # Safety
///
/// 使用 unsafe 代码时的安全要求
pub fn example(x: i32) -> i32 {
    x * 2
}
```

## 实用工具

### Cargo 命令

```bash
# 检查代码
cargo check

# 构建
cargo build
cargo build --release

# 运行
cargo run

# 测试
cargo test
cargo test --doc

# 文档
cargo doc
cargo doc --open

# 清理
cargo clean

# 更新依赖
cargo update

# 审计依赖
cargo audit
```

### Cargo.toml 高级配置

```toml
[package]
name = "my_project"
version = "0.1.0"
edition = "2021"

# 优化配置
[profile.release]
opt-level = 3
lto = true
codegen-units = 1

[profile.dev]
opt-level = 0

# 特性标志
[features]
default = ["std"]
std = []
no_std = []
extra = ["dep:extra_crate"]

# 二进制目标
[[bin]]
name = "my_app"
path = "src/main.rs"
```

## 总结

本文详细介绍了 Rust 的模块和包管理:

- ✅ 模块系统:mod、pub、use
- ✅ 包和 Crate 的组织
- ✅ 路径和可见性控制
- ✅ 多文件模块拆分
- ✅ Cargo 包管理器
- ✅ 工作空间管理
- ✅ 发布到 Crates.io
- ✅ 最佳实践和工具

**关键要点:**

1. 使用模块组织代码,控制可见性
2. 合理使用 use 简化路径
3. pub use 重导出公有 API
4. 工作空间管理多包项目
5. 遵循发布规范和文档标准

掌握模块和包管理后,继续学习 [异步编程](./async-programming) 或 [宏编程](./macros)。
