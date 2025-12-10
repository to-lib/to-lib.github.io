---
sidebar_position: 7
title: 项目组织与代码分文件
---

# 项目组织与代码分文件

本文详细介绍如何在 Rust 项目中合理组织代码结构和分文件。

## 项目结构规范

### 标准二进制项目

```
my_app/
├── Cargo.toml          # 项目配置
├── Cargo.lock          # 依赖锁定
├── src/
│   ├── main.rs         # 程序入口
│   ├── lib.rs          # 可选：库代码
│   ├── config.rs       # 配置模块
│   ├── models/         # 数据模型目录
│   │   ├── mod.rs      # 模块声明
│   │   ├── user.rs
│   │   └── post.rs
│   └── utils/          # 工具函数目录
│       ├── mod.rs
│       └── helper.rs
├── tests/              # 集成测试
│   └── integration_test.rs
├── benches/            # 性能测试
│   └── benchmark.rs
└── examples/           # 示例代码
    └── demo.rs
```

### 标准库项目

```
my_lib/
├── Cargo.toml
├── src/
│   ├── lib.rs          # 库入口
│   ├── error.rs        # 错误类型
│   ├── types.rs        # 公共类型
│   └── api/            # API 模块
│       ├── mod.rs
│       ├── client.rs
│       └── server.rs
├── tests/
│   └── lib_test.rs
└── examples/
    └── usage.rs
```

## main.rs vs lib.rs

### main.rs - 二进制入口

`main.rs` 是可执行程序的入口点：

```rust
// src/main.rs
fn main() {
    println!("Hello, world!");
}
```

### lib.rs - 库入口

`lib.rs` 定义库的公共 API：

```rust
// src/lib.rs

// 公开模块
pub mod config;
pub mod models;
pub mod utils;

// 公开函数
pub fn init() {
    println!("Library initialized");
}

// 私有函数（仅库内部使用）
fn internal_helper() {
    // ...
}
```

### 同时拥有 main.rs 和 lib.rs

可以在一个项目中同时使用两者：

```rust
// src/lib.rs
pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

// src/main.rs
use my_app::greet;  // 引用库中的函数

fn main() {
    let message = greet("World");
    println!("{}", message);
}
```

## 模块系统

### 内联模块

在同一文件中定义模块：

```rust
// src/main.rs
mod utils {
    pub fn helper() {
        println!("Helper function");
    }

    fn private_helper() {
        println!("Private helper");
    }
}

fn main() {
    utils::helper();
    // utils::private_helper();  // 错误：私有函数
}
```

### 文件模块

每个 `.rs` 文件都是一个模块：

```rust
// src/config.rs
pub struct Config {
    pub host: String,
    pub port: u16,
}

impl Config {
    pub fn new(host: String, port: u16) -> Self {
        Config { host, port }
    }
}

// src/main.rs
mod config;  // 声明模块

fn main() {
    let cfg = config::Config::new("localhost".to_string(), 8080);
    println!("{}:{}", cfg.host, cfg.port);
}
```

### 目录模块

使用 `mod.rs` 组织多个文件：

```
src/
├── main.rs
└── models/
    ├── mod.rs      # 模块入口
    ├── user.rs
    └── post.rs
```

```rust
// src/models/mod.rs
pub mod user;  // 声明子模块
pub mod post;

// 可以在这里添加共享代码
pub trait Model {
    fn id(&self) -> u64;
}

// src/models/user.rs
use super::Model;  // 引用父模块

pub struct User {
    pub id: u64,
    pub name: String,
}

impl Model for User {
    fn id(&self) -> u64 {
        self.id
    }
}

// src/models/post.rs
pub struct Post {
    pub id: u64,
    pub title: String,
}

// src/main.rs
mod models;

fn main() {
    let user = models::user::User {
        id: 1,
        name: "Alice".to_string(),
    };

    let post = models::post::Post {
        id: 1,
        title: "Hello".to_string(),
    };
}
```

### 新式路径（推荐）

Rust 2018+ 支持更简洁的路径方式：

```
src/
├── main.rs
└── models/
    ├── user.rs     # 不需要 mod.rs
    └── post.rs
```

但需要在父模块中声明：

```rust
// src/main.rs 或 src/lib.rs
pub mod models {
    pub mod user;
    pub mod post;
}

// 或者创建 src/models.rs
// src/models.rs
pub mod user;
pub mod post;
```

## 可见性控制

### pub - 公开

```rust
// src/models/user.rs
pub struct User {      // 公开结构体
    pub id: u64,       // 公开字段
    pub name: String,
    age: u8,           // 私有字段
}

pub fn create_user(name: String) -> User {  // 公开函数
    User {
        id: 1,
        name,
        age: 0,
    }
}

fn internal_helper() {  // 私有函数
    // ...
}
```

### pub(crate) - crate 内公开

```rust
// src/utils/mod.rs
pub(crate) fn internal_api() {
    // 仅在当前 crate 内可见
}
```

### pub(super) - 父模块公开

```rust
// src/models/user.rs
pub(super) fn helper() {
    // 仅父模块（models）可见
}
```

### pub(in path) - 指定路径公开

```rust
// src/models/user.rs
pub(in crate::models) fn helper() {
    // 仅 models 模块内可见
}
```

## use 语句

### 基本用法

```rust
// 引入单个项
use std::collections::HashMap;

// 引入多个项
use std::io::{self, Read, Write};

// 引入所有公开项
use std::collections::*;

// 重命名
use std::io::Result as IoResult;
```

### 相对路径

```rust
// src/models/user.rs
use super::Model;      // 父模块
use crate::utils;      // 根模块
use self::helper;      // 当前模块
```

### 简化导入

```rust
// src/lib.rs
pub use models::user::User;  // 重新导出
pub use models::post::Post;

// 使用者可以直接：
use my_lib::{User, Post};
// 而不是：
// use my_lib::models::user::User;
// use my_lib::models::post::Post;
```

## 实战示例

### 示例 1: Web 应用结构

```
web_app/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── config.rs
│   ├── routes/
│   │   ├── mod.rs
│   │   ├── users.rs
│   │   └── posts.rs
│   ├── handlers/
│   │   ├── mod.rs
│   │   ├── user_handler.rs
│   │   └── post_handler.rs
│   ├── models/
│   │   ├── mod.rs
│   │   ├── user.rs
│   │   └── post.rs
│   ├── db/
│   │   ├── mod.rs
│   │   └── connection.rs
│   └── middleware/
│       ├── mod.rs
│       └── auth.rs
└── tests/
    └── api_test.rs
```

**src/lib.rs:**

```rust
pub mod config;
pub mod routes;
pub mod handlers;
pub mod models;
pub mod db;
pub mod middleware;

// 重新导出常用类型
pub use config::Config;
pub use models::{User, Post};
```

**src/main.rs:**

```rust
use web_app::{Config, routes};

#[tokio::main]
async fn main() {
    let config = Config::load();
    let app = routes::create_app(config);
    // 启动服务器...
}
```

**src/routes/mod.rs:**

```rust
pub mod users;
pub mod posts;

use axum::Router;
use crate::Config;

pub fn create_app(config: Config) -> Router {
    Router::new()
        .nest("/users", users::routes())
        .nest("/posts", posts::routes())
}
```

**src/routes/users.rs:**

```rust
use axum::{Router, routing::get};
use crate::handlers::user_handler;

pub fn routes() -> Router {
    Router::new()
        .route("/", get(user_handler::list))
        .route("/:id", get(user_handler::get))
}
```

### 示例 2: CLI 工具结构

```
cli_tool/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── cli.rs          # 命令行解析
│   ├── commands/
│   │   ├── mod.rs
│   │   ├── init.rs
│   │   ├── build.rs
│   │   └── run.rs
│   ├── config.rs
│   └── utils/
│       ├── mod.rs
│       └── fs.rs
└── tests/
    └── cli_test.rs
```

**src/main.rs:**

```rust
mod cli;
mod commands;
mod config;
mod utils;

use clap::Parser;
use cli::Cli;

fn main() {
    let cli = Cli::parse();

    match cli.command {
        cli::Commands::Init => commands::init::run(),
        cli::Commands::Build => commands::build::run(),
        cli::Commands::Run => commands::run::run(),
    }
}
```

**src/cli.rs:**

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Init,
    Build,
    Run,
}
```

**src/commands/mod.rs:**

```rust
pub mod init;
pub mod build;
pub mod run;
```

**src/commands/init.rs:**

```rust
use crate::config::Config;
use crate::utils;

pub fn run() {
    println!("初始化项目...");
    let config = Config::default();
    utils::fs::create_dirs(&config.paths);
}
```

### 示例 3: 库项目结构

```
my_lib/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── error.rs
│   ├── prelude.rs      # 常用导入
│   ├── client/
│   │   ├── mod.rs
│   │   ├── sync.rs     # 同步客户端
│   │   └── async.rs    # 异步客户端
│   ├── types/
│   │   ├── mod.rs
│   │   ├── request.rs
│   │   └── response.rs
│   └── internal/       # 内部实现
│       ├── mod.rs
│       └── parser.rs
├── examples/
│   └── basic.rs
└── tests/
    └── integration.rs
```

**src/lib.rs:**

```rust
// 错误类型
mod error;
pub use error::MyError;

// 公开模块
pub mod client;
pub mod types;

// 内部模块
mod internal;

// Prelude - 方便用户导入
pub mod prelude {
    pub use crate::client::Client;
    pub use crate::types::{Request, Response};
    pub use crate::error::MyError;
}
```

**src/prelude.rs:**

```rust
// 用户可以这样使用：
// use my_lib::prelude::*;
pub use crate::client::Client;
pub use crate::types::{Request, Response};
pub use crate::error::MyError;
```

## 最佳实践

### 1. 保持模块小而专注

每个模块应该有单一职责：

```
✅ 好的组织：
src/
├── auth/           # 认证相关
├── db/             # 数据库相关
├── api/            # API 相关

❌ 不好的组织：
src/
├── stuff.rs        # 太宽泛
├── helpers.rs      # 不明确
```

### 2. 使用 prelude 模式

为库提供便捷的导入：

```rust
// src/prelude.rs
pub use crate::most::commonly::used::Type1;
pub use crate::most::commonly::used::Type2;

// 用户使用
use my_lib::prelude::*;
```

### 3. 重新导出公共 API

在 `lib.rs` 中重新导出：

```rust
// src/lib.rs
mod internal_module;

pub use internal_module::PublicType;  // 简化路径
```

### 4. 避免循环依赖

```rust
// ❌ 不好
// a.rs 依赖 b.rs
// b.rs 依赖 a.rs

// ✅ 好：提取共享代码到第三个模块
// a.rs 依赖 common.rs
// b.rs 依赖 common.rs
```

### 5. 合理使用可见性

```rust
// 默认私有
struct InternalConfig { }

// 需要时才公开
pub struct PublicConfig { }

// crate 内部公开
pub(crate) struct CrateConfig { }
```

### 6. 使用 tests 目录

```
src/              # 源代码
tests/            # 集成测试（黑盒测试）
  └── api.rs

// src/ 中的测试是单元测试
#[cfg(test)]
mod tests {
    #[test]
    fn test_something() { }
}
```

## 常见模式

### 错误模块模式

```rust
// src/error.rs
use std::fmt;

#[derive(Debug)]
pub enum MyError {
    NotFound,
    InvalidInput(String),
    Internal(Box<dyn std::error::Error>),
}

impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MyError::NotFound => write!(f, "Not found"),
            MyError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            MyError::Internal(e) => write!(f, "Internal error: {}", e),
        }
    }
}

impl std::error::Error for MyError {}

pub type Result<T> = std::result::Result<T, MyError>;

// src/lib.rs
mod error;
pub use error::{MyError, Result};
```

### 配置模块模式

```rust
// src/config.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct Config {
    pub database_url: String,
    pub port: u16,
}

impl Config {
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        // 从文件或环境变量加载
        todo!()
    }
}
```

### Builder 模式

```rust
// src/builder.rs
pub struct ConfigBuilder {
    host: Option<String>,
    port: Option<u16>,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        ConfigBuilder {
            host: None,
            port: None,
        }
    }

    pub fn host(mut self, host: String) -> Self {
        self.host = Some(host);
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    pub fn build(self) -> Config {
        Config {
            host: self.host.unwrap_or_else(|| "localhost".to_string()),
            port: self.port.unwrap_or(8080),
        }
    }
}
```

## 总结

- ✅ 使用清晰的目录结构组织代码
- ✅ 合理使用 `mod.rs` 或新式路径
- ✅ 控制好可见性（pub、pub(crate)等）
- ✅ 使用 `use` 简化路径
- ✅ 重新导出简化公共 API
- ✅ 保持模块职责单一
- ✅ 提供 prelude 方便用户

良好的项目组织能让代码更易维护和扩展！
