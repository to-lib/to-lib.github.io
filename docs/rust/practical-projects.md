---
sidebar_position: 18
title: 实战项目
---

# Rust 实战项目

通过实践项目学习 Rust 是最有效的方式。本文提供了多个不同难度和类型的项目示例。

## CLI 工具项目

### 项目 1: grep 克隆

**难度：** ⭐⭐⭐

**目标：** 实现一个简化版的 grep 工具。

**功能：**

- 在文件中搜索匹配的行
- 支持正则表达式
- 大小写敏感/不敏感
- 显示行号

**核心代码：**

```rust
use std::env;
use std::fs;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <pattern> <file>", args[0]);
        process::exit(1);
    }

    let pattern = &args[1];
    let filename = &args[2];

    let contents = fs::read_to_string(filename)
        .unwrap_or_else(|err| {
            eprintln!("Error reading file: {}", err);
            process::exit(1);
        });

    search(pattern, &contents);
}

fn search(pattern: &str, contents: &str) {
    for (i, line) in contents.lines().enumerate() {
        if line.contains(pattern) {
            println!("{}: {}", i + 1, line);
        }
    }
}
```

**学习要点：**

- 命令行参数处理
- 文件 I/O
- 字符串处理
- 错误处理

### 项目 2: 文件搜索工具

**难度：** ⭐⭐⭐⭐

**目标：** 快速搜索文件和目录。

**功能：**

- 递归搜索目录
- 支持通配符
- 并发搜索
- 忽略 .gitignore 文件

**依赖：**

```toml
[dependencies]
clap = { version = "4", features = ["derive"] }
walkdir = "2"
regex = "1"
ignore = "0.4"
```

**核心代码：**

```rust
use clap::Parser;
use walkdir::WalkDir;
use regex::Regex;

#[derive(Parser)]
struct Args {
    pattern: String,
    path: String,
}

fn main() {
    let args = Args::parse();
    let re = Regex::new(&args.pattern).unwrap();

    for entry in WalkDir::new(&args.path) {
        let entry = entry.unwrap();
        if entry.file_type().is_file() {
            if let Some(filename) = entry.file_name().to_str() {
                if re.is_match(filename) {
                    println!("{}", entry.path().display());
                }
            }
        }
    }
}
```

## Web 服务项目

### 项目 3: REST API 服务

**难度：** ⭐⭐⭐⭐

**目标：** 创建一个简单的 RESTful API。

**功能：**

- CRUD 操作
- JSON 序列化
- 路由处理
- 数据持久化

**依赖：**

```toml
[dependencies]
axum = "0.6"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

**核心代码：**

```rust
use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Clone, Serialize, Deserialize)]
struct Todo {
    id: u64,
    title: String,
    completed: bool,
}

type SharedState = Arc<Mutex<Vec<Todo>>>;

#[tokio::main]
async fn main() {
    let state = Arc::new(Mutex::new(Vec::new()));

    let app = Router::new()
        .route("/todos", get(list_todos).post(create_todo))
        .route("/todos/:id", get(get_todo))
        .with_state(state);

    axum::Server::bind(&"127.0.0.1:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn list_todos(
    State(state): State<SharedState>,
) -> Json<Vec<Todo>> {
    let todos = state.lock().unwrap();
    Json(todos.clone())
}

async fn create_todo(
    State(state): State<SharedState>,
    Json(payload): Json<Todo>,
) -> (StatusCode, Json<Todo>) {
    let mut todos = state.lock().unwrap();
    todos.push(payload.clone());
    (StatusCode::CREATED, Json(payload))
}

async fn get_todo(
    State(state): State<SharedState>,
    Path(id): Path<u64>,
) -> Result<Json<Todo>, StatusCode> {
    let todos = state.lock().unwrap();
    todos.iter()
        .find(|t| t.id == id)
        .cloned()
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}
```

### 项目 4: WebSocket 聊天室

**难度：** ⭐⭐⭐⭐⭐

**目标：** 实现一个实时聊天服务器。

**功能：**

- WebSocket 连接
- 广播消息
- 用户加入/离开通知
- 私聊功能

**依赖：**

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
tokio-tungstenite = "0.20"
futures-util = "0.3"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

**核心思路：**

- 使用 tokio 异步处理连接
- HashMap 存储活跃连接
- mpsc 通道广播消息
- 消息类型（加入、离开、聊天）

## WebAssembly 项目

### 项目 5: Rust + WASM 前端工具库

**难度：** ⭐⭐⭐⭐

**目标：** 编写一个可在浏览器中运行的 WASM 模块（例如：字符串处理/加密摘要/图像处理的核心算法），并提供简单的 JavaScript/TypeScript 调用方式。

**功能：**

- 将 Rust 编译为 WebAssembly
- 导出少量稳定 API（函数/结构体）
- 通过 npm 包或本地 demo 页面使用

**依赖与工具：**

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

**项目骨架：**

```bash
# 创建库项目
cargo new wasm-utils --lib
cd wasm-utils

# 使用 wasm-pack 构建（会生成 pkg/）
wasm-pack build --target web
```

**核心代码：**

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn reverse(s: &str) -> String {
    s.chars().rev().collect()
}
```

**学习要点：**

- Rust 的 `crate-type` 与导出边界
- `wasm-bindgen` 的 ABI 约束与字符串/数组的传递
- 版本管理与产物发布（npm / GitHub Release）

## 系统编程项目

### 项目 6: 简单 Shell

**难度：** ⭐⭐⭐⭐⭐

**目标：** 实现一个基本的命令行 shell。

**功能：**

- 执行外部命令
- 内置命令（cd、exit）
- 管道支持
- 重定向

**核心代码：**

```rust
use std::io::{self, Write};
use std::process::{Command, Stdio};

fn main() {
    loop {
        print!("$ ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        if input == "exit" {
            break;
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        let command = parts[0];
        let args = &parts[1..];

        match command {
            "cd" => {
                if args.len() > 0 {
                    std::env::set_current_dir(args[0])
                        .unwrap_or_else(|e| eprintln!("cd: {}", e));
                }
            }
            _ => {
                Command::new(command)
                    .args(args)
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .spawn()
                    .and_then(|mut child| child.wait())
                    .unwrap_or_else(|e| {
                        eprintln!("{}: {}", command, e);
                        std::process::ExitStatus::default()
                    });
            }
        }
    }
}
```

### 项目 7: 内存分配器

**难度：** ⭐⭐⭐⭐⭐

**目标：** 实现一个简单的内存分配器。

**功能：**

- 内存分配和释放
- 空闲列表管理
- 碎片处理

**核心思路：**

- 使用 `GlobalAlloc` trait
- 实现 `alloc` 和 `dealloc`
- 内存对齐
- 线程安全

## 网络编程项目

### 项目 8: HTTP 服务器

**难度：** ⭐⭐⭐⭐

**目标：** 从零实现一个 HTTP/1.1 服务器。

**功能：**

- 解析 HTTP 请求
- 构建 HTTP 响应
- 静态文件服务
- 基本路由

**核心代码：**

```rust
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::fs;

fn main() {
    let listener = TcpListener::bind("127.0.0.1:8080").unwrap();
    println!("Server listening on port 8080");

    for stream in listener.incoming() {
        let stream = stream.unwrap();
        handle_connection(stream);
    }
}

fn handle_connection(mut stream: TcpStream) {
    let mut buffer = [0; 1024];
    stream.read(&mut buffer).unwrap();

    let request = String::from_utf8_lossy(&buffer);
    let lines: Vec<&str> = request.lines().collect();

    if lines.is_empty() {
        return;
    }

    let parts: Vec<&str> = lines[0].split_whitespace().collect();
    if parts.len() < 2 {
        return;
    }

    let method = parts[0];
    let path = parts[1];

    let (status, content) = match (method, path) {
        ("GET", "/") => {
            let contents = fs::read_to_string("static/index.html")
                .unwrap_or_else(|_| String::from("Hello, World!"));
            ("200 OK", contents)
        }
        _ => ("404 NOT FOUND", String::from("Not Found")),
    };

    let response = format!(
        "HTTP/1.1 {}\r\nContent-Length: {}\r\n\r\n{}",
        status,
        content.len(),
        content
    );

    stream.write_all(response.as_bytes()).unwrap();
}
```

### 项目 9: TCP 聊天室

**难度：** ⭐⭐⭐⭐

**目标：** 实现一个多用户 TCP 聊天服务器。

**功能：**

- 多客户端连接
- 消息广播
- 用户列表
- 私聊

**学习要点：**

- 异步 I/O
- 通道通信
- 并发控制

## 数据处理项目

### 项目 10: JSON 解析器

**难度：** ⭐⭐⭐⭐⭐

**目标：** 实现一个 JSON 解析器。

**功能：**

- 词法分析
- 语法分析
- 错误处理
- 格式化输出

**核心思路：**

- 状态机解析
- 递归下降解析
- 枚举表示 JSON 值类型

### 项目 11: CSV 处理工具

**难度：** ⭐⭐⭐

**目标：** CSV 文件读写和转换。

**功能：**

- CSV 读取
- 数据过滤
- 统计分析
- 格式转换（JSON、Excel）

**依赖：**

```toml
[dependencies]
csv = "1"
serde = { version = "1", features = ["derive"] }
```

**示例：**

```rust
use csv::Reader;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Record {
    name: String,
    age: u32,
    city: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rdr = Reader::from_path("data.csv")?;

    for result in rdr.deserialize() {
        let record: Record = result?;
        println!("{:?}", record);
    }

    Ok(())
}
```

## 学习路径建议

### 初级（1-3 个月）

1. **CLI 计算器** - 基础语法练习
2. **Todo 列表** - 数据结构和文件 I/O
3. **简单 HTTP 客户端** - 网络编程入门

### 中级（3-6 个月）

1. **REST API 服务** - Web 开发
2. **并发下载器** - 并发编程
3. **Markdown 解析器** - 字符串处理

### 高级（6+ 个月）

1. **数据库引擎** - 存储和查询
2. **编程语言解释器** - 编译原理
3. **分布式 KV 存储** - 分布式系统

## 推荐学习资源

### 官方资源

- [The Rust Book](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
- [Rustlings](https://github.com/rust-lang/rustlings/)

### 项目灵感

- [Awesome Rust](https://github.com/rust-unofficial/awesome-rust)
- [Exercism Rust Track](https://exercism.org/tracks/rust)
- [Build Your Own X](https://github.com/codecrafters-io/build-your-own-x)

### 开源项目学习

- **ripgrep** - 快速文本搜索
- **tokio** - 异步运行时
- **serde** - 序列化框架
- **actix-web** - Web 框架

## 项目开发建议

### 1. 从小开始

不要一开始就追求完美，先实现核心功能。

### 2. 迭代开发

- 第一版：基本功能
- 第二版：错误处理
- 第三版：性能优化
- 第四版：测试和文档

### 3. 测试驱动

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse() {
        assert_eq!(parse("input"), expected);
    }
}
```

### 4. 代码质量

```bash
# 格式化
cargo fmt

# 检查
cargo clippy

# 测试
cargo test
```

### 5. 文档

````rust
/// 解析输入字符串
///
/// # Examples
///
/// ```
/// let result = parse("hello");
/// assert_eq!(result, "HELLO");
/// ```
pub fn parse(input: &str) -> String {
    input.to_uppercase()
}
````

## 总结

通过实践项目学习 Rust：

- ✅ 选择感兴趣的项目
- ✅ 从简单到复杂
- ✅ 注重代码质量
- ✅ 阅读他人代码
- ✅ 参与开源项目
- ✅ 持续学习和实践

最重要的是**动手实践**，理论知识只有在实践中才能真正掌握！
