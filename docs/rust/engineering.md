---
sidebar_position: 13
title: 工程化实践
---

# 工程化实践

本文介绍 Rust 项目的工程化最佳实践，涵盖错误处理、日志、配置管理、序列化等生产级项目必备的能力。

## 错误处理生态

### thiserror - 自定义错误类型

`thiserror` 用于为库定义错误类型,提供派生宏简化实现：

```toml
[dependencies]
thiserror = "1.0"
```

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DataError {
    #[error("数据未找到: {0}")]
    NotFound(String),

    #[error("无效的输入: {field}")]
    InvalidInput { field: String },

    #[error("IO 错误")]
    Io(#[from] std::io::Error),

    #[error("解析错误: {0}")]
    Parse(#[from] serde_json::Error),
}

fn load_config(path: &str) -> Result<Config, DataError> {
    let content = std::fs::read_to_string(path)?; // 自动转换 io::Error
    let config: Config = serde_json::from_str(&content)?; // 自动转换 serde Error
    Ok(config)
}
```

### anyhow - 应用级错误处理

`anyhow` 适用于应用程序,提供简单的错误传播：

```toml
[dependencies]
anyhow = "1.0"
```

```rust
use anyhow::{Context, Result, bail, ensure};

fn read_config(path: &str) -> Result<Config> {
    let content = std::fs::read_to_string(path)
        .context("读取配置文件失败")?;

    let config: Config = serde_json::from_str(&content)
        .context("解析配置失败")?;

    ensure!(config.port > 0, "端口必须大于 0");

    if config.host.is_empty() {
        bail!("主机地址不能为空");
    }

    Ok(config)
}

fn main() -> Result<()> {
    let config = read_config("config.json")?;
    println!("Config loaded: {:?}", config);
    Ok(())
}
```

### thiserror vs anyhow 选择

| 场景             | 推荐      |
| ---------------- | --------- |
| 库开发           | thiserror |
| 应用开发         | anyhow    |
| 需要匹配错误类型 | thiserror |
| 快速原型开发     | anyhow    |

## 日志系统 - tracing

`tracing` 是 Rust 生态中最强大的结构化日志框架：

```toml
[dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

### 基础使用

```rust
use tracing::{info, warn, error, debug, trace, instrument, span, Level};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() {
    // 初始化日志订阅器
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("应用启动");
    debug!(port = 8080, "监听端口");
    warn!(user_id = 42, "用户尝试访问受限资源");
    error!("发生严重错误");
}
```

### 结构化日志

```rust
use tracing::{info, instrument};

#[derive(Debug)]
struct User {
    id: u64,
    name: String,
}

#[instrument(skip(password))]
fn login(username: &str, password: &str) -> Result<User, String> {
    info!("用户尝试登录");
    // 登录逻辑
    Ok(User { id: 1, name: username.to_string() })
}

#[instrument]
async fn fetch_user(id: u64) -> Option<User> {
    info!("获取用户信息");
    Some(User { id, name: "Alice".to_string() })
}
```

### Span 和上下文

```rust
use tracing::{span, Level, info};

fn process_request(request_id: &str) {
    let span = span!(Level::INFO, "request", id = request_id);
    let _guard = span.enter();

    info!("开始处理请求");
    // 处理逻辑...
    info!("请求处理完成");
}
```

### JSON 格式输出

```rust
use tracing_subscriber::fmt::format::FmtSpan;

fn init_logging() {
    tracing_subscriber::fmt()
        .json()
        .with_span_events(FmtSpan::CLOSE)
        .with_current_span(true)
        .init();
}
```

## 配置管理

### config crate

```toml
[dependencies]
config = "0.13"
serde = { version = "1.0", features = ["derive"] }
```

```rust
use config::{Config, ConfigError, Environment, File};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Settings {
    pub database: DatabaseSettings,
    pub server: ServerSettings,
}

#[derive(Debug, Deserialize)]
pub struct DatabaseSettings {
    pub url: String,
    pub max_connections: u32,
}

#[derive(Debug, Deserialize)]
pub struct ServerSettings {
    pub host: String,
    pub port: u16,
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        let config = Config::builder()
            // 默认配置
            .add_source(File::with_name("config/default"))
            // 环境特定配置
            .add_source(File::with_name(&format!(
                "config/{}",
                std::env::var("RUN_ENV").unwrap_or_else(|_| "development".into())
            )).required(false))
            // 环境变量覆盖 (APP_DATABASE__URL 会覆盖 database.url)
            .add_source(Environment::with_prefix("APP").separator("__"))
            .build()?;

        config.try_deserialize()
    }
}
```

配置文件示例 `config/default.toml`:

```toml
[database]
url = "postgres://localhost/myapp"
max_connections = 10

[server]
host = "0.0.0.0"
port = 8080
```

## 序列化 - serde

### 基础用法

```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct User {
    id: u64,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    email: Option<String>,
}

fn main() {
    let user = User {
        id: 1,
        name: "Alice".to_string(),
        email: Some("alice@example.com".to_string()),
    };

    // 序列化为 JSON
    let json = serde_json::to_string_pretty(&user).unwrap();
    println!("{}", json);

    // 反序列化
    let parsed: User = serde_json::from_str(&json).unwrap();
    println!("{:?}", parsed);
}
```

### 高级属性

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiResponse {
    user_id: u64,
    user_name: String,

    #[serde(rename = "type")]
    response_type: String,

    #[serde(default)]
    is_verified: bool,

    #[serde(skip)]
    internal_data: String,

    #[serde(with = "chrono::serde::ts_seconds")]
    created_at: chrono::DateTime<chrono::Utc>,
}

// 枚举序列化
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
enum Message {
    Text { content: String },
    Image { url: String, width: u32, height: u32 },
    #[serde(rename = "file")]
    Attachment { name: String, size: u64 },
}
```

### 自定义序列化

```rust
use serde::{Deserialize, Deserializer, Serialize, Serializer};

fn serialize_uppercase<S>(value: &str, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&value.to_uppercase())
}

fn deserialize_from_str<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    s.parse().map_err(serde::de::Error::custom)
}

#[derive(Serialize, Deserialize)]
struct Item {
    #[serde(serialize_with = "serialize_uppercase")]
    name: String,

    #[serde(deserialize_with = "deserialize_from_str")]
    quantity: u64,
}
```

## Cargo Workspace

### 工作空间结构

```toml
# 根 Cargo.toml
[workspace]
resolver = "2"
members = [
    "crates/core",
    "crates/api",
    "crates/cli",
]

[workspace.package]
version = "0.1.0"
edition = "2021"
authors = ["Your Name <you@example.com>"]
license = "MIT"

[workspace.dependencies]
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
anyhow = "1.0"
tracing = "0.1"
```

### 成员包配置

```toml
# crates/core/Cargo.toml
[package]
name = "myapp-core"
version.workspace = true
edition.workspace = true

[dependencies]
serde.workspace = true
anyhow.workspace = true
```

### 内部依赖

```toml
# crates/api/Cargo.toml
[package]
name = "myapp-api"
version.workspace = true
edition.workspace = true

[dependencies]
myapp-core = { path = "../core" }
tokio.workspace = true
```

## Feature Flags

### 定义 Features

```toml
[package]
name = "mylib"
version = "0.1.0"

[features]
default = ["json"]
json = ["serde_json"]
yaml = ["serde_yaml"]
full = ["json", "yaml", "async"]
async = ["tokio"]

[dependencies]
serde = "1.0"
serde_json = { version = "1.0", optional = true }
serde_yaml = { version = "0.9", optional = true }
tokio = { version = "1", features = ["full"], optional = true }
```

### 条件编译

```rust
#[cfg(feature = "json")]
pub mod json {
    use serde::Serialize;

    pub fn to_json<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(value)
    }
}

#[cfg(feature = "async")]
pub mod async_utils {
    pub async fn sleep_ms(ms: u64) {
        tokio::time::sleep(tokio::time::Duration::from_millis(ms)).await;
    }
}

// 条件导入
#[cfg(feature = "json")]
pub use json::to_json;
```

### 使用 Features

```bash
# 默认 features
cargo build

# 禁用默认,启用特定 feature
cargo build --no-default-features --features yaml

# 启用所有 features
cargo build --features full
```

## CI/CD 配置

### GitHub Actions

```yaml
name: Rust CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy

      - name: Cache cargo
        uses: Swatinem/rust-cache@v2

      - name: Format check
        run: cargo fmt --all -- --check

      - name: Clippy
        run: cargo clippy --all-targets --all-features -- -D warnings

      - name: Build
        run: cargo build --all-features

      - name: Test
        run: cargo test --all-features
```

## 完整项目示例

```
myapp/
├── Cargo.toml
├── config/
│   ├── default.toml
│   ├── development.toml
│   └── production.toml
├── crates/
│   ├── core/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── error.rs
│   │       └── config.rs
│   ├── api/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       └── routes/
│   └── cli/
│       ├── Cargo.toml
│       └── src/
│           └── main.rs
├── tests/
│   └── integration_tests.rs
└── .github/
    └── workflows/
        └── ci.yml
```

## 最佳实践总结

| 方面     | 推荐工具/做法                 |
| -------- | ----------------------------- |
| 错误处理 | 库用 thiserror，应用用 anyhow |
| 日志     | tracing + tracing-subscriber  |
| 配置     | config crate + 环境变量       |
| 序列化   | serde 全家桶                  |
| 项目组织 | cargo workspace               |
| 可选功能 | feature flags                 |
| 代码质量 | cargo fmt + cargo clippy      |
| CI/CD    | GitHub Actions                |

掌握这些工程化技能，你就能写出生产级的 Rust 项目！
