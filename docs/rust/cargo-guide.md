---
sidebar_position: 6
title: Cargo 使用指南
---

# Cargo 使用指南

Cargo 是 Rust 的包管理器和构建系统，提供了项目管理、依赖管理、构建和发布等功能。

## Cargo 基础

### 安装 Cargo

Cargo 随 Rust 一起安装：

```bash
# 验证安装
cargo --version
```

### 创建项目

```bash
# 创建二进制项目
cargo new my_project

# 创建库项目
cargo new my_lib --lib

# 在当前目录创建
cargo init
```

### 项目结构

```
my_project/
├── Cargo.toml       # 项目配置
├── Cargo.lock       # 依赖锁定（自动生成）
├── src/
│   └── main.rs      # 二进制入口
└── target/          # 构建输出（自动生成）
```

库项目结构：

```
my_lib/
├── Cargo.toml
├── src/
│   └── lib.rs       # 库入口
└── tests/           # 集成测试
```

## Cargo.toml 配置

### 基本配置

```toml
[package]
name = "my_project"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <you@example.com>"]
description = "A short description"
license = "MIT"
repository = "https://github.com/username/repo"
keywords = ["cli", "tool"]
categories = ["command-line-utilities"]
```

### 依赖管理

```toml
[dependencies]
serde = "1.0"                          # 最新 1.x 版本
tokio = "1.28.0"                       # 精确到补丁版本
rand = "0.8.5"                         # 兼容范围

# 从 git 仓库
reqwest = { git = "https://github.com/seanmonstar/reqwest" }

# 本地路径
my_lib = { path = "../my_lib" }

# 指定 features
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }

# 可选依赖
clap = { version = "4", optional = true }

[dev-dependencies]
# 仅用于测试和基准测试
criterion = "0.5"

[build-dependencies]
# 用于构建脚本
cc = "1.0"
```

### 版本指定

```toml
# 插入符号（默认）
serde = "^1.0"     # >= 1.0.0, < 2.0.0
serde = "^1.2"     # >= 1.2.0, < 2.0.0

# 波浪号
serde = "~1.2.3"   # >= 1.2.3, < 1.3.0

# 通配符
serde = "1.*"      # >= 1.0.0, < 2.0.0

# 比较运算符
serde = ">= 1.0, < 2.0"

# 精确版本
serde = "= 1.0.0"
```

### Features 特性

```toml
[features]
default = ["std"]              # 默认特性
std = []                       # 标准库支持
advanced = ["dep:tokio"]       # 启用可选依赖

[dependencies]
tokio = { version = "1", optional = true }
```

使用特性：

```bash
# 使用默认特性
cargo build

# 不使用默认特性
cargo build --no-default-features

# 指定特性
cargo build --features "advanced"
```

### 构建配置

```toml
[profile.dev]
opt-level = 0           # 无优化（快速编译）

[profile.release]
opt-level = 3           # 最大优化
lto = true              # 链接时优化
codegen-units = 1       # 单线程代码生成（更好优化）
strip = true            # 移除符号信息

[profile.test]
opt-level = 1           # 测试时轻微优化
```

## Cargo 命令

### 构建命令

```bash
# 编译项目
cargo build

# 发布构建（优化）
cargo build --release

# 检查代码（不生成可执行文件）
cargo check

# 清理构建产物
cargo clean
```

### 运行命令

```bash
# 运行项目
cargo run

# 运行并传递参数
cargo run -- arg1 arg2

# 运行示例
cargo run --example example_name

# 发布模式运行
cargo run --release
```

### 测试命令

```bash
# 运行所有测试
cargo test

# 运行特定测试
cargo test test_name

# 显示测试输出
cargo test -- --nocapture

# 运行单个测试
cargo test -- --test-threads=1

# 运行文档测试
cargo test --doc

# 运行集成测试
cargo test --test integration_test
```

### 文档命令

```bash
# 生成文档
cargo doc

# 生成并打开文档
cargo doc --open

# 包含所有依赖的文档
cargo doc --no-deps
```

### 发布命令

```bash
# 打包项目
cargo package

# 发布到 crates.io
cargo publish

# 撤回已发布版本
cargo yank --vers 1.0.1

# 检查是否可以发布
cargo publish --dry-run
```

## 工作空间 (Workspace)

### 创建工作空间

项目根目录的 `Cargo.toml`：

```toml
[workspace]
members = [
    "crate1",
    "crate2",
    "crate3",
]

[workspace.dependencies]
serde = "1.0"
tokio = { version = "1", features = ["full"] }
```

目录结构：

```
workspace/
├── Cargo.toml
├── Cargo.lock
├── crate1/
│   ├── Cargo.toml
│   └── src/
├── crate2/
│   ├── Cargo.toml
│   └── src/
└── target/
```

### 工作空间成员配置

`crate1/Cargo.toml`：

```toml
[package]
name = "crate1"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { workspace = true }
crate2 = { path = "../crate2" }
```

### 工作空间命令

```bash
# 构建所有成员
cargo build

# 构建特定成员
cargo build -p crate1

# 运行特定成员
cargo run -p crate1

# 测试所有成员
cargo test

# 测试特定成员
cargo test -p crate2
```

## 常用插件

### cargo-edit

管理依赖的工具：

```bash
# 安装
cargo install cargo-edit

# 添加依赖
cargo add serde

# 添加开发依赖
cargo add --dev tokio-test

# 删除依赖
cargo rm serde

# 升级依赖
cargo upgrade
```

### cargo-watch

自动构建和测试：

```bash
# 安装
cargo install cargo-watch

# 自动重新构建
cargo watch

# 自动运行测试
cargo watch -x test

# 自动运行程序
cargo watch -x run
```

### cargo-expand

展开宏：

```bash
# 安装
cargo install cargo-expand

# 展开宏
cargo expand
```

### cargo-tree

显示依赖树：

```bash
# 显示依赖树
cargo tree

# 显示特定包的依赖
cargo tree -p serde

# 反向依赖
cargo tree -i serde
```

### cargo-audit

安全审计：

```bash
# 安装
cargo install cargo-audit

# 检查依赖漏洞
cargo audit
```

### cargo-outdated

检查过时依赖：

```bash
# 安装
cargo install cargo-outdated

# 检查过时依赖
cargo outdated
```

### cargo-criterion

性能基准测试：

```bash
# 安装
cargo install cargo-criterion

# 运行基准测试
cargo criterion
```

## 构建脚本

### build.rs

项目根目录创建 `build.rs`：

```rust
fn main() {
    println!("cargo:rerun-if-changed=src/hello.c");

    // 编译 C 代码
    cc::Build::new()
        .file("src/hello.c")
        .compile("hello");

    // 设置环境变量
    println!("cargo:rustc-env=BUILD_TIME={}", chrono::Utc::now());
}
```

### 常用指令

```rust
// 链接库
println!("cargo:rustc-link-lib=sqlite3");

// 搜索路径
println!("cargo:rustc-link-search=/path/to/lib");

// 传递给 rustc 的参数
println!("cargo:rustc-cfg=feature=\"custom\"");

// 重新运行条件
println!("cargo:rerun-if-changed=build.rs");
println!("cargo:rerun-if-env-changed=CC");
```

## 配置文件

### .cargo/config.toml

项目或用户级配置：

```toml
# 构建配置
[build]
target = "x86_64-unknown-linux-musl"
jobs = 4

# 目标配置
[target.x86_64-unknown-linux-gnu]
linker = "clang"

# 别名
[alias]
b = "build"
t = "test"
r = "run"
br = "build --release"

# 环境变量
[env]
RUST_BACKTRACE = "1"

# 注册表镜像
[source.crates-io]
replace-with = "tuna"

[source.tuna]
registry = "https://mirrors.tuna.tsinghua.edu.cn/git/crates.io-index.git"
```

## 发布到 crates.io

### 准备发布

1. 完善 `Cargo.toml`：

```toml
[package]
name = "my_crate"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <you@example.com>"]
license = "MIT OR Apache-2.0"
description = "A useful crate"
repository = "https://github.com/username/my_crate"
documentation = "https://docs.rs/my_crate"
readme = "README.md"
keywords = ["cli", "utility"]
categories = ["command-line-utilities"]
```

2. 登录 crates.io：

```bash
cargo login <your-api-token>
```

3. 发布：

```bash
# 检查
cargo publish --dry-run

# 发布
cargo publish
```

### 发布检查清单

- ✅ README.md 存在且完善
- ✅ 许可证文件
- ✅ 文档注释完整
- ✅ 所有测试通过
- ✅ 版本号符合语义化版本
- ✅ 示例代码可运行
- ✅ 依赖版本合理

## 最佳实践

### 1. 使用工作空间

```toml
# 大型项目使用工作空间
[workspace]
members = ["core", "cli", "server"]
```

### 2. 锁定依赖版本

```bash
# 提交 Cargo.lock（二进制项目）
git add Cargo.lock

# 不提交 Cargo.lock（库项目）
echo "Cargo.lock" >> .gitignore
```

### 3. 使用 features 控制功能

```toml
[features]
default = ["std"]
std = []
no_std = []
experimental = []
```

### 4. 使用 dev-dependencies

```toml
[dev-dependencies]
# 仅测试时需要
proptest = "1.0"
```

### 5. 定期更新依赖

```bash
cargo update
cargo audit
cargo outdated
```

### 6. 使用 clippy 检查代码

```bash
cargo clippy
cargo clippy -- -D warnings
```

### 7. 使用 rustfmt 格式化

```bash
cargo fmt
cargo fmt -- --check
```

## 常见问题

### 依赖冲突

```bash
# 查看依赖树
cargo tree

# 更新依赖
cargo update -p package_name
```

### 构建缓存

```bash
# 清理缓存
cargo clean

# 清理特定 target
rm -rf target/debug
```

### 编译速度优化

```toml
# Cargo.toml
[profile.dev]
incremental = true

[profile.dev.package."*"]
opt-level = 0
debug = false
```

### 交叉编译

```bash
# 添加目标
rustup target add x86_64-pc-windows-gnu

# 构建
cargo build --target x86_64-pc-windows-gnu
```

## 总结

- ✅ Cargo 是 Rust 的核心工具
- ✅ Cargo.toml 配置项目信息和依赖
- ✅ 工作空间管理多个包
- ✅ 丰富的插件生态
- ✅ 支持构建脚本自定义构建
- ✅ 发布到 crates.io 很简单
- ✅ 使用 features 控制可选功能

掌握 Cargo，能大大提升 Rust 开发效率！
