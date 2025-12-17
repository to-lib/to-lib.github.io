---
sidebar_position: 2
title: Rust 开发环境搭建
---

# Rust 开发环境搭建

本文档指导你在不同操作系统上安装和配置 Rust 开发环境。

> [!IMPORTANT]
> Rust 使用 **rustup** 作为官方工具链管理器，它可以轻松安装、更新和管理多个 Rust 版本。

## 为什么选择 Rust

### 核心优势

- ✅ **内存安全** - 编译时检查，无需垃圾回收器
- ✅ **零成本抽象** - 抽象不会带来运行时开销
- ✅ **并发无惧** - 编译器保证线程安全
- ✅ **现代工具链** - Cargo 包管理器和构建系统
- ✅ **跨平台** - 支持多种操作系统和架构

### Rust 版本选择

| 版本        | 说明                | 推荐场景         |
| ----------- | ------------------- | ---------------- |
| **stable**  | 稳定版，每 6 周发布 | 生产环境（推荐） |
| **beta**    | 测试版，稳定前预览  | 测试新特性       |
| **nightly** | 每日构建，实验特性  | 尝试最新功能     |

## Windows 安装

### 步骤 1：下载 rustup

从 [rustup.rs](https://rustup.rs/) 下载安装程序：

1. 访问 https://rustup.rs/
2. 点击下载 `rustup-init.exe`
3. 运行安装程序

### 步骤 2：安装 Rust

双击运行 `rustup-init.exe`，按照提示选择：

```
1) Proceed with installation (default)
```

安装完成后，重新打开命令提示符。

### 步骤 3：安装 Visual Studio C++ 生成工具

Rust 在 Windows 上需要 C++ 链接器。有两种选择：

**选项 1：安装 Visual Studio Build Tools（推荐）**

1. 下载 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. 安装时选择 "C++ 生成工具" 工作负载

**选项 2：安装 MinGW-w64**

```powershell
# 使用 winget
winget install MinGW.MinGW

# 或使用 chocolatey
choco install mingw
```

### 步骤 4：验证安装

打开新的命令提示符：

```bash
rustc --version
```

预期输出：

```
rustc 1.xx.x (xxxxxxxx 2024-xx-xx)
```

```bash
cargo --version
```

预期输出：

```
cargo 1.xx.x (xxxxxxxx 2024-xx-xx)
```

## macOS 安装

### 方法 1：使用 rustup（推荐）

打开终端，运行：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

按照提示选择默认安装：

```
1) Proceed with installation (default)
```

### 方法 2：使用 Homebrew

```bash
# 安装 Homebrew（如未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装 Rust
brew install rust
```

> [!NOTE]
> 推荐使用 rustup 安装，因为它提供了更好的版本管理和工具链切换功能。

### 配置环境变量

安装后，将 Cargo 的 bin 目录添加到 PATH：

编辑 `~/.zshrc` 或 `~/.bash_profile`：

```bash
source "$HOME/.cargo/env"
```

使配置生效：

```bash
source ~/.zshrc
# 或
source ~/.bash_profile
```

### 验证安装

```bash
rustc --version
cargo --version
rustup --version
```

## Linux 安装

### Ubuntu/Debian

```bash
# 更新包列表
sudo apt update

# 安装必要的依赖
sudo apt install build-essential curl

# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 配置环境
source "$HOME/.cargo/env"
```

### CentOS/RHEL/Fedora

```bash
# 安装必要的依赖
sudo dnf groupinstall "Development Tools"
sudo dnf install curl

# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 配置环境
source "$HOME/.cargo/env"
```

### Arch Linux

```bash
# 方法 1：使用 rustup（推荐）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 方法 2：使用 pacman
sudo pacman -S rust
```

### 验证安装

```bash
rustc --version
cargo --version
echo $PATH | grep cargo
```

## 工具链管理

### rustup 常用命令

```bash
# 更新所有工具链
rustup update

# 查看已安装的工具链
rustup show

# 安装特定版本
rustup install stable
rustup install nightly
rustup install 1.70.0

# 切换默认版本
rustup default stable
rustup default nightly

# 添加组件
rustup component add rustfmt
rustup component add clippy
rustup component add rust-src
rustup component add rust-analyzer

# 添加目标平台（交叉编译）
rustup target add wasm32-unknown-unknown
rustup target add x86_64-unknown-linux-gnu
```

### 项目级版本指定

在项目根目录创建 `rust-toolchain.toml`：

```toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
targets = ["wasm32-unknown-unknown"]
```

或者创建简单的 `rust-toolchain` 文件：

```
stable
```

## IDE 配置

### VS Code（推荐）

1. 安装 VS Code
2. 安装扩展：**rust-analyzer**
3. 安装扩展：**Even Better TOML**（可选，用于 Cargo.toml）
4. 安装扩展：**crates**（可选，用于依赖版本检查）

**settings.json 配置：**

```json
{
  "rust-analyzer.check.command": "clippy",
  "rust-analyzer.cargo.features": "all",
  "[rust]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "rust-lang.rust-analyzer"
  }
}
```

### IntelliJ IDEA / CLion

1. 安装 JetBrains 的 **Rust 插件**
2. 打开 **Settings** → **Languages & Frameworks** → **Rust**
3. 配置 Rust 工具链路径

### Neovim

使用 LSP 配置：

```lua
-- 使用 nvim-lspconfig
require('lspconfig').rust_analyzer.setup {
  settings = {
    ['rust-analyzer'] = {
      checkOnSave = {
        command = "clippy",
      },
    },
  },
}
```

## 开发工具

### 代码格式化：rustfmt

```bash
# 安装
rustup component add rustfmt

# 格式化当前项目
cargo fmt

# 检查格式（不修改）
cargo fmt -- --check
```

**配置文件 `rustfmt.toml`：**

```toml
max_width = 100
tab_spaces = 4
edition = "2021"
```

### 代码检查：clippy

```bash
# 安装
rustup component add clippy

# 运行检查
cargo clippy

# 严格模式（所有警告视为错误）
cargo clippy -- -D warnings

# 自动修复
cargo clippy --fix
```

### 文档生成

```bash
# 生成文档
cargo doc

# 生成并打开文档
cargo doc --open

# 包含私有项
cargo doc --document-private-items
```

### 代码分析：rust-analyzer

```bash
# 安装（作为 rustup 组件）
rustup component add rust-analyzer
```

## 第一个 Rust 程序

### 创建项目

```bash
# 创建新项目
cargo new hello_rust
cd hello_rust
```

### 项目结构

```
hello_rust/
├── Cargo.toml    # 项目配置和依赖
└── src/
    └── main.rs   # 主程序入口
```

### 编写代码

**src/main.rs：**

```rust
fn main() {
    println!("Hello, Rust! 🦀");

    // 变量
    let name = "World";
    println!("Hello, {}!", name);

    // 函数调用
    let result = add(2, 3);
    println!("2 + 3 = {}", result);
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

### 构建和运行

```bash
# 检查代码（不生成二进制）
cargo check

# 编译（Debug 模式）
cargo build

# 编译并运行
cargo run

# 编译（Release 模式，优化）
cargo build --release
cargo run --release
```

### 运行测试

```bash
# 运行所有测试
cargo test

# 运行特定测试
cargo test test_name

# 显示输出
cargo test -- --show-output
```

## 常用 Cargo 命令

```bash
# 项目管理
cargo new <name>        # 创建新项目
cargo new <name> --lib  # 创建库项目
cargo init              # 在当前目录初始化

# 构建和运行
cargo build             # 编译
cargo build --release   # 发布编译
cargo run               # 编译并运行
cargo check             # 快速检查

# 测试和文档
cargo test              # 运行测试
cargo doc               # 生成文档
cargo bench             # 运行基准测试

# 依赖管理
cargo add <crate>       # 添加依赖
cargo update            # 更新依赖
cargo tree              # 显示依赖树

# 代码质量
cargo fmt               # 格式化代码
cargo clippy            # 代码检查
cargo audit             # 安全审查（需安装）

# 发布
cargo publish           # 发布到 crates.io
cargo login             # 登录 crates.io
```

## 常见问题

### Q1: 提示 "rustc" 不是内部或外部命令？

**原因**：环境变量未正确配置

**解决**：

1. 确认 `~/.cargo/bin` 在 PATH 中
2. 重新运行 `source "$HOME/.cargo/env"`
3. 重新打开终端

### Q2: 编译时提示缺少链接器？

**Windows 解决**：

安装 Visual Studio Build Tools 或 MinGW-w64

**Linux 解决**：

```bash
# Ubuntu/Debian
sudo apt install build-essential

# CentOS/RHEL
sudo dnf groupinstall "Development Tools"
```

### Q3: cargo build 很慢？

**解决方案**：

1. 使用增量编译（默认开启）
2. 使用 `cargo check` 代替 `cargo build` 进行快速检查
3. 安装 sccache 加速编译：

```bash
cargo install sccache
export RUSTC_WRAPPER=sccache
```

### Q4: 如何更新 Rust？

```bash
rustup update
```

### Q5: 如何卸载 Rust？

```bash
rustup self uninstall
```

## 推荐配置

### ~/.cargo/config.toml

```toml
[build]
# 使用更多并行任务
jobs = 8

[net]
# 使用稀疏索引（更快）
git-fetch-with-cli = true

[registries.crates-io]
protocol = "sparse"

# 国内镜像（可选）
# [source.crates-io]
# replace-with = 'ustc'
#
# [source.ustc]
# registry = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"
```

### .gitignore

```gitignore
/target/
Cargo.lock
**/*.rs.bk
```

> [!TIP]
> 对于二进制项目，建议提交 `Cargo.lock`；对于库项目，建议在 `.gitignore` 中忽略它。

## 下一步

环境配置完成后，开始学习 [Rust 基础语法](/docs/rust/basic-syntax)！

## 相关资源

- [Rust 官方文档](https://doc.rust-lang.org/)
- [The Rust Book](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
- [Rustlings](https://github.com/rust-lang/rustlings/) - 交互式练习
- [crates.io](https://crates.io/) - Rust 包仓库
