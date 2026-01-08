---
sidebar_position: 2
title: 开发环境配置
---

# C++ 开发环境配置

本文介绍如何在不同操作系统上配置 C++ 开发环境。

## 🖥️ 编译器安装

### macOS

```bash
# 安装 Xcode 命令行工具（包含 clang++）
xcode-select --install

# 验证安装
clang++ --version

# 或使用 Homebrew 安装 GCC
brew install gcc
g++-13 --version
```

### Linux (Ubuntu/Debian)

```bash
# 安装 GCC/G++
sudo apt update
sudo apt install build-essential

# 验证安装
g++ --version

# 安装 Clang（可选）
sudo apt install clang
```

### Linux (CentOS/RHEL)

```bash
# 安装开发工具
sudo yum groupinstall "Development Tools"

# 或使用 dnf
sudo dnf install gcc-c++
```

### Windows

#### 选项 1: Visual Studio

1. 下载 [Visual Studio](https://visualstudio.microsoft.com/)
2. 安装时选择 "Desktop development with C++"
3. 使用 Developer Command Prompt 编译

#### 选项 2: MinGW-w64

1. 下载 [MinGW-w64](https://www.mingw-w64.org/)
2. 添加 `bin` 目录到 PATH
3. 验证：`g++ --version`

#### 选项 3: WSL

```bash
# 在 PowerShell 中安装 WSL
wsl --install

# 进入 WSL 后安装 GCC
sudo apt install build-essential
```

## 🛠️ IDE 与编辑器

### Visual Studio Code

推荐的轻量级编辑器配置：

1. 安装 VS Code
2. 安装扩展：

   - **C/C++** (Microsoft)
   - **C/C++ Extension Pack**
   - **CMake Tools**

3. 创建 `.vscode/tasks.json`：

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "build",
      "type": "shell",
      "command": "g++",
      "args": [
        "-std=c++17",
        "-g",
        "-Wall",
        "-Wextra",
        "${file}",
        "-o",
        "${fileDirname}/${fileBasenameNoExtension}"
      ],
      "group": {
        "kind": "build",
        "isDefault": true
      }
    }
  ]
}
```

4. 创建 `.vscode/launch.json`（调试配置）：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug",
      "type": "cppdbg",
      "request": "launch",
      "program": "${fileDirname}/${fileBasenameNoExtension}",
      "args": [],
      "stopAtEntry": false,
      "cwd": "${workspaceFolder}",
      "environment": [],
      "externalConsole": false,
      "MIMode": "gdb"
    }
  ]
}
```

### CLion

JetBrains 出品的专业 C++ IDE：

1. 下载安装 [CLion](https://www.jetbrains.com/clion/)
2. 配置工具链（自动检测或手动指定编译器路径）
3. 使用 CMake 管理项目

### Visual Studio (Windows)

1. 创建新项目 → Console App (C++)
2. 编写代码
3. F5 运行/调试

## 📦 构建工具

### CMake

跨平台构建系统，强烈推荐使用：

```bash
# 安装
# macOS
brew install cmake

# Ubuntu
sudo apt install cmake

# Windows
# 下载安装包
```

基本 `CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.16)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加可执行文件
add_executable(main main.cpp)

# 添加编译警告
target_compile_options(main PRIVATE -Wall -Wextra)
```

构建项目：

```bash
mkdir build && cd build
cmake ..
make
./main
```

### Make

传统构建工具，`Makefile` 示例：

```makefile
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g

TARGET = main
SRCS = main.cpp utils.cpp
OBJS = $(SRCS:.cpp=.o)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: clean
```

## 🔧 调试工具

### GDB

```bash
# 安装 (Ubuntu)
sudo apt install gdb

# 编译带调试信息
g++ -g main.cpp -o main

# 启动调试
gdb ./main

# 常用命令
# break main    - 在 main 函数设置断点
# run           - 运行程序
# next (n)      - 下一行
# step (s)      - 进入函数
# print var     - 打印变量
# backtrace     - 查看调用栈
# continue (c)  - 继续运行
# quit          - 退出
```

### LLDB (macOS)

```bash
lldb ./main

# 命令与 GDB 类似
# breakpoint set --name main
# run
# next
# step
# print var
```

### Valgrind

内存检测工具：

```bash
# 安装 (Ubuntu)
sudo apt install valgrind

# 检测内存泄漏
valgrind --leak-check=full ./main
```

### AddressSanitizer

编译时内存检测：

```bash
g++ -fsanitize=address -g main.cpp -o main
./main
```

## 📝 编辑器配置文件

### .clang-format

代码格式化配置：

```yaml
BasedOnStyle: Google
IndentWidth: 4
ColumnLimit: 100
AllowShortFunctionsOnASingleLine: Empty
```

使用：

```bash
clang-format -i main.cpp
```

### .clang-tidy

静态分析配置：

```yaml
Checks: "clang-analyzer-*,modernize-*,performance-*"
WarningsAsErrors: ""
```

## ✅ 验证环境

创建测试文件 `test.cpp`：

```cpp
#include <iostream>
#include <vector>
#include <string>

int main() {
    std::vector<std::string> items = {"C++17", "已就绪", "🎉"};

    for (const auto& item : items) {
        std::cout << item << " ";
    }
    std::cout << std::endl;

    // C++17 特性测试
    if (auto x = 42; x > 0) {
        std::cout << "C++17 if-init 语法正常" << std::endl;
    }

    return 0;
}
```

编译运行：

```bash
g++ -std=c++17 test.cpp -o test && ./test
```

输出：

```
C++17 已就绪 🎉
C++17 if-init 语法正常
```

环境配置完成！🚀
