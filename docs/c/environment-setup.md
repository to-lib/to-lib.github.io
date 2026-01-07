---
sidebar_position: 2
title: 环境配置
---

# C 语言开发环境配置

搭建一个高效的 C 语言开发环境是学习的第一步。

## 编译器安装

### macOS

macOS 上推荐使用 Clang（通过 Xcode 命令行工具）或 GCC。

```bash
# 安装 Xcode 命令行工具（包含 Clang）
xcode-select --install

# 验证安装
clang --version
gcc --version  # macOS 上 gcc 实际上是 clang 的别名

# 使用 Homebrew 安装真正的 GCC
brew install gcc
gcc-13 --version  # 版本号可能不同
```

### Linux (Ubuntu/Debian)

```bash
# 安装 build-essential 包
sudo apt update
sudo apt install build-essential

# 验证安装
gcc --version
make --version

# 安装 Clang（可选）
sudo apt install clang

# 安装调试工具
sudo apt install gdb valgrind
```

### Linux (CentOS/RHEL/Fedora)

```bash
# CentOS/RHEL
sudo yum groupinstall "Development Tools"

# Fedora
sudo dnf groupinstall "Development Tools"

# 验证
gcc --version
```

### Windows

#### 方法一：MinGW-w64

1. 下载 [MinGW-w64](https://www.mingw-w64.org/)
2. 运行安装程序
3. 添加 `bin` 目录到系统 PATH
4. 打开命令提示符验证：`gcc --version`

#### 方法二：WSL (推荐)

```powershell
# 在 PowerShell 中启用 WSL
wsl --install

# 安装完成后，在 WSL 中安装 GCC
sudo apt update
sudo apt install build-essential gdb
```

#### 方法三：Visual Studio

1. 下载 [Visual Studio](https://visualstudio.microsoft.com/)
2. 安装时选择 "C++ 桌面开发" 工作负载
3. 使用 Developer Command Prompt

## IDE 和编辑器

### VS Code (推荐)

VS Code 是一个轻量级但功能强大的编辑器。

```bash
# 安装 VS Code 后，安装 C/C++ 扩展
# 在扩展市场搜索 "C/C++" by Microsoft
```

**推荐的 VS Code 扩展：**

| 扩展名               | 功能               |
| -------------------- | ------------------ |
| C/C++                | 智能提示、调试支持 |
| C/C++ Extension Pack | 扩展包             |
| Code Runner          | 快速运行代码       |
| clangd               | 代码补全、诊断     |

**settings.json 配置示例：**

```json
{
  "C_Cpp.default.compilerPath": "/usr/bin/gcc",
  "C_Cpp.default.cStandard": "c17",
  "C_Cpp.default.intelliSenseMode": "gcc-x64",
  "editor.formatOnSave": true
}
```

**tasks.json 配置示例：**

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Build C",
      "type": "shell",
      "command": "gcc",
      "args": [
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

**launch.json 调试配置：**

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug C",
      "type": "cppdbg",
      "request": "launch",
      "program": "${fileDirname}/${fileBasenameNoExtension}",
      "args": [],
      "stopAtEntry": false,
      "cwd": "${workspaceFolder}",
      "environment": [],
      "externalConsole": false,
      "MIMode": "gdb",
      "preLaunchTask": "Build C"
    }
  ]
}
```

### CLion

JetBrains CLion 是专业的 C/C++ IDE。

- 下载：[CLion](https://www.jetbrains.com/clion/)
- 支持 CMake 项目
- 强大的调试和重构功能
- 学生可申请免费许可证

### Vim/Neovim

```bash
# 安装 vim-plug 插件管理器后
# 在 .vimrc 中添加：
Plug 'neoclide/coc.nvim', {'branch': 'release'}  " 代码补全
Plug 'preservim/nerdtree'                         " 文件浏览
Plug 'vim-airline/vim-airline'                    " 状态栏
```

## 调试工具

### GDB (GNU Debugger)

```bash
# 编译时添加调试信息
gcc -g program.c -o program

# 启动 GDB
gdb ./program

# 常用命令
(gdb) run              # 运行程序
(gdb) break main       # 在 main 函数设置断点
(gdb) break 10         # 在第 10 行设置断点
(gdb) next             # 单步执行（不进入函数）
(gdb) step             # 单步执行（进入函数）
(gdb) continue         # 继续执行
(gdb) print x          # 打印变量 x 的值
(gdb) backtrace        # 查看调用栈
(gdb) quit             # 退出
```

### LLDB

```bash
# 类似 GDB，用于 Clang
lldb ./program

(lldb) run
(lldb) breakpoint set --name main
(lldb) breakpoint set --file program.c --line 10
(lldb) next
(lldb) step
(lldb) continue
(lldb) print x
(lldb) bt
(lldb) quit
```

### Valgrind (内存检测)

```bash
# 安装
sudo apt install valgrind  # Linux
brew install valgrind      # macOS (可能不支持)

# 检测内存泄漏
valgrind --leak-check=full ./program

# 检测未初始化内存
valgrind --track-origins=yes ./program
```

## 构建工具

### Make

创建 `Makefile`:

```makefile
# 编译器和选项
CC = gcc
CFLAGS = -Wall -Wextra -g
LDFLAGS =

# 目标文件
TARGET = program
SRCS = main.c utils.c
OBJS = $(SRCS:.c=.o)

# 默认目标
all: $(TARGET)

# 链接
$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

# 编译
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 清理
clean:
	rm -f $(OBJS) $(TARGET)

# 重新构建
rebuild: clean all

.PHONY: all clean rebuild
```

使用：

```bash
make          # 构建项目
make clean    # 清理
make rebuild  # 重新构建
```

### CMake

创建 `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyProject C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 添加编译选项
add_compile_options(-Wall -Wextra)

# 添加可执行文件
add_executable(program main.c utils.c)

# 查找并链接库（示例）
# find_package(Threads REQUIRED)
# target_link_libraries(program Threads::Threads)
```

使用：

```bash
mkdir build && cd build
cmake ..
make
./program
```

## 项目结构

### 推荐的项目结构

```
my_project/
├── CMakeLists.txt    # 或 Makefile
├── README.md
├── .gitignore
├── include/          # 头文件
│   ├── utils.h
│   └── config.h
├── src/              # 源文件
│   ├── main.c
│   └── utils.c
├── tests/            # 测试文件
│   └── test_utils.c
├── lib/              # 第三方库
├── build/            # 构建输出（git ignore）
└── docs/             # 文档
```

### .gitignore 示例

```gitignore
# 编译输出
*.o
*.obj
*.exe
*.out

# 构建目录
build/
cmake-build-*/

# IDE
.vscode/
.idea/
*.swp
*.swo

# 调试文件
*.dSYM/

# 系统文件
.DS_Store
Thumbs.db
```

## 编译选项

### 常用 GCC 选项

| 选项           | 说明               |
| -------------- | ------------------ |
| `-o <file>`    | 指定输出文件名     |
| `-c`           | 只编译不链接       |
| `-g`           | 生成调试信息       |
| `-O0/O1/O2/O3` | 优化级别           |
| `-Wall`        | 开启所有警告       |
| `-Wextra`      | 额外警告           |
| `-Werror`      | 将警告视为错误     |
| `-std=c17`     | 指定 C 标准        |
| `-I<dir>`      | 添加头文件搜索路径 |
| `-L<dir>`      | 添加库搜索路径     |
| `-l<lib>`      | 链接库             |
| `-D<macro>`    | 定义宏             |

### 示例

```bash
# 完整编译
gcc -std=c17 -Wall -Wextra -g -O2 \
    -Iinclude \
    src/main.c src/utils.c \
    -o build/program \
    -lm -lpthread
```

## 验证环境

创建测试程序 `test_env.c`:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    printf("=== C 环境测试 ===\n\n");

    // 编译器信息
    #ifdef __clang__
        printf("编译器: Clang %d.%d.%d\n",
               __clang_major__, __clang_minor__, __clang_patchlevel__);
    #elif defined(__GNUC__)
        printf("编译器: GCC %d.%d.%d\n",
               __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
    #endif

    // C 标准
    #if __STDC_VERSION__ >= 201710L
        printf("C 标准: C17\n");
    #elif __STDC_VERSION__ >= 201112L
        printf("C 标准: C11\n");
    #elif __STDC_VERSION__ >= 199901L
        printf("C 标准: C99\n");
    #else
        printf("C 标准: C89/C90\n");
    #endif

    // 数据类型大小
    printf("\n--- 数据类型大小 ---\n");
    printf("char:      %zu 字节\n", sizeof(char));
    printf("short:     %zu 字节\n", sizeof(short));
    printf("int:       %zu 字节\n", sizeof(int));
    printf("long:      %zu 字节\n", sizeof(long));
    printf("long long: %zu 字节\n", sizeof(long long));
    printf("float:     %zu 字节\n", sizeof(float));
    printf("double:    %zu 字节\n", sizeof(double));
    printf("指针:      %zu 字节\n", sizeof(void*));

    // 动态内存测试
    printf("\n--- 动态内存测试 ---\n");
    int *ptr = malloc(sizeof(int) * 10);
    if (ptr != NULL) {
        printf("内存分配成功\n");
        free(ptr);
        printf("内存释放成功\n");
    }

    printf("\n✅ 环境配置完成！\n");
    return 0;
}
```

```bash
gcc -std=c17 -Wall test_env.c -o test_env && ./test_env
```

环境配置完成后，就可以开始学习 C 语言的基础语法了！
