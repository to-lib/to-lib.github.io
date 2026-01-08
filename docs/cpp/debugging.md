---
sidebar_position: 27
title: 调试技巧
---

# C++ 调试技巧

掌握调试工具和技巧，快速定位和修复问题。

## 🔍 GDB/LLDB 调试器

```bash
# 编译时加入调试信息
g++ -g -O0 main.cpp -o main

# 启动 GDB
gdb ./main

# 常用命令
break main          # 在 main 设置断点
break file.cpp:20   # 在特定行设断点
run                 # 运行程序
next (n)            # 单步（不进入函数）
step (s)            # 单步（进入函数）
continue (c)        # 继续执行
print var           # 打印变量
backtrace (bt)      # 查看调用栈
watch var           # 监视变量变化
quit                # 退出
```

## 🧪 Sanitizers

### AddressSanitizer (内存错误)

```bash
g++ -fsanitize=address -g main.cpp -o main
./main
```

检测：越界访问、use-after-free、内存泄漏

### ThreadSanitizer (数据竞争)

```bash
g++ -fsanitize=thread -g main.cpp -o main
```

### UndefinedBehaviorSanitizer

```bash
g++ -fsanitize=undefined -g main.cpp -o main
```

## 💾 Valgrind

```bash
# 内存泄漏检测
valgrind --leak-check=full ./main

# 内存错误检测
valgrind --tool=memcheck ./main
```

## 📋 调试宏

```cpp
#include <iostream>
#include <cassert>

// 断言
assert(x > 0 && "x must be positive");

// 调试输出宏
#ifdef DEBUG
    #define LOG(msg) std::cerr << __FILE__ << ":" << __LINE__ << " " << msg << std::endl
#else
    #define LOG(msg)
#endif

// 使用
LOG("Value: " << x);
```

## 🔧 静态分析

```bash
# Clang-Tidy
clang-tidy main.cpp -- -std=c++17

# Cppcheck
cppcheck --enable=all main.cpp
```

## ⚡ 调试建议

1. **使用 -Wall -Wextra** - 开启所有警告
2. **使用 Sanitizers** - 运行时错误检测
3. **写单元测试** - 隔离问题
4. **二分法定位** - 缩小问题范围
5. **打印日志** - 追踪执行流程
