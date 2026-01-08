---
sidebar_position: 19.5
title: 预处理器
---

# C++ 预处理器

预处理器在编译前处理源代码，用于宏定义、条件编译等。

## 🎯 宏定义

### 对象宏

```cpp
#define PI 3.14159
#define MAX_SIZE 100
#define VERSION "1.0.0"

double area = PI * r * r;
```

### 函数宏

```cpp
#define SQUARE(x) ((x) * (x))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define PRINT_VAR(x) std::cout << #x << " = " << (x) << std::endl

int result = SQUARE(5);  // 25
PRINT_VAR(result);       // result = 25
```

### 宏运算符

```cpp
#define CONCAT(a, b) a##b      // 连接
#define STRINGIFY(x) #x        // 转字符串

int xy = 10;
std::cout << CONCAT(x, y);     // 输出 10
std::cout << STRINGIFY(hello); // 输出 "hello"
```

## 🔀 条件编译

```cpp
#define DEBUG

#ifdef DEBUG
    std::cout << "Debug mode" << std::endl;
#endif

#ifndef RELEASE
    // 非 Release 模式代码
#endif

#if defined(WIN32) || defined(_WIN32)
    // Windows 代码
#elif defined(__linux__)
    // Linux 代码
#elif defined(__APPLE__)
    // macOS 代码
#endif

// 编译器版本检查
#if __cplusplus >= 201703L
    // C++17 或更高
#endif
```

## 📦 头文件保护

```cpp
// 传统方式
#ifndef MY_HEADER_H
#define MY_HEADER_H

// 头文件内容

#endif

// 现代方式 (大多数编译器支持)
#pragma once

// 头文件内容
```

## 📋 预定义宏

```cpp
std::cout << __FILE__ << std::endl;     // 当前文件名
std::cout << __LINE__ << std::endl;     // 当前行号
std::cout << __func__ << std::endl;     // 当前函数名
std::cout << __DATE__ << std::endl;     // 编译日期
std::cout << __TIME__ << std::endl;     // 编译时间
std::cout << __cplusplus << std::endl;  // C++ 标准版本
```

## 🔧 常用指令

```cpp
#include <iostream>      // 系统头文件
#include "myheader.h"    // 用户头文件

#pragma once             // 防止重复包含
#pragma warning(disable: 4996)  // MSVC 禁用警告

#error "Unsupported platform"  // 编译错误
#warning "Deprecated feature"  // 编译警告 (GCC/Clang)

#line 100 "newfile.cpp"  // 修改行号和文件名
```

## ⚠️ 现代 C++ 替代

```cpp
// ❌ 宏常量
#define MAX_SIZE 100

// ✅ 使用 constexpr
constexpr int MAX_SIZE = 100;

// ❌ 宏函数
#define SQUARE(x) ((x) * (x))

// ✅ 使用内联函数或模板
template<typename T>
constexpr T square(T x) { return x * x; }

// ❌ 类型别名宏
#define UINT unsigned int

// ✅ 使用 using
using UINT = unsigned int;
```

## ⚡ 最佳实践

1. **减少宏的使用** - 优先用 constexpr、inline、template
2. **宏名全大写** - 区分宏和普通代码
3. **使用括号** - 宏参数和整体都加括号
4. **使用 #pragma once** - 简洁的头文件保护
5. **条件编译用于平台兼容** - 不用于普通逻辑分支
