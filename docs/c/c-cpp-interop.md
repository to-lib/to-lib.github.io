---
sidebar_position: 24
title: C 与 C++ 互操作
---

# C 与 C++ 互操作

C 和 C++ 可以很好地协同工作，了解如何正确混合使用。

## extern "C" 基础

### 为什么需要 extern "C"

C++ 使用名称修饰（name mangling）支持函数重载：

```cpp
// C++ 中 add(int, int) 可能被编译为 _Z3addii
int add(int a, int b);
int add(double a, double b);

// C 中没有名称修饰，add 就是 add
```

`extern "C"` 告诉 C++ 编译器使用 C 语言的链接约定。

### 基本用法

```cpp
// 在 C++ 中声明 C 函数
extern "C" {
    int c_function(int x);
    void another_c_func(void);
}

// 或者单个函数
extern "C" int c_function(int x);
```

## 从 C++ 调用 C 代码

### C 库头文件 (math_utils.h)

```c
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

int add(int a, int b);
int subtract(int a, int b);
double divide(double a, double b);

#ifdef __cplusplus
}
#endif

#endif
```

### C 实现 (math_utils.c)

```c
#include "math_utils.h"

int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}

double divide(double a, double b) {
    return b != 0 ? a / b : 0;
}
```

### C++ 使用 (main.cpp)

```cpp
#include <iostream>
#include "math_utils.h"

int main() {
    std::cout << "10 + 5 = " << add(10, 5) << std::endl;
    std::cout << "10 / 3 = " << divide(10, 3) << std::endl;
    return 0;
}
```

### 编译

```bash
gcc -c math_utils.c -o math_utils.o
g++ -c main.cpp -o main.o
g++ math_utils.o main.o -o program
```

## 从 C 调用 C++ 代码

### C++ 实现需要包装

```cpp
// string_utils.cpp
#include <string>
#include <cstring>

// C++ 实现
std::string cpp_reverse(const std::string& s) {
    return std::string(s.rbegin(), s.rend());
}

// C 接口包装
extern "C" {
    char* reverse_string(const char* s) {
        std::string result = cpp_reverse(s);
        char* ret = (char*)malloc(result.length() + 1);
        strcpy(ret, result.c_str());
        return ret;  // 调用者负责释放
    }

    void free_string(char* s) {
        free(s);
    }
}
```

### C++ 头文件 (string_utils.h)

```c
#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

char* reverse_string(const char* s);
void free_string(char* s);

#ifdef __cplusplus
}
#endif

#endif
```

### C 使用 (main.c)

```c
#include <stdio.h>
#include "string_utils.h"

int main(void) {
    char* reversed = reverse_string("Hello");
    printf("反转: %s\n", reversed);
    free_string(reversed);
    return 0;
}
```

### 编译

```bash
g++ -c string_utils.cpp -o string_utils.o
gcc -c main.c -o main.o
g++ string_utils.o main.o -o program  # 用 g++ 链接
```

## C++ 类的 C 接口

### C++ 类

```cpp
// widget.cpp
#include <string>

class Widget {
public:
    Widget(int id) : id_(id), name_("Widget") {}

    void setName(const std::string& name) { name_ = name; }
    std::string getName() const { return name_; }
    int getId() const { return id_; }

private:
    int id_;
    std::string name_;
};

// C 接口
extern "C" {
    typedef void* WidgetHandle;

    WidgetHandle widget_create(int id) {
        return new Widget(id);
    }

    void widget_destroy(WidgetHandle h) {
        delete static_cast<Widget*>(h);
    }

    void widget_set_name(WidgetHandle h, const char* name) {
        static_cast<Widget*>(h)->setName(name);
    }

    const char* widget_get_name(WidgetHandle h) {
        static Widget* w = static_cast<Widget*>(h);
        static std::string name;
        name = w->getName();
        return name.c_str();
    }

    int widget_get_id(WidgetHandle h) {
        return static_cast<Widget*>(h)->getId();
    }
}
```

### C 使用

```c
#include <stdio.h>

typedef void* WidgetHandle;

WidgetHandle widget_create(int id);
void widget_destroy(WidgetHandle h);
void widget_set_name(WidgetHandle h, const char* name);
const char* widget_get_name(WidgetHandle h);
int widget_get_id(WidgetHandle h);

int main(void) {
    WidgetHandle w = widget_create(42);
    widget_set_name(w, "MyWidget");

    printf("ID: %d\n", widget_get_id(w));
    printf("Name: %s\n", widget_get_name(w));

    widget_destroy(w);
    return 0;
}
```

## 回调函数

### C++ 调用带回调的 C 函数

```c
// c_lib.h
typedef void (*Callback)(int result, void* user_data);

void async_compute(int input, Callback cb, void* user_data);
```

```cpp
// main.cpp
#include <iostream>
extern "C" {
    #include "c_lib.h"
}

void my_callback(int result, void* user_data) {
    int* counter = static_cast<int*>(user_data);
    std::cout << "Result: " << result << std::endl;
    (*counter)++;
}

int main() {
    int counter = 0;
    async_compute(42, my_callback, &counter);
    return 0;
}
```

## 常见陷阱

### 1. 异常处理

```cpp
// C++ 异常不能传递到 C 代码
extern "C" int safe_divide(int a, int b) {
    try {
        if (b == 0) throw std::runtime_error("除零");
        return a / b;
    } catch (...) {
        return 0;  // 必须在 C++ 侧捕获
    }
}
```

### 2. 内存管理

```cpp
// C++ new/delete 和 C malloc/free 不能混用
extern "C" {
    // 错误
    char* create_string() {
        return new char[100];  // C 代码用 free 释放会出错
    }

    // 正确
    char* create_string() {
        return (char*)malloc(100);
    }
}
```

### 3. 结构体对齐

```c
// 确保两边使用相同的对齐
#pragma pack(push, 1)
struct SharedData {
    char type;
    int value;
};
#pragma pack(pop)
```

## 最佳实践

1. **使用 `#ifdef __cplusplus`** 保护头文件
2. **C++ 异常不能跨越 C 边界**
3. **使用不透明指针（handle）封装 C++ 对象**
4. **统一内存分配方式**
5. **用 `g++` 链接混合代码**

让 C 和 C++ 和谐共处！🤝
