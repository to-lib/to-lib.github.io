---
sidebar_position: 19
title: 异常处理
---

# C++ 异常处理

异常处理是 C++ 错误处理的重要机制。

## 🎯 基本语法

```cpp
#include <stdexcept>
#include <iostream>

double divide(double a, double b) {
    if (b == 0) {
        throw std::runtime_error("除数不能为零");
    }
    return a / b;
}

int main() {
    try {
        double result = divide(10, 0);
        std::cout << result << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "运行时错误: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "未知异常" << std::endl;
    }

    return 0;
}
```

## 📦 标准异常类

```cpp
// 常用异常类
std::exception         // 基类
std::runtime_error     // 运行时错误
std::logic_error       // 逻辑错误
std::invalid_argument  // 无效参数
std::out_of_range      // 越界
std::bad_alloc         // 内存分配失败
```

## 🔧 自定义异常

```cpp
class MyException : public std::exception {
private:
    std::string message;

public:
    MyException(const std::string& msg) : message(msg) {}

    const char* what() const noexcept override {
        return message.c_str();
    }
};

void test() {
    throw MyException("自定义错误");
}
```

## 🛡️ noexcept

```cpp
// 承诺不抛出异常
void safeFunc() noexcept {
    // 不会抛出异常
}

// 条件性 noexcept
template<typename T>
void process(T& t) noexcept(noexcept(t.doSomething())) {
    t.doSomething();
}

// 移动操作应该标记为 noexcept
class Widget {
public:
    Widget(Widget&& other) noexcept;
    Widget& operator=(Widget&& other) noexcept;
};
```

## 📋 RAII 与异常安全

```cpp
class Resource {
public:
    Resource() { /* 获取资源 */ }
    ~Resource() { /* 释放资源 */ }
};

void example() {
    Resource r;  // RAII

    throw std::runtime_error("Error");
    // r 的析构函数仍会被调用
}
```

## ⚡ 最佳实践

1. **按引用捕获** - `catch (const Exception& e)`
2. **使用标准异常** - 或继承自 std::exception
3. **标记 noexcept** - 不抛异常的函数
4. **RAII** - 确保资源安全释放
5. **不在析构函数中抛异常**
