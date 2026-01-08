---
sidebar_position: 20
title: C++11 新特性
---

# C++11 新特性

C++11 是现代 C++ 的起点，引入了大量重要特性。

## 🎯 类型推导

```cpp
// auto
auto x = 42;           // int
auto y = 3.14;         // double
auto s = "hello";      // const char*
auto v = std::vector<int>{1, 2, 3};

// decltype
int a = 10;
decltype(a) b = 20;    // int

// 返回类型后置
auto add(int a, int b) -> int {
    return a + b;
}
```

## 📦 统一初始化

```cpp
int x{10};
std::vector<int> v{1, 2, 3};
std::map<std::string, int> m{{"a", 1}, {"b", 2}};

class Widget {
    int value{0};  // 成员初始值
};
```

## 🔄 范围 for 循环

```cpp
std::vector<int> v = {1, 2, 3, 4, 5};
for (int x : v) {
    std::cout << x << " ";
}
for (const auto& x : v) {
    std::cout << x << " ";
}
```

## 🎭 Lambda 表达式

```cpp
auto add = [](int a, int b) { return a + b; };
auto f = [x, &y]() { /* ... */ };
```

## 📋 智能指针

```cpp
auto up = std::unique_ptr<int>(new int(42));
auto sp = std::make_shared<int>(42);
```

## ↔️ 移动语义

```cpp
class Widget {
public:
    Widget(Widget&& other) noexcept;  // 移动构造
    Widget& operator=(Widget&& other) noexcept;
};

std::string s1 = "Hello";
std::string s2 = std::move(s1);
```

## 📌 nullptr

```cpp
int* ptr = nullptr;  // 替代 NULL
```

## 🔧 其他特性

```cpp
// constexpr
constexpr int square(int x) { return x * x; }

// static_assert
static_assert(sizeof(int) >= 4, "int too small");

// enum class
enum class Color { Red, Green, Blue };

// override 和 final
class Derived : public Base {
    void foo() override;
    void bar() final;
};

// 委托构造函数
class Widget {
public:
    Widget() : Widget(0) {}
    Widget(int x) : value(x) {}
};
```

## ⚡ 核心改进

- **右值引用** - 移动语义基础
- **可变参数模板** - 泛型编程增强
- **线程库** - 标准多线程支持
- **正则表达式** - std::regex
