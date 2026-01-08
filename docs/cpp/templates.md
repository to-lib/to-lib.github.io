---
sidebar_position: 13
title: 模板编程
---

# C++ 模板编程

模板是 C++ 泛型编程的核心，允许编写类型无关的代码。

## 🎯 函数模板

```cpp
#include <iostream>

// 函数模板
template<typename T>
T maximum(T a, T b) {
    return (a > b) ? a : b;
}

// 多类型参数
template<typename T, typename U>
auto add(T a, U b) -> decltype(a + b) {
    return a + b;
}

// C++14 返回类型推导
template<typename T, typename U>
auto multiply(T a, U b) {
    return a * b;
}

int main() {
    std::cout << maximum(10, 20) << std::endl;      // int
    std::cout << maximum(3.14, 2.71) << std::endl;  // double
    std::cout << maximum<double>(10, 3.14) << std::endl;  // 显式指定

    std::cout << add(1, 2.5) << std::endl;  // 3.5
    return 0;
}
```

## 📦 类模板

```cpp
template<typename T>
class Stack {
private:
    std::vector<T> data;

public:
    void push(const T& value) {
        data.push_back(value);
    }

    T pop() {
        if (data.empty()) throw std::runtime_error("Empty");
        T value = data.back();
        data.pop_back();
        return value;
    }

    bool empty() const { return data.empty(); }
    size_t size() const { return data.size(); }
};

int main() {
    Stack<int> intStack;
    intStack.push(1);
    intStack.push(2);

    Stack<std::string> strStack;
    strStack.push("Hello");

    return 0;
}
```

## 🔧 模板特化

```cpp
// 通用模板
template<typename T>
class Printer {
public:
    void print(const T& value) {
        std::cout << value << std::endl;
    }
};

// 完全特化
template<>
class Printer<bool> {
public:
    void print(bool value) {
        std::cout << (value ? "true" : "false") << std::endl;
    }
};

// 偏特化（指针类型）
template<typename T>
class Printer<T*> {
public:
    void print(T* value) {
        if (value) std::cout << *value << std::endl;
        else std::cout << "nullptr" << std::endl;
    }
};
```

## 📋 非类型模板参数

```cpp
template<typename T, size_t N>
class Array {
private:
    T data[N];

public:
    T& operator[](size_t i) { return data[i]; }
    constexpr size_t size() const { return N; }
};

int main() {
    Array<int, 5> arr;
    arr[0] = 10;
    return 0;
}
```

## 🔄 可变参数模板

```cpp
// 递归终止
void print() {
    std::cout << std::endl;
}

// 可变参数模板
template<typename T, typename... Args>
void print(T first, Args... rest) {
    std::cout << first << " ";
    print(rest...);
}

// 折叠表达式 (C++17)
template<typename... Args>
auto sum(Args... args) {
    return (args + ...);
}

int main() {
    print(1, 2.5, "hello", 'c');  // 1 2.5 hello c
    std::cout << sum(1, 2, 3, 4) << std::endl;  // 10
    return 0;
}
```

## 🎭 SFINAE (替换失败不是错误)

SFINAE 允许在模板实例化失败时，选择其他重载而不是报错。

### enable_if

```cpp
#include <type_traits>

// 只对整数类型启用
template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
double_value(T x) {
    return x * 2;
}

// 只对浮点类型启用
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
double_value(T x) {
    return x * 2.0;
}

// C++14 简化写法
template<typename T>
std::enable_if_t<std::is_integral_v<T>, T>
triple_value(T x) {
    return x * 3;
}
```

### void_t 检测成员

```cpp
#include <type_traits>

// 检测类型是否有 size() 方法
template<typename, typename = void>
struct has_size : std::false_type {};

template<typename T>
struct has_size<T, std::void_t<decltype(std::declval<T>().size())>>
    : std::true_type {};

static_assert(has_size<std::vector<int>>::value, "");
static_assert(!has_size<int>::value, "");
```

## 🔷 Concepts (C++20)

Concepts 是 C++20 引入的约束模板参数的方式，比 SFINAE 更清晰。

### 定义和使用 Concept

```cpp
#include <concepts>

// 定义 Concept
template<typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::same_as<T>;
};

template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template<typename T>
concept Printable = requires(T t, std::ostream& os) {
    { os << t } -> std::same_as<std::ostream&>;
};
```

### 使用 Concept 约束

```cpp
// 方式1：requires 子句
template<typename T>
    requires Addable<T>
T add(T a, T b) {
    return a + b;
}

// 方式2：Concept 作为类型约束
template<Numeric T>
T multiply(T a, T b) {
    return a * b;
}

// 方式3：简写语法
auto divide(std::floating_point auto a, std::floating_point auto b) {
    return a / b;
}
```

### 标准库 Concepts

```cpp
#include <concepts>

// 常用标准 Concepts
std::integral<T>           // 整数类型
std::floating_point<T>     // 浮点类型
std::signed_integral<T>    // 有符号整数
std::same_as<T, U>         // 类型相同
std::derived_from<T, U>    // T 派生自 U
std::convertible_to<T, U>  // T 可转换为 U
std::invocable<F, Args...> // F 可以用 Args 调用
std::copyable<T>           // 可拷贝
std::movable<T>            // 可移动
std::default_initializable<T>  // 可默认初始化
```

### requires 表达式

```cpp
template<typename T>
concept Container = requires(T t) {
    typename T::value_type;           // 要求有 value_type 类型
    typename T::iterator;             // 要求有 iterator 类型
    { t.begin() } -> std::same_as<typename T::iterator>;
    { t.end() } -> std::same_as<typename T::iterator>;
    { t.size() } -> std::convertible_to<std::size_t>;
    { t.empty() } -> std::same_as<bool>;
};

template<Container C>
void process_container(const C& c) {
    for (const auto& item : c) {
        // ...
    }
}
```

## ⚡ 最佳实践

1. **使用 typename 或 class** - 模板参数关键字
2. **编译期计算** - 利用 constexpr
3. **优先使用 Concepts (C++20)** - 比 SFINAE 更清晰、错误信息更友好
4. **SFINAE** - C++20 前使用，控制模板实例化
5. **限制模板参数** - 使用 static_assert 或 Concepts
