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

## ⚡ 最佳实践

1. **使用 typename 或 class** - 模板参数关键字
2. **编译期计算** - 利用 constexpr
3. **SFINAE** - 控制模板实例化
4. **Concepts (C++20)** - 约束模板参数
