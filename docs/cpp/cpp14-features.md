---
sidebar_position: 21
title: C++14 新特性
---

# C++14 新特性

C++14 是 C++11 的小幅改进，完善了语言特性。

## 🎯 泛型 Lambda

```cpp
// auto 参数
auto add = [](auto a, auto b) { return a + b; };

add(1, 2);       // int
add(1.5, 2.5);   // double
add("a"s, "b"s); // string
```

## 📦 Lambda 初始化捕获

```cpp
auto ptr = std::make_unique<int>(42);
auto f = [p = std::move(ptr)]() {
    std::cout << *p << std::endl;
};
```

## 🔄 返回类型推导

```cpp
// 不需要 -> 尾置返回类型
auto factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
```

## 📋 变量模板

```cpp
template<typename T>
constexpr T pi = T(3.1415926535897932385);

auto f = pi<float>;   // float
auto d = pi<double>;  // double
```

## 🔧 其他改进

```cpp
// 二进制字面量
int binary = 0b1010;  // 10

// 数字分隔符
int million = 1'000'000;
double pi = 3.141'592'653;

// [[deprecated]] 属性
[[deprecated("Use newFunc instead")]]
void oldFunc();

// std::make_unique
auto ptr = std::make_unique<int>(42);
```

## ⚡ 总结

C++14 主要是对 C++11 的完善：

- 泛型 Lambda 更灵活
- 返回类型推导更简洁
- 增加实用工具
