---
sidebar_position: 22
title: C++17 新特性
---

# C++17 新特性

C++17 带来了许多实用的新特性和库组件。

## 🎯 结构化绑定

```cpp
// 数组
int arr[] = {1, 2, 3};
auto [a, b, c] = arr;

// pair/tuple
auto [x, y] = std::make_pair(1, 2);
auto [p, q, r] = std::make_tuple(1, 2.0, "three");

// 结构体
struct Point { int x, y; };
Point pt{10, 20};
auto [px, py] = pt;

// map 遍历
std::map<std::string, int> m;
for (const auto& [key, value] : m) {
    std::cout << key << ": " << value << std::endl;
}
```

## 📦 if/switch 初始化

```cpp
// if 带初始化
if (auto it = m.find("key"); it != m.end()) {
    std::cout << it->second << std::endl;
}

// switch 带初始化
switch (auto val = getValue(); val) {
    case 1: break;
    case 2: break;
    default: break;
}
```

## 🔄 if constexpr

```cpp
template<typename T>
auto getValue(T t) {
    if constexpr (std::is_pointer_v<T>) {
        return *t;
    } else {
        return t;
    }
}
```

## 📋 std::optional

```cpp
std::optional<int> find(int x) {
    if (x > 0) return x;
    return std::nullopt;
}

auto result = find(5);
if (result) {
    std::cout << *result << std::endl;
}
std::cout << result.value_or(-1) << std::endl;
```

## 🔧 std::variant

```cpp
std::variant<int, double, std::string> v;
v = 42;
v = 3.14;
v = "hello";

std::visit([](auto&& arg) {
    std::cout << arg << std::endl;
}, v);
```

## 📜 std::string_view

```cpp
void print(std::string_view sv) {
    std::cout << sv << std::endl;
}
print("Hello");  // 无拷贝
```

## 🗂️ 文件系统库

```cpp
#include <filesystem>
namespace fs = std::filesystem;

fs::path p = "/home/user/file.txt";
if (fs::exists(p)) {
    std::cout << fs::file_size(p) << std::endl;
}

for (const auto& entry : fs::directory_iterator("/home")) {
    std::cout << entry.path() << std::endl;
}
```

## ⚡ 其他特性

- **折叠表达式** - 简化可变参数模板
- **内联变量** - 头文件中定义变量
- **[[nodiscard]]** - 警告忽略返回值
- **[[maybe_unused]]** - 抑制未使用警告
