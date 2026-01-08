---
sidebar_position: 5
title: 数组和字符串
---

# C++ 数组和字符串

C++ 提供了多种处理数组和字符串的方式，从 C 风格到现代 STL 容器。

## 📊 C 风格数组

```cpp
#include <iostream>

int main() {
    // 声明和初始化
    int arr1[5] = {1, 2, 3, 4, 5};
    int arr2[] = {1, 2, 3};        // 自动推断大小
    int arr3[5] = {};              // 全部初始化为 0

    // 访问和遍历
    std::cout << arr1[0] << std::endl;

    for (int x : arr1) {
        std::cout << x << " ";
    }

    // 二维数组
    int matrix[3][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };

    return 0;
}
```

## 📦 std::array (C++11)

更安全的固定大小数组：

```cpp
#include <array>
#include <algorithm>

int main() {
    std::array<int, 5> arr = {1, 2, 3, 4, 5};

    // 访问
    arr[0] = 10;
    arr.at(1) = 20;  // 带边界检查

    // 属性
    arr.size();
    arr.empty();
    arr.front();
    arr.back();

    // 算法
    std::sort(arr.begin(), arr.end());

    return 0;
}
```

## 🔗 std::vector

动态大小数组，最常用的容器：

```cpp
#include <vector>
#include <algorithm>

int main() {
    // 创建
    std::vector<int> v1;
    std::vector<int> v2(5, 10);       // 5 个 10
    std::vector<int> v3 = {1, 2, 3};

    // 添加元素
    v1.push_back(10);
    v1.emplace_back(20);  // 原地构造

    // 访问
    v3[0];
    v3.at(1);
    v3.front();
    v3.back();

    // 大小和容量
    v3.size();
    v3.capacity();
    v3.reserve(100);

    // 修改
    v3.insert(v3.begin(), 0);
    v3.erase(v3.begin());
    v3.pop_back();
    v3.clear();

    // 二维 vector
    std::vector<std::vector<int>> mat(3, std::vector<int>(4, 0));

    return 0;
}
```

## 🔤 C 风格字符串

```cpp
#include <cstring>

int main() {
    char str[] = "Hello";
    const char* ptr = "World";

    strlen(str);              // 长度
    strcpy(dest, src);        // 拷贝
    strcat(dest, src);        // 连接
    strcmp(s1, s2);           // 比较
    strchr(str, 'l');         // 查找字符
    strstr(str, "llo");       // 查找子串

    return 0;
}
```

## 📜 std::string

现代 C++ 推荐的字符串类型：

```cpp
#include <string>

int main() {
    // 创建
    std::string s1 = "Hello";
    std::string s2(5, 'a');   // "aaaaa"
    std::string s3 = s1 + " World";

    // 访问
    s1[0];
    s1.at(1);
    s1.front();
    s1.back();

    // 修改
    s1.push_back('!');
    s1.append(" World");
    s1 += "!";
    s1.insert(5, ",");
    s1.erase(0, 2);
    s1.replace(0, 5, "Hi");

    // 查找
    s1.find("World");
    s1.rfind("o");
    s1.substr(0, 5);

    // 比较
    s1 == s2;
    s1.compare(s2);

    // 转换
    s1.c_str();           // 转 C 字符串
    std::stoi("42");      // 字符串转整数
    std::to_string(42);   // 整数转字符串

    return 0;
}
```

## 🔡 std::string_view (C++17)

轻量级只读字符串视图：

```cpp
#include <string_view>

void print(std::string_view sv) {
    std::cout << sv << std::endl;
}

int main() {
    std::string str = "Hello";
    std::string_view sv = str;

    sv.substr(0, 3);      // 无拷贝子串
    sv.remove_prefix(1);  // 移除前缀

    print(str);
    print("Literal");

    return 0;
}
```

## ⚡ 最佳实践

1. **优先使用 std::string** - 而非 C 风格字符串
2. **使用 std::vector** - 替代 C 风格动态数组
3. **使用 std::array** - 替代固定大小数组
4. **使用 string_view** - 只读访问，避免拷贝
5. **使用 at()** - 需要边界检查时
