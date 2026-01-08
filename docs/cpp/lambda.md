---
sidebar_position: 17
title: Lambda 表达式
---

# C++ Lambda 表达式

Lambda 是 C++11 引入的匿名函数，简化回调和算法使用。

## 🎯 基本语法

```cpp
// [捕获列表](参数列表) -> 返回类型 { 函数体 }
auto add = [](int a, int b) { return a + b; };
std::cout << add(3, 5) << std::endl;  // 8

// 自动推导返回类型
auto square = [](int x) { return x * x; };

// 显式指定返回类型
auto divide = [](double a, double b) -> double {
    if (b == 0) return 0;
    return a / b;
};
```

## 📦 捕获方式

```cpp
int x = 10, y = 20;

// 值捕获
auto f1 = [x]() { return x; };

// 引用捕获
auto f2 = [&x]() { x++; };

// 隐式值捕获所有
auto f3 = [=]() { return x + y; };

// 隐式引用捕获所有
auto f4 = [&]() { x++; y++; };

// 混合捕获
auto f5 = [=, &x]() { x++; return y; };
auto f6 = [&, x]() { y++; return x; };

// 初始化捕获 (C++14)
auto f7 = [z = x + y]() { return z; };
auto f8 = [ptr = std::make_unique<int>(10)]() { return *ptr; };
```

## 🔄 mutable Lambda

```cpp
int x = 10;
// 值捕获默认不可修改
auto f1 = [x]() mutable {
    x++;  // 修改的是副本
    return x;
};
std::cout << f1() << std::endl;  // 11
std::cout << x << std::endl;     // 10 (原值不变)
```

## 🎭 泛型 Lambda (C++14)

```cpp
// auto 参数
auto print = [](const auto& x) {
    std::cout << x << std::endl;
};

print(42);
print("Hello");
print(3.14);

// 多参数
auto add = [](auto a, auto b) { return a + b; };
```

## 📋 与 STL 配合

```cpp
std::vector<int> nums = {3, 1, 4, 1, 5, 9};

// 排序
std::sort(nums.begin(), nums.end(), [](int a, int b) {
    return a > b;  // 降序
});

// 查找
auto it = std::find_if(nums.begin(), nums.end(),
    [](int x) { return x > 4; });

// 遍历
std::for_each(nums.begin(), nums.end(),
    [](int x) { std::cout << x << " "; });

// 变换
std::transform(nums.begin(), nums.end(), nums.begin(),
    [](int x) { return x * 2; });
```

## ⚡ 最佳实践

1. **优先使用 Lambda** - 替代简单函数对象
2. **明确捕获列表** - 避免隐式捕获
3. **使用引用捕获** - 避免大对象拷贝
4. **注意生命周期** - 引用捕获不能超出范围
