---
sidebar_position: 4
title: 函数
---

# C++ 函数

C++ 函数是代码复用的基本单元，提供了强大的特性如重载、默认参数和内联函数。

## 🎯 函数基础

### 函数定义

```cpp
#include <iostream>

// 函数声明（原型）
int add(int a, int b);

// 函数定义
int add(int a, int b) {
    return a + b;
}

// 无返回值函数
void greet(const std::string& name) {
    std::cout << "Hello, " << name << "!" << std::endl;
}

// 无参数函数
int getRandomNumber() {
    return 42;
}

int main() {
    std::cout << add(3, 5) << std::endl;  // 8
    greet("World");                        // Hello, World!
    return 0;
}
```

### 参数传递方式

```cpp
#include <iostream>

// 值传递（拷贝）
void byValue(int x) {
    x = 100;  // 不影响原变量
}

// 引用传递
void byReference(int& x) {
    x = 100;  // 修改原变量
}

// 常量引用（只读，避免拷贝）
void byConstRef(const std::string& str) {
    std::cout << str << std::endl;
    // str = "new";  // 错误：不能修改
}

// 指针传递
void byPointer(int* ptr) {
    if (ptr) {
        *ptr = 100;
    }
}

int main() {
    int a = 10;

    byValue(a);
    std::cout << "byValue: " << a << std::endl;     // 10

    byReference(a);
    std::cout << "byReference: " << a << std::endl; // 100

    byPointer(&a);
    std::cout << "byPointer: " << a << std::endl;   // 100

    return 0;
}
```

## 🔄 函数重载

C++ 允许同名函数有不同的参数列表：

```cpp
#include <iostream>
#include <string>

// 同名函数，不同参数
int add(int a, int b) {
    return a + b;
}

double add(double a, double b) {
    return a + b;
}

int add(int a, int b, int c) {
    return a + b + c;
}

std::string add(const std::string& a, const std::string& b) {
    return a + b;
}

int main() {
    std::cout << add(1, 2) << std::endl;          // 调用 int 版本
    std::cout << add(1.5, 2.5) << std::endl;      // 调用 double 版本
    std::cout << add(1, 2, 3) << std::endl;       // 调用三参数版本
    std::cout << add("Hello", "World") << std::endl;
    return 0;
}
```

:::warning 重载注意事项

- 仅返回类型不同不能重载
- 参数类型、数量或顺序必须不同
  :::

## 📋 默认参数

```cpp
#include <iostream>

// 默认参数从右往左
void printMessage(const std::string& msg, int times = 1, bool newline = true) {
    for (int i = 0; i < times; i++) {
        std::cout << msg;
        if (newline) std::cout << std::endl;
    }
}

// 只能在声明或定义中指定一次默认值
void greet(const std::string& name = "World");

void greet(const std::string& name) {
    std::cout << "Hello, " << name << "!" << std::endl;
}

int main() {
    printMessage("Hi");              // Hi (1次，换行)
    printMessage("Hi", 3);           // Hi Hi Hi (3次，换行)
    printMessage("Hi", 2, false);    // HiHi (2次，不换行)

    greet();           // Hello, World!
    greet("C++");      // Hello, C++!

    return 0;
}
```

## ⚡ 内联函数

建议编译器将函数代码直接插入调用处，减少函数调用开销：

```cpp
#include <iostream>

// inline 关键字
inline int square(int x) {
    return x * x;
}

// 类内定义的函数默认是内联的
class Math {
public:
    int cube(int x) { return x * x * x; }  // 隐式内联
};

// constexpr 函数（隐式内联）
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

int main() {
    std::cout << square(5) << std::endl;     // 25

    // 编译期计算
    constexpr int result = factorial(5);      // 120
    static_assert(result == 120, "Error");

    return 0;
}
```

## 🔁 递归函数

```cpp
#include <iostream>

// 阶乘
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// 斐波那契数列
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// 尾递归优化（编译器可能优化）
int factorialTail(int n, int acc = 1) {
    if (n <= 1) return acc;
    return factorialTail(n - 1, n * acc);
}

int main() {
    std::cout << "5! = " << factorial(5) << std::endl;
    std::cout << "Fib(10) = " << fibonacci(10) << std::endl;
    return 0;
}
```

## 🎭 函数指针

```cpp
#include <iostream>

// 普通函数
int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }
int multiply(int a, int b) { return a * b; }

// 函数指针类型
typedef int (*Operation)(int, int);
// 或使用 using (C++11)
using Operation2 = int(*)(int, int);

// 接受函数指针的函数
int calculate(int a, int b, Operation op) {
    return op(a, b);
}

int main() {
    // 声明函数指针
    int (*funcPtr)(int, int) = add;
    std::cout << funcPtr(3, 4) << std::endl;  // 7

    // 使用 typedef
    Operation op = multiply;
    std::cout << op(3, 4) << std::endl;  // 12

    // 传递函数指针
    std::cout << calculate(10, 5, add) << std::endl;       // 15
    std::cout << calculate(10, 5, subtract) << std::endl;  // 5

    // 函数指针数组
    Operation ops[] = {add, subtract, multiply};
    for (auto op : ops) {
        std::cout << op(6, 2) << " ";  // 8 4 12
    }

    return 0;
}
```

## 📦 返回多个值

```cpp
#include <iostream>
#include <tuple>
#include <utility>

// 方法1：使用引用参数
void divide(int a, int b, int& quotient, int& remainder) {
    quotient = a / b;
    remainder = a % b;
}

// 方法2：使用 std::pair
std::pair<int, int> divideWithPair(int a, int b) {
    return {a / b, a % b};
}

// 方法3：使用 std::tuple
std::tuple<int, int, bool> divideWithTuple(int a, int b) {
    if (b == 0) {
        return {0, 0, false};
    }
    return {a / b, a % b, true};
}

// 方法4：使用结构体
struct DivResult {
    int quotient;
    int remainder;
};

DivResult divideWithStruct(int a, int b) {
    return {a / b, a % b};
}

int main() {
    // 方法1
    int q, r;
    divide(17, 5, q, r);
    std::cout << q << ", " << r << std::endl;  // 3, 2

    // 方法2
    auto [q2, r2] = divideWithPair(17, 5);  // C++17 结构化绑定
    std::cout << q2 << ", " << r2 << std::endl;

    // 方法3
    auto [q3, r3, ok] = divideWithTuple(17, 5);
    if (ok) {
        std::cout << q3 << ", " << r3 << std::endl;
    }

    // 方法4
    auto result = divideWithStruct(17, 5);
    std::cout << result.quotient << ", " << result.remainder << std::endl;

    return 0;
}
```

## 🛡️ 函数属性

### noexcept

```cpp
// 承诺不抛出异常
void safeFunction() noexcept {
    // 不会抛出异常的代码
}

// 条件性 noexcept
template<typename T>
void process(T& t) noexcept(noexcept(t.doSomething())) {
    t.doSomething();
}
```

### [[nodiscard]] (C++17)

```cpp
// 警告调用者不要忽略返回值
[[nodiscard]] int computeValue() {
    return 42;
}

int main() {
    computeValue();  // 编译警告：忽略了返回值
    int v = computeValue();  // OK
    return 0;
}
```

### [[deprecated]] (C++14)

```cpp
// 标记函数已弃用
[[deprecated("Use newFunction() instead")]]
void oldFunction() {
    // ...
}
```

## 🔧 实用技巧

### 可变参数模板

```cpp
#include <iostream>

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

int main() {
    print(1, 2.5, "hello", 'c');  // 1 2.5 hello c
    return 0;
}
```

### 折叠表达式 (C++17)

```cpp
#include <iostream>

template<typename... Args>
auto sum(Args... args) {
    return (args + ...);  // 右折叠
}

template<typename... Args>
void printAll(Args... args) {
    ((std::cout << args << " "), ...);  // 逗号折叠
    std::cout << std::endl;
}

int main() {
    std::cout << sum(1, 2, 3, 4, 5) << std::endl;  // 15
    printAll(1, 2.5, "hello");  // 1 2.5 hello
    return 0;
}
```

## ⚡ 最佳实践

1. **优先使用 const 引用** - 避免不必要的拷贝
2. **函数职责单一** - 一个函数只做一件事
3. **避免过长函数** - 保持函数简短易读
4. **使用 nodiscard** - 重要返回值不应被忽略
5. **谨慎使用函数指针** - 优先考虑 Lambda 或 std::function

掌握了函数特性，你已迈向 C++ 进阶之路！🚀
