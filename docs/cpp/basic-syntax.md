---
sidebar_position: 3
title: 基础语法
---

# C++ 基础语法

掌握 C++ 的基本语法元素，包括变量、数据类型、运算符和控制流。

## 📦 程序结构

### Hello World

```cpp
#include <iostream>  // 输入输出库

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;  // 返回 0 表示程序正常结束
}
```

### 命名空间

```cpp
#include <iostream>

// 使用 std 命名空间
using namespace std;  // 不推荐在头文件中使用

int main() {
    cout << "不需要 std:: 前缀" << endl;
    return 0;
}

// 推荐方式：只引入需要的
using std::cout;
using std::endl;
```

### 自定义命名空间

```cpp
namespace MyLib {
    int version = 1;

    void hello() {
        std::cout << "Hello from MyLib" << std::endl;
    }

    namespace Utils {
        void helper() { /* ... */ }
    }
}

int main() {
    std::cout << MyLib::version << std::endl;
    MyLib::hello();
    MyLib::Utils::helper();
    return 0;
}
```

## 📊 数据类型

### 基本类型

```cpp
#include <iostream>

int main() {
    // 整数类型
    short s = 32767;           // 至少 16 位
    int i = 2147483647;        // 至少 16 位，通常 32 位
    long l = 2147483647L;      // 至少 32 位
    long long ll = 9223372036854775807LL;  // 至少 64 位

    // 无符号整数
    unsigned int ui = 4294967295U;

    // 浮点类型
    float f = 3.14f;           // 单精度
    double d = 3.141592653589; // 双精度
    long double ld = 3.14159265358979323846L;

    // 字符类型
    char c = 'A';
    wchar_t wc = L'中';
    char16_t c16 = u'中';      // C++11
    char32_t c32 = U'😀';      // C++11

    // 布尔类型
    bool flag = true;

    return 0;
}
```

### 类型大小

```cpp
#include <iostream>

int main() {
    std::cout << "char: " << sizeof(char) << " 字节" << std::endl;
    std::cout << "int: " << sizeof(int) << " 字节" << std::endl;
    std::cout << "long: " << sizeof(long) << " 字节" << std::endl;
    std::cout << "long long: " << sizeof(long long) << " 字节" << std::endl;
    std::cout << "float: " << sizeof(float) << " 字节" << std::endl;
    std::cout << "double: " << sizeof(double) << " 字节" << std::endl;
    std::cout << "bool: " << sizeof(bool) << " 字节" << std::endl;
    return 0;
}
```

### 固定宽度整数 (C++11)

```cpp
#include <cstdint>

int main() {
    int8_t  i8  = 127;
    int16_t i16 = 32767;
    int32_t i32 = 2147483647;
    int64_t i64 = 9223372036854775807LL;

    uint8_t  u8  = 255;
    uint16_t u16 = 65535;
    uint32_t u32 = 4294967295U;
    uint64_t u64 = 18446744073709551615ULL;

    return 0;
}
```

## 🔤 变量与常量

### 变量声明

```cpp
int main() {
    // 声明并初始化
    int a = 10;
    int b(20);         // 直接初始化
    int c{30};         // 统一初始化 (C++11)
    int d = {40};      // 拷贝列表初始化

    // 多变量声明
    int x = 1, y = 2, z = 3;

    // auto 自动类型推导 (C++11)
    auto num = 42;        // int
    auto pi = 3.14;       // double
    auto ch = 'A';        // char
    auto flag = true;     // bool

    // decltype 获取类型 (C++11)
    decltype(a) e = 50;   // int

    return 0;
}
```

### 常量

```cpp
#include <iostream>

// 宏定义（不推荐）
#define PI_MACRO 3.14159

// const 常量
const double PI = 3.14159265358979;
const int MAX_SIZE = 100;

// constexpr 编译期常量 (C++11)
constexpr int ARRAY_SIZE = 10;
constexpr double square(double x) { return x * x; }

int main() {
    // 编译期计算
    constexpr double area = PI * square(5.0);

    // 数组大小必须是常量表达式
    int arr[ARRAY_SIZE];

    std::cout << "Area: " << area << std::endl;
    return 0;
}
```

## 🔢 枚举类型

### 传统枚举 (C 风格)

```cpp
enum Color { Red, Green, Blue };  // Red=0, Green=1, Blue=2
enum Status { Success = 1, Failure = -1, Pending = 0 };

Color c = Red;
int value = c;  // 隐式转换为 int
```

### 强类型枚举 (C++11 enum class)

```cpp
enum class Direction {
    Up,
    Down,
    Left,
    Right
};

enum class HttpStatus : int {
    OK = 200,
    NotFound = 404,
    InternalError = 500
};

Direction d = Direction::Up;
// int value = d;  // 错误！不能隐式转换
int value = static_cast<int>(d);  // 显式转换

HttpStatus status = HttpStatus::OK;

// switch 使用
switch (d) {
    case Direction::Up:    break;
    case Direction::Down:  break;
    case Direction::Left:  break;
    case Direction::Right: break;
}
```

:::tip 推荐使用 enum class

- 作用域隔离，避免命名冲突
- 类型安全，不能隐式转换为整数
- 可指定底层类型
  :::

## 🧱 结构体与联合体

### 结构体 (struct)

```cpp
// 定义结构体
struct Point {
    double x;
    double y;
};

// 带成员函数的结构体
struct Rectangle {
    double width;
    double height;

    double area() const { return width * height; }
    double perimeter() const { return 2 * (width + height); }
};

int main() {
    // 初始化方式
    Point p1 = {1.0, 2.0};        // 聚合初始化
    Point p2{3.0, 4.0};           // 统一初始化
    Point p3;                      // 默认初始化（值未定义）
    Point p4 = {};                // 零初始化

    // C++20 指定初始化
    Point p5 = {.x = 5.0, .y = 6.0};

    Rectangle rect{10, 20};
    std::cout << rect.area() << std::endl;  // 200

    return 0;
}
```

### 联合体 (union)

```cpp
// 联合体：所有成员共享同一块内存
union Data {
    int i;
    float f;
    char c;
};

int main() {
    Data d;
    d.i = 42;
    std::cout << d.i << std::endl;  // 42

    d.f = 3.14f;  // 覆盖之前的值
    // d.i 现在是未定义的

    std::cout << sizeof(Data) << std::endl;  // 通常是 4
    return 0;
}
```

### std::variant (C++17，类型安全的联合体)

```cpp
#include <variant>

std::variant<int, double, std::string> value;

value = 42;
std::cout << std::get<int>(value) << std::endl;

value = 3.14;
std::cout << std::get<double>(value) << std::endl;

value = "hello";
std::cout << std::get<std::string>(value) << std::endl;

// 使用 std::visit
std::visit([](auto&& arg) {
    std::cout << arg << std::endl;
}, value);
```

## ⚙️ constexpr 深入

### constexpr 变量

```cpp
constexpr int SIZE = 10;              // 编译期常量
constexpr double PI = 3.14159;
constexpr int arr[] = {1, 2, 3};      // 编译期数组

int runtime_value = 5;
// constexpr int x = runtime_value;  // 错误：必须是编译期已知
```

### constexpr 函数

```cpp
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

constexpr int result = factorial(5);  // 编译期计算 = 120
static_assert(result == 120, "factorial error");

// C++14 允许更复杂的 constexpr 函数
constexpr int fibonacci(int n) {
    if (n <= 1) return n;
    int a = 0, b = 1;
    for (int i = 2; i <= n; ++i) {
        int temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}
```

### consteval (C++20) - 必须编译期执行

```cpp
consteval int compiletime_only(int n) {
    return n * 2;
}

constexpr int a = compiletime_only(10);  // OK
// int b = compiletime_only(runtime_value);  // 错误：必须编译期
```

### constinit (C++20) - 静态初始化

```cpp
constinit int global = 42;  // 保证静态初始化
```

## ➕ 运算符

### 算术运算符

```cpp
int main() {
    int a = 10, b = 3;

    int sum = a + b;    // 13
    int diff = a - b;   // 7
    int prod = a * b;   // 30
    int quot = a / b;   // 3 (整数除法)
    int rem = a % b;    // 1 (取模)

    // 浮点除法
    double result = static_cast<double>(a) / b;  // 3.333...

    // 自增/自减
    int x = 5;
    int y = x++;  // y = 5, x = 6
    int z = ++x;  // z = 7, x = 7

    return 0;
}
```

### 比较运算符

```cpp
int main() {
    int a = 10, b = 20;

    bool eq = (a == b);   // false
    bool neq = (a != b);  // true
    bool lt = (a < b);    // true
    bool le = (a <= b);   // true
    bool gt = (a > b);    // false
    bool ge = (a >= b);   // false

    // C++20 三路比较（太空船运算符）
    // auto cmp = a <=> b;  // std::strong_ordering::less

    return 0;
}
```

### 逻辑运算符

```cpp
int main() {
    bool a = true, b = false;

    bool andResult = a && b;  // false
    bool orResult = a || b;   // true
    bool notResult = !a;      // false

    // 短路求值
    int x = 0;
    if (x != 0 && 10 / x > 1) {  // 第二个条件不会执行
        // ...
    }

    return 0;
}
```

### 位运算符

```cpp
int main() {
    unsigned int a = 0b1010;  // 10
    unsigned int b = 0b1100;  // 12

    unsigned int andR = a & b;   // 0b1000 = 8
    unsigned int orR = a | b;    // 0b1110 = 14
    unsigned int xorR = a ^ b;   // 0b0110 = 6
    unsigned int notR = ~a;      // 按位取反
    unsigned int leftR = a << 2; // 0b101000 = 40
    unsigned int rightR = a >> 1;// 0b0101 = 5

    return 0;
}
```

### 赋值运算符

```cpp
int main() {
    int a = 10;

    a += 5;   // a = 15
    a -= 3;   // a = 12
    a *= 2;   // a = 24
    a /= 4;   // a = 6
    a %= 4;   // a = 2

    a &= 1;   // 位与赋值
    a |= 2;   // 位或赋值
    a ^= 3;   // 异或赋值
    a <<= 1;  // 左移赋值
    a >>= 1;  // 右移赋值

    return 0;
}
```

## 🔀 控制流

### if-else 语句

```cpp
#include <iostream>

int main() {
    int score = 85;

    if (score >= 90) {
        std::cout << "优秀" << std::endl;
    } else if (score >= 80) {
        std::cout << "良好" << std::endl;
    } else if (score >= 60) {
        std::cout << "及格" << std::endl;
    } else {
        std::cout << "不及格" << std::endl;
    }

    // C++17: if 带初始化
    if (int x = getValue(); x > 0) {
        std::cout << "Positive: " << x << std::endl;
    }

    // 条件运算符
    std::string result = (score >= 60) ? "通过" : "未通过";

    return 0;
}
```

### switch 语句

```cpp
#include <iostream>

int main() {
    int day = 3;

    switch (day) {
        case 1:
            std::cout << "星期一" << std::endl;
            break;
        case 2:
            std::cout << "星期二" << std::endl;
            break;
        case 3:
            std::cout << "星期三" << std::endl;
            break;
        case 6:
        case 7:
            std::cout << "周末" << std::endl;
            break;
        default:
            std::cout << "其他" << std::endl;
    }

    // C++17: switch 带初始化
    switch (int n = getValue(); n) {
        case 0: break;
        case 1: break;
        default: break;
    }

    return 0;
}
```

### for 循环

```cpp
#include <iostream>
#include <vector>

int main() {
    // 传统 for 循环
    for (int i = 0; i < 5; i++) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    // 范围 for 循环 (C++11)
    std::vector<int> nums = {1, 2, 3, 4, 5};
    for (int n : nums) {
        std::cout << n << " ";
    }
    std::cout << std::endl;

    // 使用引用修改元素
    for (int& n : nums) {
        n *= 2;
    }

    // 使用 auto
    for (const auto& n : nums) {
        std::cout << n << " ";
    }

    return 0;
}
```

### while 和 do-while

```cpp
#include <iostream>

int main() {
    // while 循环
    int i = 0;
    while (i < 5) {
        std::cout << i << " ";
        i++;
    }
    std::cout << std::endl;

    // do-while 循环（至少执行一次）
    int j = 0;
    do {
        std::cout << j << " ";
        j++;
    } while (j < 5);

    return 0;
}
```

### break 和 continue

```cpp
#include <iostream>

int main() {
    // break 跳出循环
    for (int i = 0; i < 10; i++) {
        if (i == 5) break;
        std::cout << i << " ";  // 0 1 2 3 4
    }
    std::cout << std::endl;

    // continue 跳过当前迭代
    for (int i = 0; i < 10; i++) {
        if (i % 2 == 0) continue;
        std::cout << i << " ";  // 1 3 5 7 9
    }

    return 0;
}
```

## 📥 输入输出

### 标准输出

```cpp
#include <iostream>
#include <iomanip>

int main() {
    // 基本输出
    std::cout << "Hello" << std::endl;
    std::cout << "Value: " << 42 << std::endl;

    // 格式化输出
    double pi = 3.14159265358979;
    std::cout << std::fixed << std::setprecision(2) << pi << std::endl;  // 3.14

    // 宽度和对齐
    std::cout << std::setw(10) << std::right << 42 << std::endl;
    std::cout << std::setw(10) << std::left << "hi" << std::endl;

    // 进制输出
    int num = 255;
    std::cout << std::dec << num << std::endl;  // 255
    std::cout << std::hex << num << std::endl;  // ff
    std::cout << std::oct << num << std::endl;  // 377

    return 0;
}
```

### 标准输入

```cpp
#include <iostream>
#include <string>

int main() {
    // 输入整数
    int age;
    std::cout << "请输入年龄: ";
    std::cin >> age;

    // 输入字符串（单词）
    std::string name;
    std::cout << "请输入姓名: ";
    std::cin >> name;

    // 输入整行
    std::cin.ignore();  // 清除上次输入的换行符
    std::string line;
    std::cout << "请输入一行文本: ";
    std::getline(std::cin, line);

    // 检查输入是否有效
    if (std::cin.fail()) {
        std::cin.clear();  // 清除错误状态
        std::cin.ignore(10000, '\n');  // 忽略错误输入
    }

    return 0;
}
```

## 🎯 类型转换

```cpp
#include <iostream>

int main() {
    // C 风格转换（不推荐）
    double d = 3.14;
    int i = (int)d;

    // C++ 风格转换
    // static_cast: 编译期类型转换
    int j = static_cast<int>(d);

    // dynamic_cast: 运行时多态类型转换
    // const_cast: 移除 const 属性
    // reinterpret_cast: 底层位模式转换（危险）

    // 安全的数值转换
    long long big = 1000000000000LL;
    // int small = static_cast<int>(big);  // 可能溢出！

    // 字符串转换
    std::string str = "42";
    int num = std::stoi(str);
    double dbl = std::stod("3.14");
    std::string numStr = std::to_string(num);

    return 0;
}
```

## ⚡ 最佳实践

1. **优先使用 `{}` 统一初始化** - 更安全，防止窄化转换
2. **使用 `auto`** - 简化代码，特别是复杂类型
3. **使用 `constexpr`** - 编译期常量优于运行时常量
4. **避免 `using namespace std`** - 防止命名冲突
5. **使用 `static_cast`** - 明确的类型转换意图
6. **范围 for 循环** - 更简洁安全

恭喜你掌握了 C++ 基础语法！🎉
