---
sidebar_position: 25
title: 最佳实践
---

# C++ 最佳实践

遵循这些最佳实践，编写安全、高效、可维护的 C++ 代码。

## 🛡️ 内存安全

```cpp
// ✅ 使用智能指针
auto ptr = std::make_unique<Widget>();
auto shared = std::make_shared<Resource>();

// ✅ RAII 管理资源
std::lock_guard<std::mutex> lock(mtx);
std::ifstream file("data.txt");  // 自动关闭

// ❌ 避免裸 new/delete
// int* p = new int(42);
// delete p;
```

## 📦 现代 C++ 特性

```cpp
// ✅ 使用 auto
auto iter = container.begin();
auto result = calculate();

// ✅ 使用范围 for
for (const auto& item : container) { }

// ✅ 使用统一初始化
std::vector<int> v{1, 2, 3};
Widget w{};

// ✅ 使用 nullptr
if (ptr == nullptr) { }
```

## 🔧 类设计

```cpp
class Widget {
public:
    // ✅ 构造函数使用初始化列表
    Widget(int x, std::string s) : value(x), name(std::move(s)) {}

    // ✅ 标记 const 成员函数
    int getValue() const { return value; }

    // ✅ 移动操作标记 noexcept
    Widget(Widget&&) noexcept = default;
    Widget& operator=(Widget&&) noexcept = default;

    // ✅ 使用 override
    void foo() override;

private:
    int value;
    std::string name;
};
```

## ⚡ 性能优化

```cpp
// ✅ 传递大对象使用 const 引用
void process(const std::vector<int>& data);

// ✅ 使用 move 转移所有权
void consume(std::string&& s);

// ✅ 预分配容器容量
std::vector<int> v;
v.reserve(1000);

// ✅ 使用 emplace
v.emplace_back(1, 2, 3);
```

## 📋 代码风格

```cpp
// ✅ 使用有意义的命名
int userCount;
void calculateTotalPrice();

// ✅ 使用 constexpr 编译期常量
constexpr int MAX_SIZE = 100;

// ✅ 开启编译警告
// g++ -Wall -Wextra -Wpedantic -Werror
```

## 🔍 调试建议

- 使用 AddressSanitizer 检测内存问题
- 使用 Valgrind 检测内存泄漏
- 使用静态分析工具 (clang-tidy)
- 编写单元测试

## ⚠️ 常见陷阱

1. **避免未定义行为** - 空指针解引用、越界访问
2. **避免数据竞争** - 使用锁保护共享数据
3. **避免悬空引用** - 注意对象生命周期
4. **避免隐式转换** - 使用 explicit
