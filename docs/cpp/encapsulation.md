---
sidebar_position: 11
title: 封装
---

# C++ 封装

封装是面向对象的核心原则，将数据和操作数据的方法绑定在一起，隐藏内部实现细节。

## 🔐 访问控制

```cpp
class BankAccount {
private:        // 私有：只有类内部可访问
    double balance;
    std::string accountNumber;

protected:      // 保护：类内部和派生类可访问
    std::string ownerName;

public:         // 公有：任何地方都可访问
    BankAccount(const std::string& owner, double initial)
        : ownerName(owner), balance(initial) {}

    // 公有接口
    double getBalance() const { return balance; }

    bool deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            return true;
        }
        return false;
    }

    bool withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            return true;
        }
        return false;
    }
};
```

## 🎯 Getter 和 Setter

```cpp
class Person {
private:
    std::string name;
    int age;

public:
    // Getter
    const std::string& getName() const { return name; }
    int getAge() const { return age; }

    // Setter（带验证）
    void setName(const std::string& n) {
        if (!n.empty()) name = n;
    }

    void setAge(int a) {
        if (a >= 0 && a <= 150) age = a;
    }
};
```

## 👥 友元

```cpp
class Box {
private:
    double width;

    // 友元函数可访问私有成员
    friend void printWidth(const Box& b);

    // 友元类
    friend class BoxFactory;
};

void printWidth(const Box& b) {
    std::cout << b.width << std::endl;  // 可访问
}

class BoxFactory {
public:
    Box createBox(double w) {
        Box b;
        b.width = w;  // 可访问私有成员
        return b;
    }
};
```

## 📦 Pimpl 模式

隐藏实现细节，减少编译依赖：

```cpp
// widget.h
class Widget {
public:
    Widget();
    ~Widget();
    void doSomething();

private:
    class Impl;  // 前向声明
    std::unique_ptr<Impl> pImpl;
};

// widget.cpp
class Widget::Impl {
public:
    void doSomethingImpl() { /* 实现 */ }
};

Widget::Widget() : pImpl(std::make_unique<Impl>()) {}
Widget::~Widget() = default;
void Widget::doSomething() { pImpl->doSomethingImpl(); }
```

## ⚡ 最佳实践

1. **数据成员私有化** - 通过公有方法访问
2. **最小化公有接口** - 只暴露必要的方法
3. **使用 const** - 不修改状态的方法标记为 const
4. **验证输入** - 在 setter 中检查有效性
5. **考虑 Pimpl** - 减少编译依赖
