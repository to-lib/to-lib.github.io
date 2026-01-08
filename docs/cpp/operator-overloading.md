---
sidebar_position: 8.5
title: 运算符重载
---

# C++ 运算符重载

运算符重载允许自定义类型使用内置运算符，使代码更直观。

## 🎯 基本语法

```cpp
class Complex {
private:
    double real, imag;

public:
    Complex(double r = 0, double i = 0) : real(r), imag(i) {}

    // 成员函数重载 +
    Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }

    // 成员函数重载 +=
    Complex& operator+=(const Complex& other) {
        real += other.real;
        imag += other.imag;
        return *this;
    }

    // 友元函数重载 <<
    friend std::ostream& operator<<(std::ostream& os, const Complex& c) {
        return os << c.real << " + " << c.imag << "i";
    }
};

int main() {
    Complex a(1, 2), b(3, 4);
    Complex c = a + b;
    std::cout << c << std::endl;  // 4 + 6i
    return 0;
}
```

## 📦 常见运算符重载

### 算术运算符

```cpp
class Vector2D {
public:
    double x, y;

    Vector2D operator+(const Vector2D& v) const {
        return {x + v.x, y + v.y};
    }

    Vector2D operator-(const Vector2D& v) const {
        return {x - v.x, y - v.y};
    }

    Vector2D operator*(double scalar) const {
        return {x * scalar, y * scalar};
    }

    // 一元负号
    Vector2D operator-() const {
        return {-x, -y};
    }
};

// 非成员函数：scalar * vector
Vector2D operator*(double scalar, const Vector2D& v) {
    return v * scalar;
}
```

### 比较运算符

```cpp
class Point {
public:
    int x, y;

    bool operator==(const Point& p) const {
        return x == p.x && y == p.y;
    }

    bool operator!=(const Point& p) const {
        return !(*this == p);
    }

    bool operator<(const Point& p) const {
        return (x < p.x) || (x == p.x && y < p.y);
    }

    // C++20: 太空船运算符
    auto operator<=>(const Point&) const = default;
};
```

### 下标运算符

```cpp
class Array {
    int* data;
    size_t size;

public:
    int& operator[](size_t i) {
        return data[i];
    }

    const int& operator[](size_t i) const {
        return data[i];
    }
};
```

### 函数调用运算符

```cpp
class Adder {
    int value;

public:
    Adder(int v) : value(v) {}

    int operator()(int x) const {
        return x + value;
    }
};

Adder add5(5);
std::cout << add5(10) << std::endl;  // 15
```

### 自增/自减

```cpp
class Counter {
    int count;

public:
    // 前置 ++
    Counter& operator++() {
        ++count;
        return *this;
    }

    // 后置 ++ (int 是占位符)
    Counter operator++(int) {
        Counter temp = *this;
        ++count;
        return temp;
    }
};
```

### 类型转换

```cpp
class Fraction {
    int num, den;

public:
    // 转换为 double
    explicit operator double() const {
        return static_cast<double>(num) / den;
    }

    // 转换为 bool
    explicit operator bool() const {
        return num != 0;
    }
};
```

## ⚠️ 不能重载的运算符

- `::` 作用域解析
- `.` 成员访问
- `.*` 成员指针访问
- `?:` 条件运算符
- `sizeof`
- `typeid`

## ⚡ 最佳实践

1. **保持语义一致** - 运算符行为符合直觉
2. **返回引用** - 赋值运算符返回 `*this` 引用
3. **使用 const** - 不修改对象的运算符
4. **对称运算符用非成员** - 如 `a + b` 和 `b + a`
5. **使用 explicit** - 防止隐式类型转换
