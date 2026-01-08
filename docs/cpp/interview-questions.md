---
sidebar_position: 30
title: 面试题精选
---

# C++ 面试题精选

常见 C++ 面试问题和答案。

## 🎯 基础概念

### 指针和引用的区别？

| 指针           | 引用         |
| -------------- | ------------ |
| 可为空         | 必须初始化   |
| 可重新赋值     | 不能重新绑定 |
| 需要解引用 `*` | 直接使用     |
| 有内存空间     | 是别名       |

### new/delete 和 malloc/free 的区别？

- `new/delete` 调用构造/析构函数
- `new` 可以重载
- `new` 返回类型安全的指针
- `malloc` 返回 `void*`

### 什么是虚函数？

```cpp
class Base {
public:
    virtual void foo() { }  // 虚函数，支持多态
};
```

运行时根据对象实际类型调用对应函数。

## 📦 内存管理

### 智能指针类型？

- `unique_ptr`: 独占所有权
- `shared_ptr`: 共享所有权，引用计数
- `weak_ptr`: 弱引用，不增加计数

### 什么是 RAII？

Resource Acquisition Is Initialization

- 构造时获取资源
- 析构时释放资源
- 自动管理生命周期

## 🔧 面向对象

### 什么是三/五法则？

如果定义了以下任一项，应考虑定义全部：

- 析构函数
- 拷贝构造函数
- 拷贝赋值运算符
- (C++11) 移动构造函数
- (C++11) 移动赋值运算符

### 虚析构函数的作用？

```cpp
Base* p = new Derived();
delete p;  // 需要虚析构才能正确释放 Derived
```

## ⚡ 现代 C++

### 左值和右值？

- 左值：有名字、可取地址
- 右值：临时对象、字面量

### 完美转发？

```cpp
template<typename T>
void wrapper(T&& arg) {
    func(std::forward<T>(arg));
}
```

保持参数的值类别。

### auto 和 decltype？

```cpp
auto x = 10;       // 推导类型
decltype(x) y;     // 获取 x 的类型
```

## 📋 常见陷阱

### 避免悬空指针

```cpp
int* p;
{
    int x = 10;
    p = &x;
}
// p 现在是悬空指针
```

### 避免对象切片

```cpp
Derived d;
Base b = d;  // 切片！丢失派生类数据
Base& ref = d;  // OK
```

## 🔗 快速参考

| 概念     | 关键点                  |
| -------- | ----------------------- |
| 多态     | 虚函数 + 基类指针/引用  |
| RAII     | 资源管理 = 对象生命周期 |
| 移动语义 | 右值引用 + std::move    |
| 智能指针 | 自动内存管理            |
