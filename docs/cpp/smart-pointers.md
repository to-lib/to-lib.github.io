---
sidebar_position: 15
title: 智能指针
---

# C++ 智能指针

智能指针自动管理动态内存，是现代 C++ 的核心特性。

## 🔒 unique_ptr

独占所有权，不能拷贝：

```cpp
#include <memory>

// 创建
auto ptr = std::make_unique<int>(42);
std::unique_ptr<int[]> arr = std::make_unique<int[]>(10);

// 使用
std::cout << *ptr << std::endl;
arr[0] = 100;

// 转移所有权
auto ptr2 = std::move(ptr);  // ptr 现在为空

// 检查
if (ptr2) {
    std::cout << *ptr2 << std::endl;
}

// 释放所有权
int* raw = ptr2.release();
delete raw;

// 重置
ptr2.reset(new int(100));
ptr2.reset();  // 释放并置空

// 自定义删除器
auto deleter = [](FILE* f) { fclose(f); };
std::unique_ptr<FILE, decltype(deleter)> file(fopen("test.txt", "r"), deleter);
```

## 🔗 shared_ptr

共享所有权，引用计数：

```cpp
#include <memory>

// 创建
auto ptr1 = std::make_shared<int>(42);
std::cout << "Count: " << ptr1.use_count() << std::endl;  // 1

// 共享所有权
{
    auto ptr2 = ptr1;
    std::cout << "Count: " << ptr1.use_count() << std::endl;  // 2
}
std::cout << "Count: " << ptr1.use_count() << std::endl;  // 1

// 自定义删除器
auto sp = std::shared_ptr<int>(new int(42), [](int* p) {
    std::cout << "Custom delete" << std::endl;
    delete p;
});
```

## 🔍 weak_ptr

弱引用，不增加引用计数，避免循环引用：

```cpp
#include <memory>

struct Node {
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev;  // 使用 weak_ptr 避免循环引用
    int value;
};

int main() {
    auto shared = std::make_shared<int>(42);
    std::weak_ptr<int> weak = shared;

    // 使用前需要锁定
    if (auto locked = weak.lock()) {
        std::cout << *locked << std::endl;
    }

    // 检查是否过期
    std::cout << weak.expired() << std::endl;  // false

    shared.reset();
    std::cout << weak.expired() << std::endl;  // true

    return 0;
}
```

## 🔄 循环引用问题

```cpp
struct A {
    std::shared_ptr<B> b_ptr;
    ~A() { std::cout << "A destroyed" << std::endl; }
};

struct B {
    std::weak_ptr<A> a_ptr;  // 使用 weak_ptr
    ~B() { std::cout << "B destroyed" << std::endl; }
};

int main() {
    auto a = std::make_shared<A>();
    auto b = std::make_shared<B>();
    a->b_ptr = b;
    b->a_ptr = a;
    // 离开作用域时正确释放
    return 0;
}
```

## ⚡ 最佳实践

1. **优先使用 make_unique/make_shared** - 更安全高效
2. **默认使用 unique_ptr** - 只在需要共享时用 shared_ptr
3. **使用 weak_ptr** - 打破循环引用
4. **避免裸指针所有权** - 裸指针仅用于非所有权场景
5. **按值传递智能指针** - 明确所有权转移
