---
sidebar_position: 9
title: 继承
---

# C++ 继承

继承允许创建基于现有类的新类，实现代码复用。

## 🎯 基本继承

```cpp
#include <iostream>
#include <string>

// 基类
class Animal {
protected:
    std::string name;

public:
    Animal(const std::string& n) : name(n) {}

    void eat() const {
        std::cout << name << " 正在吃东西" << std::endl;
    }
};

// 派生类
class Dog : public Animal {
public:
    Dog(const std::string& n) : Animal(n) {}

    void bark() const {
        std::cout << name << " 汪汪叫" << std::endl;
    }
};

int main() {
    Dog dog("旺财");
    dog.eat();   // 继承自 Animal
    dog.bark();  // Dog 自己的方法
    return 0;
}
```

## 🔐 继承方式

```cpp
class Base {
public:    int pub;
protected: int prot;
private:   int priv;
};

// 公有继承：最常用
class DerivedPublic : public Base {
    // pub -> public
    // prot -> protected
    // priv -> 不可访问
};

// 保护继承
class DerivedProtected : protected Base {
    // pub -> protected
    // prot -> protected
    // priv -> 不可访问
};

// 私有继承
class DerivedPrivate : private Base {
    // pub -> private
    // prot -> private
    // priv -> 不可访问
};
```

## 🔄 构造与析构顺序

```cpp
class Base {
public:
    Base() { std::cout << "Base 构造" << std::endl; }
    ~Base() { std::cout << "Base 析构" << std::endl; }
};

class Derived : public Base {
public:
    Derived() { std::cout << "Derived 构造" << std::endl; }
    ~Derived() { std::cout << "Derived 析构" << std::endl; }
};

int main() {
    Derived d;
    // 输出：
    // Base 构造
    // Derived 构造
    // Derived 析构
    // Base 析构
    return 0;
}
```

## 🔀 多重继承

```cpp
class Flyable {
public:
    void fly() { std::cout << "飞行中" << std::endl; }
};

class Swimmable {
public:
    void swim() { std::cout << "游泳中" << std::endl; }
};

// 继承多个类
class Duck : public Flyable, public Swimmable {
public:
    void quack() { std::cout << "嘎嘎" << std::endl; }
};

int main() {
    Duck duck;
    duck.fly();
    duck.swim();
    duck.quack();
    return 0;
}
```

## 💎 菱形继承与虚继承

```cpp
class Animal {
public:
    int age;
};

// 虚继承解决菱形继承问题
class Mammal : virtual public Animal {};
class Bird : virtual public Animal {};

class Bat : public Mammal, public Bird {
    // 只有一份 Animal::age
};
```

## ⚡ 最佳实践

1. **优先使用公有继承** - 表示 "is-a" 关系
2. **使用组合优于继承** - 表示 "has-a" 关系
3. **虚析构函数** - 基类指针删除派生类对象时
4. **谨慎使用多重继承** - 可能导致复杂性
5. **使用 override 关键字** - 明确重写意图
