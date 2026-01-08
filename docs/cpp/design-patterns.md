---
sidebar_position: 24
title: 设计模式
---

# C++ 设计模式

设计模式是解决常见设计问题的可复用方案。

## 🏭 创建型模式

### 单例模式

```cpp
class Singleton {
private:
    Singleton() = default;

public:
    static Singleton& getInstance() {
        static Singleton instance;
        return instance;
    }

    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
};
```

### 工厂模式

```cpp
class Product {
public:
    virtual void use() = 0;
    virtual ~Product() = default;
};

class ConcreteProductA : public Product {
public:
    void use() override { std::cout << "Product A" << std::endl; }
};

class Factory {
public:
    static std::unique_ptr<Product> create(const std::string& type) {
        if (type == "A") return std::make_unique<ConcreteProductA>();
        return nullptr;
    }
};
```

## 🔧 结构型模式

### 适配器模式

```cpp
class Target {
public:
    virtual void request() = 0;
};

class Adaptee {
public:
    void specificRequest() { std::cout << "Specific" << std::endl; }
};

class Adapter : public Target {
private:
    Adaptee adaptee;

public:
    void request() override {
        adaptee.specificRequest();
    }
};
```

## 🎭 行为型模式

### 观察者模式

```cpp
class Observer {
public:
    virtual void update(int value) = 0;
};

class Subject {
    std::vector<Observer*> observers;
    int state;

public:
    void attach(Observer* o) { observers.push_back(o); }

    void setState(int s) {
        state = s;
        for (auto o : observers) o->update(state);
    }
};
```

### 策略模式

```cpp
class Strategy {
public:
    virtual int execute(int a, int b) = 0;
};

class AddStrategy : public Strategy {
public:
    int execute(int a, int b) override { return a + b; }
};

class Context {
    std::unique_ptr<Strategy> strategy;

public:
    void setStrategy(std::unique_ptr<Strategy> s) {
        strategy = std::move(s);
    }

    int doWork(int a, int b) {
        return strategy->execute(a, b);
    }
};
```

## ⚡ 最佳实践

1. **优先组合而非继承**
2. **面向接口编程**
3. **使用智能指针管理对象**
4. **不要过度设计**
