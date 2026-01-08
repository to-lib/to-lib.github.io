---
sidebar_position: 12
title: 抽象类和接口
---

# C++ 抽象类和接口

抽象类定义了派生类必须实现的接口，是多态的基础。

## 🔷 抽象类

包含至少一个纯虚函数的类：

```cpp
class Shape {
public:
    // 纯虚函数
    virtual double area() const = 0;
    virtual double perimeter() const = 0;
    virtual void draw() const = 0;

    // 普通虚函数
    virtual std::string getName() const {
        return "Shape";
    }

    virtual ~Shape() = default;
};

class Circle : public Shape {
private:
    double radius;

public:
    Circle(double r) : radius(r) {}

    double area() const override {
        return 3.14159 * radius * radius;
    }

    double perimeter() const override {
        return 2 * 3.14159 * radius;
    }

    void draw() const override {
        std::cout << "○" << std::endl;
    }
};
```

## 🎭 纯接口

只包含纯虚函数的抽象类：

```cpp
// 接口
class Drawable {
public:
    virtual void draw() const = 0;
    virtual ~Drawable() = default;
};

class Serializable {
public:
    virtual std::string serialize() const = 0;
    virtual void deserialize(const std::string& data) = 0;
    virtual ~Serializable() = default;
};

// 实现多个接口
class Document : public Drawable, public Serializable {
public:
    void draw() const override {
        std::cout << "绘制文档" << std::endl;
    }

    std::string serialize() const override {
        return "document_data";
    }

    void deserialize(const std::string& data) override {
        // 反序列化
    }
};
```

## 📋 接口设计模式

```cpp
// 策略模式
class SortStrategy {
public:
    virtual void sort(std::vector<int>& data) = 0;
    virtual ~SortStrategy() = default;
};

class QuickSort : public SortStrategy {
public:
    void sort(std::vector<int>& data) override {
        // 快速排序实现
    }
};

class MergeSort : public SortStrategy {
public:
    void sort(std::vector<int>& data) override {
        // 归并排序实现
    }
};

class Sorter {
private:
    std::unique_ptr<SortStrategy> strategy;

public:
    void setStrategy(std::unique_ptr<SortStrategy> s) {
        strategy = std::move(s);
    }

    void performSort(std::vector<int>& data) {
        if (strategy) strategy->sort(data);
    }
};
```

## ⚡ 最佳实践

1. **接口只定义行为** - 不包含数据成员
2. **使用虚析构函数** - 确保正确清理
3. **小而专注的接口** - 接口隔离原则
4. **使用 override** - 明确重写意图
5. **考虑 Concepts (C++20)** - 更强的接口约束
