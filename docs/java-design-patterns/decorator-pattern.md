---
sidebar_position: 6
---

# 装饰器模式 (Decorator Pattern)

## 模式定义

**装饰器模式**是一种结构型设计模式，它允许向一个对象动态添加新的功能，同时保持其结构不变。装饰器模式是继承的一个替代方案。

## 问题分析

如果需要为对象添加新功能，传统方法是使用继承：

```java
// 不好的做法 - 会导致类爆炸
public class SimpleCoffee { }
public class SimpleCoffeeWithMilk extends SimpleCoffee { }
public class SimpleCoffeeWithMilkAndSugar extends SimpleCoffeeWithMilk { }
public class SimpleCoffeeWithMilkAndSugarAndCream extends SimpleCoffeeWithMilkAndSugar { }
```

**问题**：
- 类的数量呈指数增长
- 难以维护和扩展
- 违反单一职责原则

## 解决方案

使用组合而不是继承：

```
┌─────────────┐
│  Component  │
│  (接口)     │
│+ operation()│
└──────┬──────┘
       △
       │
    ┌──┴──────────────────┐
    │                     │
┌────────┐          ┌──────────────┐
│Concrete│          │   Decorator  │
│Component           │ - component  │
└────────┘          │ + operation()│
                    └──────┬───────┘
                           △
                           │
                    ┌──────┴──────┐
                    │             │
              ┌─────────────┐  ┌──────────┐
              │Decorator A  │  │Decorator │
              │             │  │   B      │
              └─────────────┘  └──────────┘
```

## 代码实现

### 1. 定义组件接口

```java
public interface Coffee {
    double getCost();
    String getDescription();
}
```

### 2. 具体组件

```java
public class SimpleCoffee implements Coffee {
    @Override
    public double getCost() {
        return 10.0;
    }
    
    @Override
    public String getDescription() {
        return "简单咖啡";
    }
}
```

### 3. 定义装饰器抽象类

```java
public abstract class CoffeeDecorator implements Coffee {
    protected Coffee coffee;
    
    public CoffeeDecorator(Coffee coffee) {
        this.coffee = coffee;
    }
    
    @Override
    public double getCost() {
        return coffee.getCost();
    }
    
    @Override
    public String getDescription() {
        return coffee.getDescription();
    }
}
```

### 4. 具体装饰器

```java
public class MilkDecorator extends CoffeeDecorator {
    public MilkDecorator(Coffee coffee) {
        super(coffee);
    }
    
    @Override
    public double getCost() {
        return super.getCost() + 2.0;
    }
    
    @Override
    public String getDescription() {
        return super.getDescription() + ", 加牛奶";
    }
}

public class SugarDecorator extends CoffeeDecorator {
    public SugarDecorator(Coffee coffee) {
        super(coffee);
    }
    
    @Override
    public double getCost() {
        return super.getCost() + 0.5;
    }
    
    @Override
    public String getDescription() {
        return super.getDescription() + ", 加糖";
    }
}

public class VanillaDecorator extends CoffeeDecorator {
    public VanillaDecorator(Coffee coffee) {
        super(coffee);
    }
    
    @Override
    public double getCost() {
        return super.getCost() + 3.0;
    }
    
    @Override
    public String getDescription() {
        return super.getDescription() + ", 加香草";
    }
}
```

### 5. 客户端使用

```java
public class Demo {
    public static void main(String[] args) {
        // 简单咖啡
        Coffee coffee = new SimpleCoffee();
        System.out.println(coffee.getDescription() + ": ¥" + coffee.getCost());
        
        // 咖啡 + 牛奶
        coffee = new MilkDecorator(coffee);
        System.out.println(coffee.getDescription() + ": ¥" + coffee.getCost());
        
        // 咖啡 + 牛奶 + 糖
        coffee = new SugarDecorator(coffee);
        System.out.println(coffee.getDescription() + ": ¥" + coffee.getCost());
        
        // 咖啡 + 牛奶 + 糖 + 香草
        coffee = new VanillaDecorator(coffee);
        System.out.println(coffee.getDescription() + ": ¥" + coffee.getCost());
    }
}
```

**输出**：
```
简单咖啡: ¥10.0
简单咖啡, 加牛奶: ¥12.0
简单咖啡, 加牛奶, 加糖: ¥12.5
简单咖啡, 加牛奶, 加糖, 加香草: ¥15.5
```

## 实际应用示例

### IO流装饰器

Java的IO框架就是装饰器模式的经典应用：

```java
// InputStream是基础组件
InputStream input = new FileInputStream("file.txt");

// 使用装饰器添加功能
input = new BufferedInputStream(input);    // 缓冲
input = new DataInputStream(input);        // 数据类型支持

// 继续装饰
InputStream gzipInput = new GZIPInputStream(input);  // 压缩
```

### UI组件装饰器

```java
public interface UIComponent {
    void render();
}

public class SimpleButton implements UIComponent {
    @Override
    public void render() {
        System.out.println("绘制按钮");
    }
}

public abstract class UIDecorator implements UIComponent {
    protected UIComponent component;
    
    public UIDecorator(UIComponent component) {
        this.component = component;
    }
}

public class BorderDecorator extends UIDecorator {
    public BorderDecorator(UIComponent component) {
        super(component);
    }
    
    @Override
    public void render() {
        System.out.println("添加边框");
        component.render();
    }
}

public class ScrollDecorator extends UIDecorator {
    public ScrollDecorator(UIComponent component) {
        super(component);
    }
    
    @Override
    public void render() {
        System.out.println("添加滚动条");
        component.render();
    }
}
```

## 装饰器模式 vs 继承

| 特性 | 装饰器 | 继承 |
|------|-------|------|
| 灵活性 | 高 | 低 |
| 类数量 | 多 | 少 |
| 运行时变化 | 可以 | 不能 |
| 代码复杂度 | 中 | 低 |

## Java内置装饰器

```java
// Collections.synchronizedList是装饰器
List<String> list = Collections.synchronizedList(new ArrayList<>());

// Collections.unmodifiableList是装饰器
List<String> immutable = Collections.unmodifiableList(list);

// String.toLowerCase() 等方法也遵循装饰器思想
```

## 优缺点

### 优点
- ✅ 比继承更灵活
- ✅ 可以动态添加功能
- ✅ 符合开闭原则
- ✅ 避免类的爆炸

### 缺点
- ❌ 代码复杂度增加
- ❌ 类的数量增多
- ❌ 装饰器顺序很重要
- ❌ 调试困难

## 适用场景

- ✓ 需要动态添加功能
- ✓ 功能组合多样
- ✓ 继承不适用
- ✓ 需要保持原有接口
- ✓ IO流、UI组件等

## 与其他模式的关系

- **代理模式** - 控制访问，装饰器添加功能
- **适配器模式** - 改变接口，装饰器增强功能
- **策略模式** - 改变算法，装饰器改变对象
