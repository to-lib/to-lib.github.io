---
sidebar_position: 3
---

# 工厂方法模式 (Factory Method Pattern)

## 模式定义

**工厂方法模式**是一种创建型设计模式，它定义了一个创建对象的接口，让子类来决定实例化哪个具体类。

## 问题分析

当系统需要创建多种类型的对象时，直接使用 `new` 关键字会导致：

- 客户端与具体类紧耦合
- 扩展新产品类型需要修改客户端代码
- 创建逻辑分散在各处

## 解决方案

工厂方法模式通过抽象工厂方法来隔离对象创建：

```
┌─────────────────────────────────────────────┐
│          Creator（创建者）                   │
│  ┌─────────────────────────────────────┐   │
│  │ + factoryMethod(): Product          │   │
│  │ + businessMethod()                  │   │
│  └─────────────────────────────────────┘   │
│                    △                        │
│                    │                        │
│    ┌───────────────┴───────────────┐      │
│    │                               │      │
│ ConcreteCreator1          ConcreteCreator2│
│ + factoryMethod()         + factoryMethod()│
└─────────────────────────────────────────────┘
            ↓ creates                ↓ creates
        ┌────────┐             ┌────────┐
        │Product │             │Product │
        │  (接口) │             │ (接口) │
        └────────┘             └────────┘
            △                       △
            │                       │
    ConcreteProduct1      ConcreteProduct2
```

## 代码实现

### 1. 定义产品接口

```java
public interface Button {
    void render();
    void onClick();
}
```

### 2. 具体产品实现

```java
public class WindowsButton implements Button {
    @Override
    public void render() {
        System.out.println("绘制Windows风格按钮");
    }
    
    @Override
    public void onClick() {
        System.out.println("Windows按钮被点击");
    }
}

public class MacButton implements Button {
    @Override
    public void render() {
        System.out.println("绘制Mac风格按钮");
    }
    
    @Override
    public void onClick() {
        System.out.println("Mac按钮被点击");
    }
}
```

### 3. 定义创建者接口

```java
public abstract class Dialog {
    public void render() {
        Button button = createButton();
        button.render();
    }
    
    // 工厂方法
    public abstract Button createButton();
}
```

### 4. 具体创建者实现

```java
public class WindowsDialog extends Dialog {
    @Override
    public Button createButton() {
        return new WindowsButton();
    }
}

public class MacDialog extends Dialog {
    @Override
    public Button createButton() {
        return new MacButton();
    }
}
```

### 5. 客户端使用

```java
public class Application {
    private Dialog dialog;
    
    public Application(String os) {
        if ("Windows".equals(os)) {
            dialog = new WindowsDialog();
        } else if ("Mac".equals(os)) {
            dialog = new MacDialog();
        }
    }
    
    public void run() {
        dialog.render();
    }
    
    public static void main(String[] args) {
        Application app = new Application("Windows");
        app.run();
    }
}
```

## 实际应用示例

### 数据库连接工厂

```java
// 产品接口
public interface DatabaseConnection {
    void connect();
    void query(String sql);
}

// 具体产品
public class MySQLConnection implements DatabaseConnection {
    @Override
    public void connect() {
        System.out.println("连接到MySQL数据库");
    }
    
    @Override
    public void query(String sql) {
        System.out.println("执行MySQL查询: " + sql);
    }
}

public class PostgreSQLConnection implements DatabaseConnection {
    @Override
    public void connect() {
        System.out.println("连接到PostgreSQL数据库");
    }
    
    @Override
    public void query(String sql) {
        System.out.println("执行PostgreSQL查询: " + sql);
    }
}

// 工厂方法
public abstract class DatabaseFactory {
    public abstract DatabaseConnection createConnection();
    
    public void executeQuery(String sql) {
        DatabaseConnection conn = createConnection();
        conn.connect();
        conn.query(sql);
    }
}

// 具体工厂
public class MySQLFactory extends DatabaseFactory {
    @Override
    public DatabaseConnection createConnection() {
        return new MySQLConnection();
    }
}

public class PostgreSQLFactory extends DatabaseFactory {
    @Override
    public DatabaseConnection createConnection() {
        return new PostgreSQLConnection();
    }
}
```

## 工厂方法 vs 简单工厂

### 简单工厂（不是标准设计模式）

```java
public class SimpleFactory {
    public static Button createButton(String type) {
        if ("Windows".equals(type)) {
            return new WindowsButton();
        } else if ("Mac".equals(type)) {
            return new MacButton();
        }
        return null;
    }
}
```

**对比**：
- 工厂方法：每个创建者对应一个具体产品
- 简单工厂：一个工厂方法对应多个产品

## 优缺点

### 优点
- ✅ 解耦客户端和具体类
- ✅ 符合开闭原则，易于扩展
- ✅ 符合单一职责原则
- ✅ 支持多种产品的创建

### 缺点
- ❌ 代码复杂度增加
- ❌ 类的数量增多
- ❌ 产品数量多时维护困难

## 适用场景

- ✓ 系统不依赖产品的具体实现
- ✓ 需要支持多种产品类型
- ✓ 产品创建逻辑复杂
- ✓ 需要延迟产品的创建
- ✓ 提供产品库供客户端使用

## 与其他模式的关系

- **抽象工厂模式** - 处理产品族
- **模板方法模式** - 定义算法骨架
- **策略模式** - 运行时选择算法
