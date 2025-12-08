---
sidebar_position: 11
---

# 抽象工厂模式 (Abstract Factory Pattern)

## 模式定义

**抽象工厂模式**是一种创建型设计模式，它提供了一个接口来创建**相关或相依对象的族**，而不需要明确指定它们的具体类。

## 问题分析

当系统需要独立于它所使用的具体产品的创建过程，且系统中有多个产品族时，直接创建会导致：

- 代码与具体产品类紧耦合
- 新增产品族时需要修改大量代码
- 难以切换不同的产品族

## 解决方案

```
┌─────────────────────────────────┐
│    AbstractFactory(接口)         │
│  + createProductA()             │
│  + createProductB()             │
└───────────┬─────────────────────┘
            △
            │ implements
    ┌───────┴────────┐
    │                │
┌──────────────┐  ┌──────────────┐
│ConcreteFactory│  │ConcreteFactory
│      1       │  │      2       │
└──────────────┘  └──────────────┘
     │                   │
     ├─ createProductA()─┤
     │  (返回ProductA1) │  (返回ProductA2)
     │                   │
     ├─ createProductB()─┤
     │  (返回ProductB1) │  (返回ProductB2)
```

## 代码实现

### 1. 定义抽象产品接口

```java
// 产品族 A
public interface Button {
    void render();
}

// 产品族 B
public interface Checkbox {
    void render();
}
```

### 2. 具体产品实现

```java
// Windows风格产品
public class WindowsButton implements Button {
    @Override
    public void render() {
        System.out.println("绘制Windows风格按钮");
    }
}

public class WindowsCheckbox implements Checkbox {
    @Override
    public void render() {
        System.out.println("绘制Windows风格复选框");
    }
}

// Mac风格产品
public class MacButton implements Button {
    @Override
    public void render() {
        System.out.println("绘制Mac风格按钮");
    }
}

public class MacCheckbox implements Checkbox {
    @Override
    public void render() {
        System.out.println("绘制Mac风格复选框");
    }
}

// Linux风格产品
public class LinuxButton implements Button {
    @Override
    public void render() {
        System.out.println("绘制Linux风格按钮");
    }
}

public class LinuxCheckbox implements Checkbox {
    @Override
    public void render() {
        System.out.println("绘制Linux风格复选框");
    }
}
```

### 3. 定义抽象工厂

```java
public interface UIFactory {
    Button createButton();
    Checkbox createCheckbox();
}
```

### 4. 具体工厂实现

```java
public class WindowsFactory implements UIFactory {
    @Override
    public Button createButton() {
        return new WindowsButton();
    }
    
    @Override
    public Checkbox createCheckbox() {
        return new WindowsCheckbox();
    }
}

public class MacFactory implements UIFactory {
    @Override
    public Button createButton() {
        return new MacButton();
    }
    
    @Override
    public Checkbox createCheckbox() {
        return new MacCheckbox();
    }
}

public class LinuxFactory implements UIFactory {
    @Override
    public Button createButton() {
        return new LinuxButton();
    }
    
    @Override
    public Checkbox createCheckbox() {
        return new LinuxCheckbox();
    }
}
```

### 5. 客户端使用

```java
public class Application {
    private UIFactory factory;
    private Button button;
    private Checkbox checkbox;
    
    public Application(UIFactory factory) {
        this.factory = factory;
    }
    
    public void render() {
        button = factory.createButton();
        checkbox = factory.createCheckbox();
        
        button.render();
        checkbox.render();
    }
    
    public static void main(String[] args) {
        // 根据操作系统选择工厂
        UIFactory factory = null;
        String os = System.getProperty("os.name");
        
        if (os.contains("Windows")) {
            factory = new WindowsFactory();
        } else if (os.contains("Mac")) {
            factory = new MacFactory();
        } else {
            factory = new LinuxFactory();
        }
        
        Application app = new Application(factory);
        app.render();
    }
}
```

## 实际应用示例

### 数据库驱动工厂

```java
// 抽象产品
public interface Connection {
    void connect();
}

public interface Statement {
    void execute(String sql);
}

// MySQL驱动
public class MySQLConnection implements Connection {
    @Override
    public void connect() {
        System.out.println("连接到MySQL");
    }
}

public class MySQLStatement implements Statement {
    @Override
    public void execute(String sql) {
        System.out.println("MySQL执行: " + sql);
    }
}

// PostgreSQL驱动
public class PostgreSQLConnection implements Connection {
    @Override
    public void connect() {
        System.out.println("连接到PostgreSQL");
    }
}

public class PostgreSQLStatement implements Statement {
    @Override
    public void execute(String sql) {
        System.out.println("PostgreSQL执行: " + sql);
    }
}

// 抽象工厂
public interface DatabaseFactory {
    Connection createConnection();
    Statement createStatement();
}

// 具体工厂
public class MySQLFactory implements DatabaseFactory {
    @Override
    public Connection createConnection() {
        return new MySQLConnection();
    }
    
    @Override
    public Statement createStatement() {
        return new MySQLStatement();
    }
}

public class PostgreSQLFactory implements DatabaseFactory {
    @Override
    public Connection createConnection() {
        return new PostgreSQLConnection();
    }
    
    @Override
    public Statement createStatement() {
        return new PostgreSQLStatement();
    }
}
```

### 文档导出工厂

```java
public interface Document {
    void save(String filename);
}

public interface Exporter {
    void export(Document doc);
}

// PDF实现
public class PDFDocument implements Document {
    @Override
    public void save(String filename) {
        System.out.println("保存为PDF: " + filename);
    }
}

public class PDFExporter implements Exporter {
    @Override
    public void export(Document doc) {
        System.out.println("导出为PDF");
    }
}

// Word实现
public class WordDocument implements Document {
    @Override
    public void save(String filename) {
        System.out.println("保存为Word: " + filename);
    }
}

public class WordExporter implements Exporter {
    @Override
    public void export(Document doc) {
        System.out.println("导出为Word");
    }
}

// 抽象工厂
public interface DocumentFactory {
    Document createDocument();
    Exporter createExporter();
}

// 具体工厂
public class PDFFactory implements DocumentFactory {
    @Override
    public Document createDocument() {
        return new PDFDocument();
    }
    
    @Override
    public Exporter createExporter() {
        return new PDFExporter();
    }
}
```

## 抽象工厂 vs 工厂方法

| 特性 | 抽象工厂 | 工厂方法 |
|------|--------|---------|
| 创建对象 | 产品族 | 单一产品 |
| 工厂数量 | 多个 | 一个或多个 |
| 接口方法 | 多个 | 一个 |
| 复杂度 | 高 | 低 |

## 优缺点

### 优点
- ✅ 保证产品族的一致性
- ✅ 易于切换产品族
- ✅ 符合开闭原则
- ✅ 便于维护和扩展

### 缺点
- ❌ 代码复杂度高
- ❌ 类的数量多
- ❌ 添加新产品困难

## 适用场景

- ✓ 系统有多个产品族
- ✓ 需要切换产品族
- ✓ 产品族之间有关系
- ✓ 需要保证产品族的一致性

## Java中的应用

```java
// AWT中的抽象工厂
Toolkit toolkit = Toolkit.getDefaultToolkit();

// Swing中的LookAndFeel
UIManager.setLookAndFeel(new MetalLookAndFeel());
```
