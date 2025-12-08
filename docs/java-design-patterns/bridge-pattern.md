---
sidebar_position: 12
---

# 桥接模式 (Bridge Pattern)

## 📌 模式定义

桥接模式是一种结构型设计模式，用于**将抽象与实现分离，使它们可以独立地变化**。

通过引入一个抽象层来桥接抽象和实现，从而使两者可以沿着各自的维度独立变化。

## 🤔 问题分析

### 为什么需要桥接模式？

假设你需要开发跨平台的图形API：

```
需求：支持多种形状（圆形、矩形）和多个绘制系统（Windows、Mac、Linux）

直接做法：创建 WindowsCircle, WindowsRectangle, MacCircle, MacRectangle...
问题：类爆炸！N个形状 × M个平台 = N×M 个类
```

另一个常见场景：

```
需求：数据库驱动（MySQL、PostgreSQL、Oracle）需要支持不同的连接池

直接做法：MySQLWithC3P0, MySQLWithHikariCP, PostgreSQLWithC3P0...
问题：维护困难，修改某一维度时需要改多个类
```

**根本问题**：
- 多维度变化导致类的爆炸性增长
- 修改一个维度时，需要改动所有相关类
- 子类继承导致抽象层和实现层耦合

## 💡 解决方案

**桥接模式的核心思想**：
- 将问题分为两个维度
- 为每个维度创建独立的抽象层
- 通过一个"桥"连接两个抽象层
- 两个维度可以独立变化

### 架构图

```
客户端
  ↓
抽象化角色 (Abstraction)
  ↓
┌─────────────────┐
│  桥接（聚合）   │
└─────────────────┘
  ↓
实现化角色接口 (Implementor)
  ↓
具体实现 (ConcreteImplementor)
```

## 💻 代码实现

### 示例1：跨平台图形API

```java
// 实现化接口 - 平台相关
public interface Implementor {
    void drawCircle(double radius);
    void drawRectangle(double width, double height);
}

// 具体实现 - Windows平台
public class WindowsImplementor implements Implementor {
    @Override
    public void drawCircle(double radius) {
        System.out.println("使用Windows API绘制圆形，半径: " + radius);
    }

    @Override
    public void drawRectangle(double width, double height) {
        System.out.println("使用Windows API绘制矩形，宽: " + width + ", 高: " + height);
    }
}

// 具体实现 - Mac平台
public class MacImplementor implements Implementor {
    @Override
    public void drawCircle(double radius) {
        System.out.println("使用Mac API绘制圆形，半径: " + radius);
    }

    @Override
    public void drawRectangle(double width, double height) {
        System.out.println("使用Mac API绘制矩形，宽: " + width + ", 高: " + height);
    }
}

// 抽象化角色 - 形状
public abstract class Shape {
    protected Implementor implementor;
    
    public Shape(Implementor implementor) {
        this.implementor = implementor;
    }
    
    public abstract void draw();
}

// 具体抽象化 - 圆形
public class Circle extends Shape {
    private double radius;
    
    public Circle(double radius, Implementor implementor) {
        super(implementor);
        this.radius = radius;
    }
    
    @Override
    public void draw() {
        implementor.drawCircle(radius);
    }
}

// 具体抽象化 - 矩形
public class Rectangle extends Shape {
    private double width;
    private double height;
    
    public Rectangle(double width, double height, Implementor implementor) {
        super(implementor);
        this.width = width;
        this.height = height;
    }
    
    @Override
    public void draw() {
        implementor.drawRectangle(width, height);
    }
}

// 客户端使用
public class Client {
    public static void main(String[] args) {
        // 创建Windows平台的圆形
        Shape windowsCircle = new Circle(5, new WindowsImplementor());
        windowsCircle.draw();  // 使用Windows API绘制圆形
        
        // 创建Mac平台的矩形
        Shape macRect = new Rectangle(10, 20, new MacImplementor());
        macRect.draw();  // 使用Mac API绘制矩形
        
        // 切换平台很容易！
        Shape macCircle = new Circle(5, new MacImplementor());
        macCircle.draw();  // 使用Mac API绘制圆形
    }
}
```

### 示例2：数据库驱动+连接池

```java
// 实现化接口 - 连接池
public interface ConnectionPool {
    Connection getConnection();
    void releaseConnection(Connection conn);
}

// 具体实现 - C3P0连接池
public class C3P0ConnectionPool implements ConnectionPool {
    @Override
    public Connection getConnection() {
        System.out.println("从C3P0连接池获取连接");
        return null;  // 实际返回连接
    }
    
    @Override
    public void releaseConnection(Connection conn) {
        System.out.println("归还连接到C3P0连接池");
    }
}

// 具体实现 - HikariCP连接池
public class HikariCPConnectionPool implements ConnectionPool {
    @Override
    public Connection getConnection() {
        System.out.println("从HikariCP连接池获取连接");
        return null;
    }
    
    @Override
    public void releaseConnection(Connection conn) {
        System.out.println("归还连接到HikariCP连接池");
    }
}

// 抽象化 - 数据库驱动
public abstract class DatabaseDriver {
    protected ConnectionPool connectionPool;
    
    public DatabaseDriver(ConnectionPool connectionPool) {
        this.connectionPool = connectionPool;
    }
    
    public abstract void executeQuery(String sql);
}

// 具体抽象化 - MySQL驱动
public class MySQLDriver extends DatabaseDriver {
    @Override
    public void executeQuery(String sql) {
        Connection conn = connectionPool.getConnection();
        System.out.println("MySQL执行查询: " + sql);
        connectionPool.releaseConnection(conn);
    }
}

// 具体抽象化 - PostgreSQL驱动
public class PostgreSQLDriver extends DatabaseDriver {
    @Override
    public void executeQuery(String sql) {
        Connection conn = connectionPool.getConnection();
        System.out.println("PostgreSQL执行查询: " + sql);
        connectionPool.releaseConnection(conn);
    }
}

// 客户端使用
public class DatabaseClient {
    public static void main(String[] args) {
        // MySQL + C3P0
        DatabaseDriver mysql = new MySQLDriver(new C3P0ConnectionPool());
        mysql.executeQuery("SELECT * FROM users");
        
        // PostgreSQL + HikariCP
        DatabaseDriver postgres = new PostgreSQLDriver(new HikariCPConnectionPool());
        postgres.executeQuery("SELECT * FROM orders");
    }
}
```

### 示例3：日志框架 (slf4j + logback/log4j)

```java
// 实现化接口 - 日志输出
public interface Logger {
    void log(String message);
}

// 具体实现 - 控制台输出
public class ConsoleLogger implements Logger {
    @Override
    public void log(String message) {
        System.out.println("[Console] " + message);
    }
}

// 具体实现 - 文件输出
public class FileLogger implements Logger {
    @Override
    public void log(String message) {
        System.out.println("[File] 写入文件: " + message);
    }
}

// 抽象化 - 日志门面
public abstract class LoggerFacade {
    protected Logger logger;
    
    public LoggerFacade(Logger logger) {
        this.logger = logger;
    }
    
    public abstract void info(String message);
    public abstract void error(String message);
}

// 具体抽象化 - 应用日志
public class ApplicationLogger extends LoggerFacade {
    @Override
    public void info(String message) {
        logger.log("[INFO] " + message);
    }
    
    @Override
    public void error(String message) {
        logger.log("[ERROR] " + message);
    }
}

// 具体抽象化 - 系统日志
public class SystemLogger extends LoggerFacade {
    @Override
    public void info(String message) {
        logger.log("[SYSTEM INFO] " + message);
    }
    
    @Override
    public void error(String message) {
        logger.log("[SYSTEM ERROR] " + message);
    }
}

// 使用
public class LoggerTest {
    public static void main(String[] args) {
        // 应用日志 + 控制台
        LoggerFacade appLog = new ApplicationLogger(new ConsoleLogger());
        appLog.info("应用启动");
        appLog.error("数据库连接失败");
        
        // 系统日志 + 文件
        LoggerFacade sysLog = new SystemLogger(new FileLogger());
        sysLog.info("系统初始化");
        sysLog.error("权限不足");
    }
}
```

## ✅ 优点

- ✨ **解耦抽象与实现** - 两个维度可以独立变化
- 📦 **避免类爆炸** - 不再需要为每个组合创建子类
- 🔄 **灵活组合** - 可以动态切换实现
- 📈 **易于扩展** - 新增维度时只需添加新类
- 🔒 **符合开闭原则** - 对扩展开放，对修改关闭

## ❌ 缺点

- 🔗 **增加系统复杂度** - 引入额外的抽象层
- 📚 **理解困难** - 需要理解两个维度的关系
- ⚠️ **设计阶段困难** - 需要提前识别变化维度

## 🎯 适用场景

✓ **多维度变化** - 有两个独立变化的维度  
✓ **避免类爆炸** - 组合数量会很大  
✓ **抽象-实现分离** - 需要分离抽象和平台相关代码  
✓ **动态选择实现** - 运行时决定使用哪个实现  

**实际应用**：
- JDBC - Statement和Connection的分离
- AWT/Swing - Component和Peer
- Slf4j + Logback/Log4j
- Spring远程访问 (RMI, WebServices)
- 数据库驱动和连接池的组合

## 📊 vs 其他模式

| 模式 | 区别 |
|------|------|
| **Adapter** | Adapter是事后补救，Bridge是提前设计 |
| **Strategy** | Strategy是算法选择，Bridge是抽象-实现分离 |
| **Abstract Factory** | Abstract Factory创建对象族，Bridge分离维度 |

## 🔗 模式关系

- **与Adapter的关系** - Bridge通常在设计阶段，Adapter用于解决现有代码不匹配
- **与Abstract Factory的关系** - 可以配合使用，Abstract Factory创建实现对象
- **与Strategy的关系** - Bridge是结构性分离，Strategy是算法选择

## 💡 最佳实践

1. **提前识别维度** - 设计时要识别出两个独立变化的维度
2. **不要过度设计** - 只在确实有多个维度时使用
3. **清晰的抽象** - 定义好抽象接口和实现接口的职责
4. **考虑使用工厂** - 结合工厂模式创建桥接对象
5. **文档清晰** - 记录两个维度是什么

## 🚀 实现建议

```java
// 好的实践：清晰的维度划分
// 维度1：形状（Circle, Rectangle）
// 维度2：绘制方式（Windows, Mac）

Shape shape = new Circle(5, new WindowsImplementor());

// 不好的实践：过度复杂的抽象
// 试图用Bridge处理超过2个维度的变化
```

---

桥接模式优雅地解决了多维度变化带来的类爆炸问题。关键是**正确识别维度**！
