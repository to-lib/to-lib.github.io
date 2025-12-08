---
sidebar_position: 2
---

# 单例模式 (Singleton Pattern)

## 模式定义

**单例模式**是一种创建型设计模式，它保证一个类只有一个实例，并提供一个全局访问点来获取这个实例。

## 问题分析

在某些情况下，我们希望某个类在整个应用程序生命周期内只存在一个实例：

- 数据库连接池
- 日志记录器
- 配置管理器
- 线程池
- 缓存管理器

## 实现方式

### 1. 懒加载单例（线程不安全）

```java
public class Singleton {
    private static Singleton instance;
    
    private Singleton() {
        // 私有构造函数，防止外部实例化
    }
    
    public static Singleton getInstance() {
        if (instance == null) {
            instance = new Singleton();
        }
        return instance;
    }
}
```

**缺点**：在多线程环境下不安全，可能创建多个实例。

### 2. 饿汉式单例（线程安全）

```java
public class Singleton {
    private static final Singleton instance = new Singleton();
    
    private Singleton() {
    }
    
    public static Singleton getInstance() {
        return instance;
    }
}
```

**优点**：线程安全，实现简单
**缺点**：类加载时就创建实例，可能浪费资源

### 3. 懒加载单例（同步方法）

```java
public class Singleton {
    private static Singleton instance;
    
    private Singleton() {
    }
    
    public static synchronized Singleton getInstance() {
        if (instance == null) {
            instance = new Singleton();
        }
        return instance;
    }
}
```

**优点**：线程安全
**缺点**：性能影响，每次调用都需要同步

### 4. 双重检查锁定（推荐）

```java
public class Singleton {
    private static volatile Singleton instance;
    
    private Singleton() {
    }
    
    public static Singleton getInstance() {
        if (instance == null) {
            synchronized (Singleton.class) {
                if (instance == null) {
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}
```

**优点**：线程安全且高效
**关键点**：volatile 关键字防止指令重排

### 5. 静态内部类单例（最推荐）

```java
public class Singleton {
    private Singleton() {
    }
    
    private static class SingletonHolder {
        static final Singleton INSTANCE = new Singleton();
    }
    
    public static Singleton getInstance() {
        return SingletonHolder.INSTANCE;
    }
}
```

**优点**：
- 线程安全（由JVM保证）
- 懒加载（第一次调用时才加载）
- 性能最优
- 代码简洁

### 6. 枚举单例（最安全）

```java
public enum Singleton {
    INSTANCE;
    
    public void doSomething() {
        // 实现业务逻辑
    }
}

// 使用方式
Singleton.INSTANCE.doSomething();
```

**优点**：
- 线程安全
- 防止反射和序列化破坏单例
- 代码最简洁

## 实际应用示例

### 日志记录器

```java
public class Logger {
    private static Logger instance;
    
    private Logger() {
    }
    
    public static synchronized Logger getInstance() {
        if (instance == null) {
            instance = new Logger();
        }
        return instance;
    }
    
    public void log(String message) {
        System.out.println("[LOG] " + message);
    }
}

// 使用
Logger logger = Logger.getInstance();
logger.log("应用启动");
```

### 配置管理器

```java
public class ConfigManager {
    private static ConfigManager instance;
    private Map<String, String> config = new HashMap<>();
    
    private ConfigManager() {
        loadConfig();
    }
    
    private static class ConfigHolder {
        static final ConfigManager INSTANCE = new ConfigManager();
    }
    
    public static ConfigManager getInstance() {
        return ConfigHolder.INSTANCE;
    }
    
    private void loadConfig() {
        // 加载配置文件
        config.put("db.url", "jdbc:mysql://localhost:3306/mydb");
        config.put("db.user", "root");
    }
    
    public String get(String key) {
        return config.get(key);
    }
}
```

## 优缺点

### 优点
- ✅ 内存中仅有一个实例，节省资源
- ✅ 全局访问点，易于获取
- ✅ 自动初始化，延迟加载

### 缺点
- ❌ 隐藏了类的依赖关系
- ❌ 违反单一职责原则
- ❌ 不利于单元测试
- ❌ 可能造成全局状态混乱

## 适用场景

- ✓ 数据库连接
- ✓ 日志记录
- ✓ 配置管理
- ✓ 线程池
- ✓ 缓存管理
- ✓ 应用程序对象

## 注意事项

1. **线程安全** - 选择合适的实现方式
2. **反射破坏** - 使用枚举可防止反射
3. **序列化问题** - 实现 readResolve 方法
4. **测试困难** - 难以进行单元测试
5. **过度使用** - 避免过度依赖全局状态
