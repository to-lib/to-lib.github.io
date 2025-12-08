---
sidebar_position: 8
---

# 代理模式 (Proxy Pattern)

## 模式定义

**代理模式**是一种结构型设计模式，它为另一个对象提供一个替身或占位符，以便控制对这个对象的访问。

## 问题分析

有时候我们需要控制或增强对象的访问：

- 延迟初始化（懒加载）
- 访问控制
- 日志记录
- 缓存
- 性能监控
- 远程代理（RPC）

## 解决方案

```
┌──────────────────┐
│   Subject接口    │
│  + doSomething() │
└────────┬─────────┘
         △
         │ implements
    ┌────┴───────────────┐
    │                    │
┌────────────┐      ┌──────────┐
│Real Subject│      │ Proxy    │
│            │      │- subject │
└────────────┘      │+ doSomething()│
                    └──────────┘
                        │uses
                     Real Subject
```

## 代码实现

### 1. 定义主题接口

```java
public interface Image {
    void display();
}
```

### 2. 真实主题

```java
public class RealImage implements Image {
    private String fileName;
    
    public RealImage(String fileName) {
        this.fileName = fileName;
        loadFromDisk();
    }
    
    private void loadFromDisk() {
        System.out.println("从磁盘加载图片: " + fileName);
    }
    
    @Override
    public void display() {
        System.out.println("显示图片: " + fileName);
    }
}
```

### 3. 代理类

```java
public class ImageProxy implements Image {
    private String fileName;
    private RealImage realImage;
    
    public ImageProxy(String fileName) {
        this.fileName = fileName;
    }
    
    @Override
    public void display() {
        // 延迟加载
        if (realImage == null) {
            realImage = new RealImage(fileName);
        }
        realImage.display();
    }
}
```

### 4. 客户端使用

```java
public class Demo {
    public static void main(String[] args) {
        Image image1 = new ImageProxy("photo1.jpg");
        Image image2 = new ImageProxy("photo2.jpg");
        
        // 此时还没有加载图片
        System.out.println("代理创建完成");
        
        // 第一次调用时才加载
        image1.display();
        
        // 第二次调用直接使用缓存
        image1.display();
        
        image2.display();
    }
}
```

## 实际应用示例

### 访问控制代理

```java
public interface User {
    void doWork();
}

public class RealUser implements User {
    private String name;
    
    public RealUser(String name) {
        this.name = name;
    }
    
    @Override
    public void doWork() {
        System.out.println(name + " 在工作");
    }
}

public class UserProxy implements User {
    private RealUser realUser;
    private String role;
    
    public UserProxy(RealUser user, String role) {
        this.realUser = user;
        this.role = role;
    }
    
    @Override
    public void doWork() {
        if ("ADMIN".equals(role) || "MANAGER".equals(role)) {
            realUser.doWork();
        } else {
            System.out.println("权限不足，无法执行操作");
        }
    }
}
```

### 日志记录代理

```java
public interface DatabaseService {
    void query(String sql);
    void update(String sql);
}

public class Database implements DatabaseService {
    @Override
    public void query(String sql) {
        System.out.println("执行查询: " + sql);
    }
    
    @Override
    public void update(String sql) {
        System.out.println("执行更新: " + sql);
    }
}

public class DatabaseProxy implements DatabaseService {
    private Database database;
    private Logger logger;
    
    public DatabaseProxy(Database database, Logger logger) {
        this.database = database;
        this.logger = logger;
    }
    
    @Override
    public void query(String sql) {
        logger.log("开始查询: " + sql);
        long start = System.currentTimeMillis();
        
        database.query(sql);
        
        long duration = System.currentTimeMillis() - start;
        logger.log("查询完成，耗时: " + duration + "ms");
    }
    
    @Override
    public void update(String sql) {
        logger.log("开始更新: " + sql);
        database.update(sql);
        logger.log("更新完成");
    }
}
```

### 缓存代理

```java
public interface CacheService {
    String getData(String key);
}

public class RealCacheService implements CacheService {
    @Override
    public String getData(String key) {
        // 从数据库或远程服务获取数据
        System.out.println("从数据源获取数据: " + key);
        return "data_" + key;
    }
}

public class CacheProxy implements CacheService {
    private RealCacheService realService;
    private Map<String, String> cache = new HashMap<>();
    
    public CacheProxy(RealCacheService realService) {
        this.realService = realService;
    }
    
    @Override
    public String getData(String key) {
        if (cache.containsKey(key)) {
            System.out.println("从缓存获取: " + key);
            return cache.get(key);
        }
        
        String data = realService.getData(key);
        cache.put(key, data);
        return data;
    }
}
```

### 远程代理（RPC）

```java
// 远程接口
public interface RemoteService {
    String remoteMethod(String param);
}

// 远程实现（在远程服务器）
public class RemoteServiceImpl implements RemoteService {
    @Override
    public String remoteMethod(String param) {
        return "远程响应: " + param;
    }
}

// 远程代理（在客户端）
public class RemoteServiceProxy implements RemoteService {
    private String serverAddress;
    
    public RemoteServiceProxy(String serverAddress) {
        this.serverAddress = serverAddress;
    }
    
    @Override
    public String remoteMethod(String param) {
        System.out.println("连接到服务器: " + serverAddress);
        System.out.println("发送请求参数: " + param);
        // 实际的网络通信
        return "收到远程响应";
    }
}
```

## 代理模式的类型

### 1. 虚代理（Virtual Proxy）
延迟初始化，当真正需要时才创建：

```java
public class LazyProxy implements Image {
    private RealImage realImage;
    private String fileName;
    
    public LazyProxy(String fileName) {
        this.fileName = fileName;
    }
    
    @Override
    public void display() {
        if (realImage == null) {
            realImage = new RealImage(fileName);
        }
        realImage.display();
    }
}
```

### 2. 保护代理（Protection Proxy）
控制对真实对象的访问权限

### 3. 远程代理（Remote Proxy）
为远程对象提供本地代理

### 4. 智能引用（Smart Reference）
在访问时进行额外处理

## 代理模式 vs 其他模式

| 模式 | 目的 | 实现 |
|------|------|------|
| 代理 | 控制访问 | 相同接口 |
| 装饰器 | 增加功能 | 增强接口 |
| 适配器 | 转换接口 | 转换接口 |
| 外观 | 简化使用 | 统一接口 |

## 优缺点

### 优点
- ✅ 控制和增强对象访问
- ✅ 延迟初始化，节省资源
- ✅ 实现日志、缓存等功能
- ✅ 遵守开闭原则

### 缺点
- ❌ 增加代码复杂性
- ❌ 性能可能下降（额外的代理层）
- ❌ 响应时间增加

## 适用场景

- ✓ 访问控制
- ✓ 日志记录
- ✓ 缓存管理
- ✓ 延迟初始化
- ✓ 远程代理（RPC）
- ✓ 性能监控

## Java中的代理

```java
// 静态代理 - 手动编写代理类

// 动态代理 - JDK自带
Object proxy = Proxy.newProxyInstance(
    targetClass.getClassLoader(),
    targetClass.getInterfaces(),
    new InvocationHandler() {
        @Override
        public Object invoke(Object o, Method method, Object[] args) {
            // 前置处理
            Object result = method.invoke(target, args);
            // 后置处理
            return result;
        }
    }
);

// 字节码增强 - CGLIB库
Enhancer enhancer = new Enhancer();
enhancer.setSuperclass(RealClass.class);
enhancer.setCallback(new MethodInterceptor() {
    // 拦截方法
});
Object proxy = enhancer.create();
```
