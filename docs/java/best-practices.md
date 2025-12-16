---
sidebar_position: 20
title: Java 最佳实践
---

# Java 最佳实践

本文总结了 Java 编程中的最佳实践，帮助你写出更优雅、高效和可维护的代码。

## 命名规范

### 类和接口

```java
// ✅ 使用 PascalCase（大驼峰）
public class UserService { }
public interface PaymentProcessor { }

// ❌ 避免
public class userservice { }
public class User_Service { }
```

### 方法和变量

```java
// ✅ 使用 camelCase（小驼峰）
public void calculateTotalPrice() { }
private String userName;

// ❌ 避免
public void CalculateTotalPrice() { }
private String user_name;
```

### 常量

```java
// ✅ 使用大写字母和下划线
public static final int MAX_RETRY_COUNT = 3;
public static final String DEFAULT_ENCODING = "UTF-8";

// ❌ 避免
public static final int maxRetryCount = 3;
```

### 包名

```java
// ✅ 全小写，使用点分隔
package com.company.project.module;

// ❌ 避免
package com.Company.Project.Module;
```

## SOLID 原则

### 单一职责原则（SRP）

每个类应该只有一个改变的理由。

```java
// ❌ 违反 SRP：类承担太多职责
public class User {
    private String name;
    private String email;

    public void save() {
        // 保存到数据库
    }

    public void sendEmail() {
        // 发送邮件
    }

    public void generateReport() {
        // 生成报告
    }
}

// ✅ 遵守 SRP：职责分离
public class User {
    private String name;
    private String email;
    // getter/setter
}

public class UserRepository {
    public void save(User user) {
        // 保存到数据库
    }
}

public class EmailService {
    public void sendEmail(User user) {
        // 发送邮件
    }
}

public class ReportGenerator {
    public void generateUserReport(User user) {
        // 生成报告
    }
}
```

### 开闭原则（OCP）

对扩展开放，对修改关闭。

```java
// ❌ 违反 OCP：添加新类型需要修改现有代码
public class DiscountCalculator {
    public double calculate(String type, double price) {
        if (type.equals("VIP")) {
            return price * 0.8;
        } else if (type.equals("REGULAR")) {
            return price * 0.9;
        }
        return price;
    }
}

// ✅ 遵守 OCP：使用策略模式
public interface DiscountStrategy {
    double calculate(double price);
}

public class VipDiscount implements DiscountStrategy {
    public double calculate(double price) {
        return price * 0.8;
    }
}

public class RegularDiscount implements DiscountStrategy {
    public double calculate(double price) {
        return price * 0.9;
    }
}

public class DiscountCalculator {
    private DiscountStrategy strategy;

    public DiscountCalculator(DiscountStrategy strategy) {
        this.strategy = strategy;
    }

    public double calculate(double price) {
        return strategy.calculate(price);
    }
}
```

### 里氏替换原则（LSP）

子类应该可以替换其父类。

```java
// ✅ 遵守 LSP
public class Rectangle {
    protected int width;
    protected int height;

    public void setWidth(int width) {
        this.width = width;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public int getArea() {
        return width * height;
    }
}

// ❌ 违反 LSP：正方形不能简单继承矩形
public class Square extends Rectangle {
    @Override
    public void setWidth(int width) {
        this.width = width;
        this.height = width;  // 破坏了父类的行为
    }
}
```

### 接口隔离原则（ISP）

不应强迫客户端依赖它们不使用的方法。

```java
// ❌ 违反 ISP：接口过大
public interface Worker {
    void work();
    void eat();
    void sleep();
}

// ✅ 遵守 ISP：接口细化
public interface Workable {
    void work();
}

public interface Eatable {
    void eat();
}

public interface Sleepable {
    void sleep();
}

public class Human implements Workable, Eatable, Sleepable {
    public void work() { }
    public void eat() { }
    public void sleep() { }
}

public class Robot implements Workable {
    public void work() { }
}
```

### 依赖倒置原则（DIP）

依赖抽象而不是具体实现。

```java
// ❌ 违反 DIP：依赖具体类
public class UserService {
    private MySQLDatabase database = new MySQLDatabase();

    public void saveUser(User user) {
        database.save(user);
    }
}

// ✅ 遵守 DIP：依赖抽象
public interface Database {
    void save(User user);
}

public class MySQLDatabase implements Database {
    public void save(User user) {
        // MySQL 实现
    }
}

public class UserService {
    private Database database;

    public UserService(Database database) {
        this.database = database;
    }

    public void saveUser(User user) {
        database.save(user);
    }
}
```

## 异常处理

### 选择合适的异常类型

```java
// ✅ 使用具体的异常类型
public User findUserById(Long id) throws UserNotFoundException {
    User user = userRepository.findById(id);
    if (user == null) {
        throw new UserNotFoundException("用户不存在: " + id);
    }
    return user;
}

// ❌ 避免使用通用异常
public User findUserById(Long id) throws Exception {
    // ...
}
```

### 不要吞掉异常

```java
// ❌ 吞掉异常
try {
    riskyOperation();
} catch (Exception e) {
    // 什么都不做
}

// ✅ 至少记录日志
try {
    riskyOperation();
} catch (Exception e) {
    logger.error("操作失败", e);
    throw new RuntimeException("操作失败", e);
}
```

### 使用 try-with-resources

```java
// ❌ 手动关闭资源
BufferedReader reader = null;
try {
    reader = new BufferedReader(new FileReader("file.txt"));
    return reader.readLine();
} finally {
    if (reader != null) {
        reader.close();
    }
}

// ✅ 使用 try-with-resources
try (BufferedReader reader = new BufferedReader(new FileReader("file.txt"))) {
    return reader.readLine();
}
```

### 创建有意义的异常

```java
// ✅ 自定义异常提供详细信息
public class UserNotFoundException extends RuntimeException {
    private final Long userId;

    public UserNotFoundException(Long userId) {
        super("用户不存在: " + userId);
        this.userId = userId;
    }

    public Long getUserId() {
        return userId;
    }
}
```

## 并发编程

### 优先使用并发工具类

```java
// ❌ 使用同步代码块
private int counter = 0;
public synchronized void increment() {
    counter++;
}

// ✅ 使用 AtomicInteger
private AtomicInteger counter = new AtomicInteger(0);
public void increment() {
    counter.incrementAndGet();
}
```

### 使用线程池

```java
// ❌ 每次创建新线程
for (int i = 0; i < 100; i++) {
    new Thread(() -> doWork()).start();
}

// ✅ 使用线程池
ExecutorService executor = Executors.newFixedThreadPool(10);
for (int i = 0; i < 100; i++) {
    executor.submit(() -> doWork());
}
executor.shutdown();
```

### 避免死锁

```java
// ✅ 按固定顺序获取锁
public void transfer(Account from, Account to, double amount) {
    Account first = from.getId() < to.getId() ? from : to;
    Account second = from.getId() < to.getId() ? to : from;

    synchronized (first) {
        synchronized (second) {
            from.debit(amount);
            to.credit(amount);
        }
    }
}
```

## 集合使用

### 指定初始容量

```java
// ✅ 预知大小时指定初始容量
List<String> list = new ArrayList<>(1000);
Map<String, Integer> map = new HashMap<>(100);

// ❌ 使用默认容量（可能导致多次扩容）
List<String> list = new ArrayList<>();
```

### 使用合适的集合类型

```java
// ✅ 根据需求选择集合
List<String> names = new ArrayList<>();  // 需要按索引访问
Set<String> uniqueNames = new HashSet<>();  // 需要去重
Map<String, User> userMap = new HashMap<>();  // 需要键值对

// ✅ 需要线程安全时
List<String> syncList = Collections.synchronizedList(new ArrayList<>());
Map<String, User> concurrentMap = new ConcurrentHashMap<>();
```

### 避免在循环中修改集合

```java
// ❌ ConcurrentModificationException
List<String> list = new ArrayList<>(Arrays.asList("a", "b", "c"));
for (String item : list) {
    if (item.equals("b")) {
        list.remove(item);
    }
}

// ✅ 使用 Iterator
Iterator<String> iterator = list.iterator();
while (iterator.hasNext()) {
    if (iterator.next().equals("b")) {
        iterator.remove();
    }
}

// ✅ 使用 removeIf
list.removeIf(item -> item.equals("b"));
```

## 资源管理

### 关闭资源

```java
// ✅ 使用 try-with-resources
try (Connection conn = dataSource.getConnection();
     Statement stmt = conn.createStatement();
     ResultSet rs = stmt.executeQuery("SELECT * FROM users")) {
    while (rs.next()) {
        // 处理结果
    }
}
```

### 及时释放大对象

```java
// ✅ 不再使用时置为 null
public void processLargeData() {
    byte[] largeArray = new byte[1024 * 1024 * 100];  // 100MB
    // 处理数据
    processData(largeArray);
    largeArray = null;  // 帮助 GC
}
```

## 性能优化

### 使用 StringBuilder

```java
// ❌ 在循环中使用字符串拼接
String result = "";
for (int i = 0; i < 1000; i++) {
    result += i;
}

// ✅ 使用 StringBuilder
StringBuilder sb = new StringBuilder();
for (int i = 0; i < 1000; i++) {
    sb.append(i);
}
String result = sb.toString();
```

### 缓存昂贵的计算

```java
// ✅ 缓存计算结果
public class MathCalculator {
    private Map<Integer, Integer> cache = new HashMap<>();

    public int factorial(int n) {
        if (cache.containsKey(n)) {
            return cache.get(n);
        }

        int result = calculateFactorial(n);
        cache.put(n, result);
        return result;
    }
}
```

### 延迟初始化

```java
// ✅ 延迟初始化（只在需要时创建）
public class HeavyObject {
    private ExpensiveResource resource;

    public ExpensiveResource getResource() {
        if (resource == null) {
            resource = new ExpensiveResource();
        }
        return resource;
    }
}
```

## 代码可读性

### 方法不要太长

```java
// ✅ 将长方法拆分
public void processOrder(Order order) {
    validateOrder(order);
    calculateTotal(order);
    applyDiscount(order);
    saveOrder(order);
    sendConfirmation(order);
}

private void validateOrder(Order order) { }
private void calculateTotal(Order order) { }
private void applyDiscount(Order order) { }
private void saveOrder(Order order) { }
private void sendConfirmation(Order order) { }
```

### 避免魔法数字

```java
// ❌ 魔法数字
if (status == 1) {
    // ...
}

// ✅ 使用常量
public static final int STATUS_ACTIVE = 1;
if (status == STATUS_ACTIVE) {
    // ...
}

// ✅ 更好：使用枚举
public enum Status {
    ACTIVE, INACTIVE, PENDING
}
if (status == Status.ACTIVE) {
    // ...
}
```

### 使用有意义的变量名

```java
// ❌ 无意义的名称
int d;  // 天数？
String s;  // 什么字符串？

// ✅ 有意义的名称
int daysUntilExpiration;
String userName;
```

## 设计模式应用

### 单例模式

```java
// ✅ 线程安全的单例
public class Singleton {
    private static volatile Singleton instance;

    private Singleton() { }

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

// ✅ 更好：使用枚举
public enum Singleton {
    INSTANCE;

    public void doSomething() { }
}
```

### 工厂模式

```java
// ✅ 工厂方法
public interface Shape {
    void draw();
}

public class ShapeFactory {
    public static Shape createShape(String type) {
        switch (type) {
            case "CIRCLE": return new Circle();
            case "SQUARE": return new Square();
            default: throw new IllegalArgumentException("未知类型");
        }
    }
}
```

## 日志记录

```java
// ✅ 使用合适的日志级别
logger.debug("调试信息: {}", debugInfo);
logger.info("用户登录: {}", username);
logger.warn("配置缺失，使用默认值");
logger.error("操作失败", exception);

// ✅ 使用参数化消息（避免字符串拼接）
logger.info("用户 {} 完成了订单 {}", userId, orderId);

// ❌ 避免
logger.info("用户 " + userId + " 完成了订单 " + orderId);
```

## 测试

```java
// ✅ 编写单元测试
@Test
public void testCalculateTotalPrice() {
    // Arrange
    Order order = new Order();
    order.addItem(new Item("商品1", 100));
    order.addItem(new Item("商品2", 200));

    // Act
    double total = order.calculateTotal();

    // Assert
    assertEquals(300, total, 0.01);
}
```

## 总结

- **命名规范**：遵循 Java 命名约定
- **SOLID 原则**：编写可维护的面向对象代码
- **异常处理**：使用合适的异常类型，不要吞掉异常
- **并发编程**：使用并发工具类，避免死锁
- **集合使用**：选择合适的集合类型，指定初始容量
- **资源管理**：使用 try-with-resources
- **性能优化**：缓存计算结果，使用 StringBuilder
- **代码可读性**：方法简短，使用有意义的命名
- **设计模式**：合理应用设计模式
- **日志记录**：使用合适的日志级别
- **测试**：编写单元测试

遵循这些最佳实践能够显著提高代码质量和可维护性。
