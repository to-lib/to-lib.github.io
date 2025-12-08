---
sidebar_position: 10
---

# 设计模式最佳实践

## 选择设计模式的原则

### 1. KISS原则（Keep It Simple, Stupid）
不要过度设计，在必要时才使用设计模式。

```java
// 简单场景，不需要设计模式
public class SimpleConfig {
    private String dbUrl = "localhost";
    
    public String getDbUrl() {
        return dbUrl;
    }
}

// 复杂场景，才使用单例模式
public enum GlobalConfig {
    INSTANCE;
    
    private final Properties props = new Properties();
    
    public String getProperty(String key) {
        return props.getProperty(key);
    }
}
```

### 2. DRY原则（Don't Repeat Yourself）
使用设计模式消除代码重复。

```java
// 不好：代码重复
public class UserService {
    public void createUser(String name) {
        System.out.println("记录日志：创建用户");
        // 创建用户逻辑
        System.out.println("记录日志：用户创建成功");
    }
    
    public void deleteUser(int id) {
        System.out.println("记录日志：删除用户");
        // 删除用户逻辑
        System.out.println("记录日志：用户删除成功");
    }
}

// 好：使用代理模式消除重复
public class LoggingProxy implements UserService {
    private UserService realService;
    
    public LoggingProxy(UserService realService) {
        this.realService = realService;
    }
    
    @Override
    public void createUser(String name) {
        log("创建用户");
        realService.createUser(name);
        log("用户创建成功");
    }
}
```

### 3. SOLID原则

#### 单一职责原则 (Single Responsibility)
每个类只负责一个功能。

```java
// 不好：一个类负责多个职责
public class User {
    private String name;
    
    public void save() { }        // 职责1：持久化
    public void validate() { }    // 职责2：验证
    public void sendEmail() { }   // 职责3：通知
}

// 好：分离职责
public class User {
    private String name;
}

public class UserRepository {
    public void save(User user) { }
}

public class UserValidator {
    public boolean validate(User user) { }
}

public class UserNotifier {
    public void sendEmail(User user) { }
}
```

#### 开闭原则 (Open/Closed Principle)
对扩展开放，对修改关闭。

```java
// 不好：添加新功能需要修改代码
public class PaymentProcessor {
    public void processPayment(String type, double amount) {
        if ("CARD".equals(type)) {
            // 信用卡处理
        } else if ("CASH".equals(type)) {
            // 现金处理
        }
        // 添加新方式时需要修改这里
    }
}

// 好：使用策略模式，对扩展开放
public interface PaymentStrategy {
    void process(double amount);
}

public class PaymentProcessor {
    private PaymentStrategy strategy;
    
    public void processPayment(double amount) {
        strategy.process(amount);  // 不需要修改代码
    }
}
```

#### 里氏替换原则 (Liskov Substitution)
子类应该能替换其父类。

```java
// 不好：违反LSP
public class Bird {
    public void fly() { }
}

public class Penguin extends Bird {
    @Override
    public void fly() {
        throw new UnsupportedOperationException("企鹅不会飞");
    }
}

// 好：使用正确的继承关系
public interface Flyable {
    void fly();
}

public class Sparrow implements Flyable {
    @Override
    public void fly() {
        System.out.println("麻雀飞行");
    }
}

public class Penguin {
    public void swim() {
        System.out.println("企鹅游泳");
    }
}
```

#### 接口隔离原则 (Interface Segregation)
客户端不应被迫依赖它们不使用的接口。

```java
// 不好：臃肿的接口
public interface Worker {
    void work();
    void manage();
    void report();
}

// 好：分离职责
public interface Worker {
    void work();
}

public interface Manager {
    void manage();
}

public interface Reporter {
    void report();
}

public class Employee implements Worker {
    @Override
    public void work() { }
}

public class Director implements Worker, Manager, Reporter {
    @Override
    public void work() { }
    
    @Override
    public void manage() { }
    
    @Override
    public void report() { }
}
```

#### 依赖倒置原则 (Dependency Inversion)
依赖抽象而不是具体实现。

```java
// 不好：依赖具体类
public class UserService {
    private MySQLDatabase database = new MySQLDatabase();
    
    public void saveUser(User user) {
        database.save(user);
    }
}

// 好：依赖抽象
public class UserService {
    private Database database;
    
    public UserService(Database database) {
        this.database = database;
    }
    
    public void saveUser(User user) {
        database.save(user);
    }
}

public interface Database {
    void save(User user);
}
```

## 常见的反模式及其解决方案

### 1. 上帝对象（God Object）
一个类做太多事情。

**解决**：分解成多个小类
```java
// 将UserManager分解为：
// - UserService：业务逻辑
// - UserRepository：数据访问
// - UserValidator：验证逻辑
// - UserNotifier：通知逻辑
```

### 2. 特性羡慕（Feature Envy）
一个类过度使用另一个类的方法。

**解决**：移动方法到正确的类
```java
// 不好
class OrderService {
    void processOrder(Order order) {
        order.setStatus("processing");
        order.setProcessDate(new Date());
        order.setTotal(calculateTotal(order));
    }
}

// 好
class Order {
    void process() {
        setStatus("processing");
        setProcessDate(new Date());
        setTotal(calculateTotal());
    }
}
```

### 3. 过度工程（Over Engineering）
使用过于复杂的设计处理简单问题。

**解决**：根据实际需求选择合适的设计
```java
// 不好：简单场景使用复杂设计
public enum LoggerFactory {
    INSTANCE;
    public Logger getLogger() { }
}

// 好：简单的实现
public class Logger {
    public static void log(String msg) {
        System.out.println(msg);
    }
}
```

## 设计模式应用清单

### 选择创建型模式

| 需求 | 推荐模式 |
|------|--------|
| 简单对象创建 | 直接new或简单工厂 |
| 多种产品类型 | 工厂方法模式 |
| 产品族 | 抽象工厂模式 |
| 复杂对象构建 | 建造者模式 |
| 使用现有对象副本 | 原型模式 |
| 全局唯一对象 | 单例模式 |

### 选择结构型模式

| 需求 | 推荐模式 |
|------|--------|
| 类接口转换 | 适配器模式 |
| 分离抽象和实现 | 桥接模式 |
| 树形结构 | 组合模式 |
| 动态添加功能 | 装饰器模式 |
| 简化复杂系统 | 外观模式 |
| 共享细粒度对象 | 享元模式 |
| 控制对象访问 | 代理模式 |

### 选择行为型模式

| 需求 | 推荐模式 |
|------|--------|
| 链式传递请求 | 责任链模式 |
| 将请求对象化 | 命令模式 |
| 遍历集合元素 | 迭代器模式 |
| 对象间通信 | 中介者模式 |
| 保存和恢复状态 | 备忘录模式 |
| 一对多通知 | 观察者模式 |
| 对象状态转换 | 状态模式 |
| 选择算法 | 策略模式 |
| 定义算法骨架 | 模板方法模式 |
| 为对象添加操作 | 访问者模式 |

## 整合多个设计模式

实际应用中通常需要结合多个设计模式：

```java
// 示例：订单处理系统

// 1. 单例模式：订单工厂
public enum OrderFactory {
    INSTANCE;
    
    public Order createOrder(OrderBuilder builder) {
        return builder.build();
    }
}

// 2. 建造者模式：构建复杂订单
public class OrderBuilder {
    private String orderId;
    private List<Item> items;
    
    public Order build() {
        return new Order(this);
    }
}

// 3. 策略模式：支付策略
public interface PaymentStrategy {
    void pay(double amount);
}

// 4. 观察者模式：订单状态变化通知
public class OrderManager {
    private List<OrderObserver> observers;
    
    public void changeStatus(Order order, String newStatus) {
        order.setStatus(newStatus);
        notifyObservers(order);
    }
}

// 5. 模板方法模式：订单处理流程
public abstract class OrderProcessor {
    public final void process(Order order) {
        validateOrder(order);
        prepareOrder(order);
        shipOrder(order);
        notifyCustomer(order);
    }
    
    protected abstract void prepareOrder(Order order);
}

// 6. 装饰器模式：订单增强
public class DiscountedOrder implements Order {
    private Order originalOrder;
    private double discountRate;
    
    public double getTotal() {
        return originalOrder.getTotal() * (1 - discountRate);
    }
}
```

## 设计模式学习路线

### 初级（理解基础）
1. 单例模式
2. 工厂方法模式
3. 策略模式
4. 观察者模式

### 中级（掌握应用）
5. 装饰器模式
6. 适配器模式
7. 建造者模式
8. 代理模式

### 高级（深入理解）
9. 抽象工厂模式
10. 模板方法模式
11. 责任链模式
12. 状态模式

## 最佳实践总结

1. **理解问题，不要盲目使用设计模式**
2. **优先选择简单方案，避免过度设计**
3. **学会重组和组合设计模式**
4. **遵循SOLID原则**
5. **定期重构代码，改进设计**
6. **通过代码审查学习他人的设计**
7. **在实际项目中不断实践和总结**
8. **阅读高质量开源项目的代码**
9. **与团队讨论和分享设计经验**
10. **持续学习和改进**

## 扩展阅读

- 《设计模式：可复用面向对象软件的基础》（Gamma等著）
- 《Head First 设计模式》
- 《重构：改善既有代码的设计》
- 《企业应用架构模式》
