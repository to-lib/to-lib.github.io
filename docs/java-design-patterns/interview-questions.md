---
sidebar_position: 100
title: 设计模式面试题精选
---

# 设计模式面试题精选

> [!TIP]
> 本文精选了常见的设计模式面试题，涵盖 23 种经典设计模式的核心概念、使用场景和实现要点。

## 🎯 设计模式概述

### 1. 什么是设计模式？为什么要使用设计模式？

**答案要点：**

**设计模式定义：**

- 针对软件设计中常见问题的可复用解决方案
- 前人经验的总结和最佳实践
- 提供了一套通用的术语和概念

**为什么使用：**

1. **提高代码复用性** - 避免重复造轮子
2. **增强代码可维护性** - 结构清晰，易于理解
3. **提升代码可扩展性** - 遵循开闭原则
4. **改善团队协作** - 统一的设计语言

**23 种经典设计模式分类：**

- **创建型（5 种）：** 对象创建机制
- **结构型（7 种）：** 类和对象的组合
- **行为型（11 种）：** 对象间的职责分配

**延伸：** 参考 [设计模式概览](/docs/java-design-patterns/overview)

---

### 2. 设计模式的六大原则是什么？

**答案要点：**

**SOLID 原则 + 其他：**

| 原则                | 说明                   | 应用               |
| ------------------- | ---------------------- | ------------------ |
| **单一职责（SRP）** | 一个类只负责一个职责   | 避免类过于臃肿     |
| **开闭原则（OCP）** | 对扩展开放，对修改关闭 | 策略模式、模板方法 |
| **里氏替换（LSP）** | 子类可以替换父类       | 继承体系设计       |
| **接口隔离（ISP）** | 接口应该小而专         | 避免接口污染       |
| **依赖倒置（DIP）** | 依赖抽象而非具体       | 依赖注入           |
| **迪米特法则**      | 最少知识原则           | 降低耦合           |

**示例：单一职责原则**

```java
// ✗ 违反SRP：一个类承担多个职责
class User {
    void login() { }
    void saveToDatabase() { }
    void sendEmail() { }
}

// ✓ 遵循SRP：职责分离
class User {
    void login() { }
}
class UserRepository {
    void save(User user) { }
}
class EmailService {
    void sendEmail(User user) { }
}
```

**延伸：** 参考 [最佳实践](/docs/java-design-patterns/best-practices)

---

## 🎯 创建型模式

### 3. 单例模式有哪些实现方式？各有什么优缺点？

**答案要点：**

**常见实现方式：**

**1. 饿汉式（类加载时创建）**

```java
public class Singleton {
    private static final Singleton INSTANCE = new Singleton();
    private Singleton() {}
    public static Singleton getInstance() { return INSTANCE; }
}
```

✓ 线程安全、简单  
✗ 可能浪费内存

**2. 懒汉式（双重检查锁）** ⭐ 推荐

```java
public class Singleton {
    private static volatile Singleton instance;
    private Singleton() {}

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

✓ 延迟加载、线程安全  
✗ 代码复杂

**3. 静态内部类** ⭐ 推荐

```java
public class Singleton {
    private Singleton() {}

    private static class Holder {
        private static final Singleton INSTANCE = new Singleton();
    }

    public static Singleton getInstance() {
        return Holder.INSTANCE;
    }
}
```

✓ 延迟加载、线程安全、优雅

**4. 枚举（最安全）** ⭐ 推荐

```java
public enum Singleton {
    INSTANCE;

    public void doSomething() { }
}
```

✓ 防止反射和序列化破坏

**延伸：** 参考 [单例模式详解](/docs/java-design-patterns/singleton-pattern)

---

### 4. 工厂模式、抽象工厂模式的区别？

**答案要点：**

**区别对比：**

| 特性 | 工厂方法模式         | 抽象工厂模式                 |
| ---- | -------------------- | ---------------------------- |
| 产品 | 单一产品族           | 多个产品族                   |
| 工厂 | 一个工厂方法         | 多个工厂方法                 |
| 扩展 | 添加新产品需要新工厂 | 添加新产品族需要修改所有工厂 |

**工厂方法模式：**

```java
// 产品接口
interface Product {
    void use();
}

// 具体产品
class ConcreteProductA implements Product {
    public void use() { System.out.println("Using A"); }
}

// 工厂接口
interface Factory {
    Product createProduct();
}

// 具体工厂
class FactoryA implements Factory {
    public Product createProduct() {
        return new ConcreteProductA();
    }
}
```

**抽象工厂模式：**

```java
// 产品族：Button和TextField
interface Button { }
interface TextField { }

// 具体产品
class WindowsButton implements Button { }
class WindowsTextField implements TextField { }

// 抽象工厂
interface GUIFactory {
    Button createButton();
    TextField createTextField();
}

// 具体工厂
class WindowsFactory implements GUIFactory {
    public Button createButton() { return new WindowsButton(); }
    public TextField createTextField() { return new WindowsTextField(); }
}
```

**应用场景：**

- **工厂方法：** 日志记录器（FileLogger, ConsoleLogger）
- **抽象工厂：** 跨平台 UI 组件（Windows, Mac）

**延伸：** 参考 [工厂模式](/docs/java-design-patterns/factory-pattern) 和 [抽象工厂模式](/docs/java-design-patterns/abstract-factory-pattern)

---

### 5. 建造者模式的使用场景？

**答案要点：**

**适用场景：**

- 对象创建过程复杂，包含多个步骤
- 对象有多个可选参数
- 需要创建不同表示的对象

**经典示例：**

```java
public class Computer {
    // 必需参数
    private String cpu;
    private String ram;

    // 可选参数
    private String gpu;
    private String storage;

    private Computer(Builder builder) {
        this.cpu = builder.cpu;
        this.ram = builder.ram;
        this.gpu = builder.gpu;
        this.storage = builder.storage;
    }

    public static class Builder {
        private String cpu;
        private String ram;
        private String gpu;
        private String storage;

        public Builder(String cpu, String ram) {
            this.cpu = cpu;
            this.ram = ram;
        }

        public Builder gpu(String gpu) {
            this.gpu = gpu;
            return this;
        }

        public Builder storage(String storage) {
            this.storage = storage;
            return this;
        }

        public Computer build() {
            return new Computer(this);
        }
    }
}

// 使用
Computer computer = new Computer.Builder("Intel i7", "16GB")
    .gpu("RTX 3080")
    .storage("1TB SSD")
    .build();
```

**实际应用：**

- `StringBuilder`：字符串构建
- `Lombok @Builder`：自动生成 Builder
- HTTP 请求构建器

**延伸：** 参考 [建造者模式](/docs/java-design-patterns/builder-pattern)

---

## 🎯 结构型模式

### 6. 代理模式有哪些类型？各有什么区别？

**答案要点：**

**三种代理类型：**

**1. 静态代理**

```java
interface Service {
    void request();
}

class RealService implements Service {
    public void request() { System.out.println("Real request"); }
}

class ProxyService implements Service {
    private RealService real = new RealService();

    public void request() {
        System.out.println("Before");
        real.request();
        System.out.println("After");
    }
}
```

✓ 简单直观  
✗ 每个接口都需要代理类

**2. JDK 动态代理（基于接口）**

```java
Service proxy = (Service) Proxy.newProxyInstance(
    Service.class.getClassLoader(),
    new Class[]{Service.class},
    (proxy, method, args) -> {
        System.out.println("Before");
        Object result = method.invoke(new RealService(), args);
        System.out.println("After");
        return result;
    }
);
```

✓ 动态生成  
✗ 只能代理接口

**3. CGLIB 动态代理（基于继承）**

```java
Enhancer enhancer = new Enhancer();
enhancer.setSuperclass(RealService.class);
enhancer.setCallback((MethodInterceptor) (obj, method, args, proxy) -> {
    System.out.println("Before");
    Object result = proxy.invokeSuper(obj, args);
    System.out.println("After");
    return result;
});
RealService proxy = (RealService) enhancer.create();
```

✓ 可以代理类  
✗ 不能代理 final 类和方法

**应用场景：**

- Spring AOP（JDK 代理 + CGLIB 代理）
- MyBatis 的 Mapper 接口
- RPC 框架的远程调用

**延伸：** 参考 [代理模式详解](/docs/java-design-patterns/proxy-pattern)

---

### 7. 装饰器模式和代理模式的区别？

**答案要点：**

**核心区别：**

| 特性     | 装饰器模式     | 代理模式         |
| -------- | -------------- | ---------------- |
| 目的     | 增强功能       | 控制访问         |
| 关注点   | 对象的功能     | 对象的访问       |
| 透明性   | 客户端知道装饰 | 客户端不知道代理 |
| 层层嵌套 | 可以多层装饰   | 通常一层代理     |

**装饰器模式示例：**

```java
// Java IO 就是装饰器模式的经典应用
InputStream in = new FileInputStream("file.txt");
in = new BufferedInputStream(in);      // 添加缓冲功能
in = new DataInputStream(in);          // 添加数据读取功能
// 层层装饰，增强功能
```

**代理模式示例：**

```java
// 权限控制代理
class AdminProxy implements Service {
    private Service target;

    public void request() {
        if (!checkPermission()) {
            throw new SecurityException("No permission");
        }
        target.request();  // 控制访问
    }
}
```

**延伸：** 参考 [装饰器模式](/docs/java-design-patterns/decorator-pattern) 和 [代理模式](/docs/java-design-patterns/proxy-pattern)

---

### 8. 适配器模式的使用场景？

**答案要点：**

**定义：** 将一个类的接口转换成客户端期望的另一个接口

**两种实现方式：**

**1. 类适配器（继承）**

```java
// 目标接口
interface Target {
    void request();
}

// 被适配者
class Adaptee {
    void specificRequest() { System.out.println("Specific request"); }
}

// 适配器
class Adapter extends Adaptee implements Target {
    public void request() {
        specificRequest();  // 调用父类方法
    }
}
```

**2. 对象适配器（组合）** ⭐ 推荐

```java
class Adapter implements Target {
    private Adaptee adaptee;

    public Adapter(Adaptee adaptee) {
        this.adaptee = adaptee;
    }

    public void request() {
        adaptee.specificRequest();  // 委托给adaptee
    }
}
```

**实际应用：**

- `Arrays.asList()`：数组到 List 的适配
- `InputStreamReader`：字节流到字符流的适配
- Spring MVC 的`HandlerAdapter`：不同 Controller 的适配

**延伸：** 参考 [适配器模式](/docs/java-design-patterns/adapter-pattern)

---

## 🎯 行为型模式

### 9. 策略模式的优缺点？如何消除 if-else？

**答案要点：**

**策略模式定义：** 定义一系列算法，封装每个算法，使它们可以互换

**优点：**

- 消除大量 if-else
- 符合开闭原则
- 算法可以自由切换

**缺点：**

- 策略类数量增多
- 客户端需要了解所有策略

**消除 if-else 示例：**

```java
// ✗ 传统if-else
public double calculate(String type, double price) {
    if ("VIP".equals(type)) {
        return price * 0.8;
    } else if ("SVIP".equals(type)) {
        return price * 0.7;
    } else {
        return price;
    }
}

// ✓ 策略模式
interface DiscountStrategy {
    double calculate(double price);
}

class VIPStrategy implements DiscountStrategy {
    public double calculate(double price) { return price * 0.8; }
}

class SVIPStrategy implements DiscountStrategy {
    public double calculate(double price) { return price * 0.7; }
}

// 使用Map消除if-else
Map<String, DiscountStrategy> strategies = new HashMap<>();
strategies.put("VIP", new VIPStrategy());
strategies.put("SVIP", new SVIPStrategy());

double result = strategies.get(userType).calculate(price);
```

**Spring 中的应用：**

```java
@Component("VIP")
class VIPStrategy implements DiscountStrategy { }

@Autowired
private Map<String, DiscountStrategy> strategyMap;

DiscountStrategy strategy = strategyMap.get(userType);
```

**延伸：** 参考 [策略模式详解](/docs/java-design-patterns/strategy-pattern)

---

### 10. 观察者模式的应用场景？

**答案要点：**

**定义：** 对象间一对多的依赖关系，一个对象状态改变，所有依赖者都会收到通知

**核心角色：**

- **Subject（主题）：** 维护观察者列表
- **Observer（观察者）：** 定义更新接口
- **ConcreteSubject：** 具体主题，状态变化时通知观察者
- **ConcreteObserver：** 具体观察者，实现更新逻辑

**实现示例：**

```java
// 观察者接口
interface Observer {
    void update(String message);
}

// 主题
class Subject {
    private List<Observer> observers = new ArrayList<>();

    public void attach(Observer observer) {
        observers.add(observer);
    }

    public void notifyObservers(String message) {
        for (Observer observer : observers) {
            observer.update(message);
        }
    }
}

// 具体观察者
class EmailObserver implements Observer {
    public void update(String message) {
        System.out.println("Email: " + message);
    }
}

// 使用
Subject subject = new Subject();
subject.attach(new EmailObserver());
subject.attach(new SMSObserver());
subject.notifyObservers("订单已发货");  // 所有观察者收到通知
```

**实际应用：**

- **Java**：`java.util.Observable` 和 `Observer`
- **Spring**：`ApplicationEvent` 和 `ApplicationListener`
- **GUI**：事件监听器
- **消息队列**：发布-订阅模式

**延伸：** 参考 [观察者模式](/docs/java-design-patterns/observer-pattern)

---

### 11. 模板方法模式和策略模式的区别？

**答案要点：**

**核心区别：**

| 特性     | 模板方法模式       | 策略模式       |
| -------- | ------------------ | -------------- |
| 控制流程 | 父类控制算法骨架   | 客户端选择算法 |
| 实现方式 | 继承               | 组合           |
| 扩展性   | 修改子类           | 切换策略对象   |
| 使用场景 | 固定流程，部分可变 | 整个算法可替换 |

**模板方法模式：**

```java
abstract class DataMiner {
    // 模板方法：定义算法骨架
    public final void mine() {
        openFile();
        extractData();    // 子类实现
        parseData();      // 子类实现
        analyzeData();    // 子类实现
        closeFile();
    }

    void openFile() { System.out.println("Open file"); }
    void closeFile() { System.out.println("Close file"); }

    // 钩子方法，子类实现
    abstract void extractData();
    abstract void parseData();
    abstract void analyzeData();
}

class CSVMiner extends DataMiner {
    void extractData() { System.out.println("Extract CSV"); }
    void parseData() { System.out.println("Parse CSV"); }
    void analyzeData() { System.out.println("Analyze CSV"); }
}
```

**对比策略模式：**

- 模板方法：定义做事的**流程**
- 策略模式：定义做事的**方法**

**实际应用：**

- **模板方法：** Spring 的`JdbcTemplate`, `RestTemplate`
- **策略模式：** 排序算法选择，支付方式选择

**延伸：** 参考 [模板方法模式](/docs/java-design-patterns/template-method-pattern) 和 [策略模式](/docs/java-design-patterns/strategy-pattern)

---

### 12. 责任链模式的应用场景？

**答案要点：**

**定义：** 多个对象都有机会处理请求，形成一条链，沿链传递请求直到被处理

**实现方式：**

```java
abstract class Handler {
    protected Handler next;

    public void setNext(Handler next) {
        this.next = next;
    }

    public abstract void handleRequest(Request request);
}

class ConcreteHandler1 extends Handler {
    public void handleRequest(Request request) {
        if (canHandle(request)) {
            // 处理请求
            System.out.println("Handler1 处理");
        } else if (next != null) {
            next.handleRequest(request);  // 传递给下一个
        }
    }
}

// 构建责任链
Handler h1 = new ConcreteHandler1();
Handler h2 = new ConcreteHandler2();
Handler h3 = new ConcreteHandler3();
h1.setNext(h2);
h2.setNext(h3);

// 发起请求
h1.handleRequest(request);
```

**实际应用：**

**1. Servlet Filter 链**

```java
public class MyFilter implements Filter {
    public void doFilter(ServletRequest request, ServletResponse response,
                        FilterChain chain) {
        // 前置处理
        chain.doFilter(request, response);  // 传递给下一个Filter
        // 后置处理
    }
}
```

**2. Spring Interceptor**

```java
public class MyInterceptor implements HandlerInterceptor {
    public boolean preHandle(HttpServletRequest request,
                            HttpServletResponse response, Object handler) {
        // 返回true继续链，false中断
        return true;
    }
}
```

**3. 日志级别处理**

```
Logger → ConsoleHandler → FileHandler → DatabaseHandler
```

**延伸：** 参考 [责任链模式](/docs/java-design-patterns/chain-of-responsibility-pattern)

---

## 📌 总结与建议

### 高频考点

1. **单例模式** - 各种实现方式的优缺点
2. **工厂模式** - 工厂方法 vs 抽象工厂
3. **代理模式** - 静态代理 vs 动态代理，与装饰器区别
4. **策略模式** - 消除 if-else，与模板方法区别
5. **观察者模式** - 发布订阅机制
6. **设计原则** - SOLID 原则的理解和应用

### 学习建议

1. **理解意图** - 每个模式解决什么问题
2. **掌握结构** - UML 类图和核心角色
3. **实践应用** - 在项目中识别和使用模式
4. **对比分析** - 相似模式的区别和选择

### 常见模式组合

- **工厂 + 单例** - 工厂本身是单例
- **策略 + 工厂** - 工厂创建策略对象
- **观察者 + 中介者** - 事件总线
- **模板方法 + 策略** - 固定流程，可变算法

### 相关资源

- [设计模式完整指南](/docs/java-design-patterns/index)
- [模式概览](/docs/java-design-patterns/overview)
- [最佳实践](/docs/java-design-patterns/best-practices)
- [模式对比分析](/docs/java-design-patterns/pattern-comparisons)
- [选择指南](/docs/java-design-patterns/selection-guide)

---

**持续更新中...** 欢迎反馈和补充！
