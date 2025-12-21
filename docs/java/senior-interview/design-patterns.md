---
sidebar_position: 8
title: 设计模式与代码设计
---

# 🎯 设计模式与代码设计（高级）

## 31. 如何在项目中正确使用设计模式？

**答案要点：**

**常用设计模式场景：**

| 模式 | 场景 | 框架应用 |
|------|------|---------|
| 单例 | 配置类、连接池 | Spring Bean |
| 工厂 | 对象创建解耦 | BeanFactory |
| 代理 | AOP、远程调用 | Spring AOP |
| 模板方法 | 算法骨架 | JdbcTemplate |
| 策略 | 算法切换 | Comparator |
| 观察者 | 事件通知 | ApplicationEvent |
| 责任链 | 请求处理链 | Filter、Interceptor |

**策略模式实战 - 支付方式：**

```java
// 1. 策略接口
public interface PaymentStrategy {
    PaymentResult pay(PaymentRequest request);
}

// 2. 具体策略
@Component("alipay")
public class AlipayStrategy implements PaymentStrategy {
    @Override
    public PaymentResult pay(PaymentRequest request) {
        // 支付宝支付逻辑
    }
}

@Component("wechat")
public class WechatPayStrategy implements PaymentStrategy {
    @Override
    public PaymentResult pay(PaymentRequest request) {
        // 微信支付逻辑
    }
}

// 3. 策略上下文
@Service
public class PaymentService {
    @Autowired
    private Map<String, PaymentStrategy> strategyMap;
    
    public PaymentResult pay(String payType, PaymentRequest request) {
        PaymentStrategy strategy = strategyMap.get(payType);
        if (strategy == null) {
            throw new IllegalArgumentException("不支持的支付方式");
        }
        return strategy.pay(request);
    }
}
```

**责任链模式实战 - 订单校验：**

```java
// 1. 处理器接口
public abstract class OrderValidator {
    protected OrderValidator next;
    
    public OrderValidator setNext(OrderValidator next) {
        this.next = next;
        return next;
    }
    
    public void validate(Order order) {
        doValidate(order);
        if (next != null) {
            next.validate(order);
        }
    }
    
    protected abstract void doValidate(Order order);
}

// 2. 具体处理器
public class StockValidator extends OrderValidator {
    @Override
    protected void doValidate(Order order) {
        if (!checkStock(order)) {
            throw new ValidationException("库存不足");
        }
    }
}

public class PriceValidator extends OrderValidator {
    @Override
    protected void doValidate(Order order) {
        if (!checkPrice(order)) {
            throw new ValidationException("价格异常");
        }
    }
}

// 3. 使用
OrderValidator chain = new StockValidator();
chain.setNext(new PriceValidator())
     .setNext(new UserValidator());
chain.validate(order);
```

**延伸：** 参考 [Java 设计模式](/docs/java-design-patterns)

---

## 32. 如何写出高质量的代码？有哪些原则？

**答案要点：**

**SOLID 原则：**

| 原则 | 说明 | 示例 |
|------|------|------|
| **S** 单一职责 | 一个类只做一件事 | UserService 只处理用户逻辑 |
| **O** 开闭原则 | 对扩展开放，对修改关闭 | 策略模式添加新策略 |
| **L** 里氏替换 | 子类可以替换父类 | 正方形不应继承长方形 |
| **I** 接口隔离 | 接口要小而专 | 拆分臃肿接口 |
| **D** 依赖倒置 | 依赖抽象而非实现 | 依赖注入 |

**代码规范示例：**

```java
// ❌ 不好的代码
public class OrderService {
    public void process(Order order) {
        // 50+ 行代码...
        // 校验
        if (order.getAmount() <= 0) throw new Exception("金额错误");
        if (order.getUserId() == null) throw new Exception("用户为空");
        
        // 计算价格
        double price = order.getAmount() * 0.9;
        if (order.isVip()) price = price * 0.95;
        
        // 保存
        orderDao.save(order);
        
        // 发送通知
        emailService.send(order.getUserEmail(), "订单创建成功");
        smsService.send(order.getUserPhone(), "订单创建成功");
    }
}

// ✅ 好的代码
@Service
@RequiredArgsConstructor
public class OrderService {
    private final OrderValidator validator;
    private final PriceCalculator priceCalculator;
    private final OrderRepository orderRepository;
    private final NotificationService notificationService;
    
    @Transactional
    public Order createOrder(CreateOrderRequest request) {
        // 1. 校验
        validator.validate(request);
        
        // 2. 计算价格
        BigDecimal price = priceCalculator.calculate(request);
        
        // 3. 创建订单
        Order order = Order.builder()
            .userId(request.getUserId())
            .amount(request.getAmount())
            .price(price)
            .status(OrderStatus.CREATED)
            .build();
        
        // 4. 保存
        order = orderRepository.save(order);
        
        // 5. 异步通知
        notificationService.notifyOrderCreated(order);
        
        return order;
    }
}
```

**延伸：** 参考 [Java 最佳实践](/docs/java/best-practices)

---

## 33. 如何进行代码重构？有哪些常见的坏味道？

**答案要点：**

**常见代码坏味道：**

| 坏味道 | 描述 | 重构方法 |
|--------|------|---------|
| 过长方法 | 方法超过50行 | 提取方法 |
| 过大类 | 类职责过多 | 拆分类 |
| 重复代码 | 相同逻辑多处出现 | 提取公共方法 |
| 过长参数列表 | 参数超过4个 | 引入参数对象 |
| 数据泥团 | 多个数据总是一起出现 | 提取类 |
| 基本类型偏执 | 过度使用基本类型 | 引入值对象 |

**重构示例 - 过长方法：**

```java
// ❌ 重构前
public void processOrder(Order order) {
    // 50+ 行代码...
    // 校验逻辑
    // 价格计算
    // 库存扣减
    // 订单保存
    // 消息发送
}

// ✅ 重构后
public void processOrder(Order order) {
    validateOrder(order);
    calculatePrice(order);
    deductInventory(order);
    saveOrder(order);
    sendNotification(order);
}

private void validateOrder(Order order) { /* ... */ }
private void calculatePrice(Order order) { /* ... */ }
private void deductInventory(Order order) { /* ... */ }
private void saveOrder(Order order) { /* ... */ }
private void sendNotification(Order order) { /* ... */ }
```

**重构示例 - 引入参数对象：**

```java
// ❌ 重构前
public User createUser(String name, String email, String phone, 
                       String address, Integer age, String gender) {
    // ...
}

// ✅ 重构后
public User createUser(CreateUserRequest request) {
    // ...
}

@Data
@Builder
public class CreateUserRequest {
    private String name;
    private String email;
    private String phone;
    private String address;
    private Integer age;
    private String gender;
}
```

**延伸：** 参考 [Java 最佳实践](/docs/java/best-practices)
