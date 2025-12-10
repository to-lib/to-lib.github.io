---
id: events
title: Spring 事件机制
sidebar_label: 事件机制
sidebar_position: 6
---

# Spring 事件机制

> [!IMPORTANT]
> **事件驱动架构**: Spring 事件机制实现组件间的松耦合通信。理解 ApplicationEvent 和 @EventListener 是构建事件驱动应用的基础。

## 1. 事件机制概述

**Spring 事件机制**是一种观察者模式的实现，允许应用组件之间进行松耦合的通信。

### 1.1 核心组件

| 组件 | 说明 |
|------|------|
| **ApplicationEvent** | 事件对象，携带事件数据 |
| **ApplicationListener** | 事件监听器接口 |
| **ApplicationEventPublisher** | 事件发布器 |
| **@EventListener** | 监听器注解（推荐） |
| **@Async** | 异步事件处理 |

### 1.2 事件流程

```
发布者 → ApplicationEventPublisher → Spring 容器 → 监听器
```

## 2. 基本事件使用

### 2.1 定义事件

```java
// 方式1：继承 ApplicationEvent
public class UserRegisteredEvent extends ApplicationEvent {
    private String username;
    private String email;
    
    public UserRegisteredEvent(Object source, String username, String email) {
        super(source);
        this.username = username;
        this.email = email;
    }
    
    public String getUsername() {
        return username;
    }
    
    public String getEmail() {
        return email;
    }
}

// 方式2：普通 POJO（Spring 4.2+）
public class OrderCreatedEvent {
    private Long orderId;
    private String orderNumber;
    private BigDecimal amount;
    
    public OrderCreatedEvent(Long orderId, String orderNumber, BigDecimal amount) {
        this.orderId = orderId;
        this.orderNumber = orderNumber;
        this.amount = amount;
    }
    
    // getters
}
```

### 2.2 发布事件

```java
@Service
public class UserService {
    
    @Autowired
    private ApplicationEventPublisher eventPublisher;
    
    public void registerUser(String username, String email) {
        // 业务逻辑
        User user = new User(username, email);
        // 保存用户...
        
        // 发布事件
        UserRegisteredEvent event = new UserRegisteredEvent(this, username, email);
        eventPublisher.publishEvent(event);
        
        // 或者发布 POJO 事件
        // eventPublisher.publishEvent(new OrderCreatedEvent(1L, "ORD001", new BigDecimal("100.00")));
    }
}
```

### 2.3 监听事件

#### 使用 @EventListener（推荐）

```java
@Component
public class UserEventListener {
    
    @EventListener
    public void handleUserRegistered(UserRegisteredEvent event) {
        System.out.println("用户注册成功: " + event.getUsername());
        System.out.println("邮箱: " + event.getEmail());
        
        // 发送欢迎邮件
        sendWelcomeEmail(event.getEmail());
    }
    
    private void sendWelcomeEmail(String email) {
        // 发送邮件逻辑
    }
}
```

#### 使用 ApplicationListener 接口

```java
@Component
public class UserRegisteredListener implements ApplicationListener<UserRegisteredEvent> {
    
    @Override
    public void onApplicationEvent(UserRegisteredEvent event) {
        System.out.println("用户注册: " + event.getUsername());
    }
}
```

## 3. 高级特性

### 3.1 条件监听

使用 SpEL 表达式进行条件过滤：

```java
@Component
public class OrderEventListener {
    
    // 只处理金额大于 1000 的订单
    @EventListener(condition = "#event.amount > 1000")
    public void handleLargeOrder(OrderCreatedEvent event) {
        System.out.println("大额订单: " + event.getOrderNumber());
        // 特殊处理逻辑
    }
    
    // 只处理特定状态的事件
    @EventListener(condition = "#event.status == 'COMPLETED'")
    public void handleCompletedOrder(OrderEvent event) {
        System.out.println("订单已完成: " + event.getOrderNumber());
    }
}
```

### 3.2 监听多个事件

```java
@Component
public class NotificationListener {
    
    // 监听多种事件类型
    @EventListener({UserRegisteredEvent.class, OrderCreatedEvent.class})
    public void handleMultipleEvents(Object event) {
        if (event instanceof UserRegisteredEvent) {
            UserRegisteredEvent userEvent = (UserRegisteredEvent) event;
            System.out.println("用户事件: " + userEvent.getUsername());
        } else if (event instanceof OrderCreatedEvent) {
            OrderCreatedEvent orderEvent = (OrderCreatedEvent) event;
            System.out.println("订单事件: " + orderEvent.getOrderNumber());
        }
    }
}
```

### 3.3 事件监听顺序

使用 @Order 注解控制监听器执行顺序：

```java
@Component
public class OrderedEventListeners {
    
    @EventListener
    @Order(1)
    public void firstListener(UserRegisteredEvent event) {
        System.out.println("第一个监听器");
    }
    
    @EventListener
    @Order(2)
    public void secondListener(UserRegisteredEvent event) {
        System.out.println("第二个监听器");
    }
    
    @EventListener
    @Order(3)
    public void thirdListener(UserRegisteredEvent event) {
        System.out.println("第三个监听器");
    }
}
```

## 4. 异步事件处理

### 4.1 启用异步支持

```java
@Configuration
@EnableAsync
public class AsyncConfig {
    
    @Bean
    public TaskExecutor taskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(5);
        executor.setMaxPoolSize(10);
        executor.setQueueCapacity(25);
        executor.setThreadNamePrefix("event-");
        executor.initialize();
        return executor;
    }
}
```

### 4.2 异步事件监听器

```java
@Component
public class AsyncEventListener {
    
    @Async
    @EventListener
    public void handleUserRegisteredAsync(UserRegisteredEvent event) {
        System.out.println("异步处理用户注册: " + event.getUsername());
        System.out.println("当前线程: " + Thread.currentThread().getName());
        
        // 耗时操作，如发送邮件
        try {
            Thread.sleep(2000);
            sendWelcomeEmail(event.getEmail());
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
    
    private void sendWelcomeEmail(String email) {
        System.out.println("发送欢迎邮件到: " + email);
    }
}
```

### 4.3 同步 vs 异步

```java
@Component
public class MixedEventListeners {
    
    // 同步监听器 - 阻塞发布者
    @EventListener
    public void syncListener(UserRegisteredEvent event) {
        System.out.println("同步处理: " + event.getUsername());
        // 如果这里很慢，会阻塞发布者
    }
    
    // 异步监听器 - 不阻塞发布者
    @Async
    @EventListener
    public void asyncListener(UserRegisteredEvent event) {
        System.out.println("异步处理: " + event.getUsername());
        // 在单独的线程中执行，不会阻塞发布者
    }
}
```

## 5. 事务事件

### 5.1 @TransactionalEventListener

在事务的特定阶段处理事件：

```java
@Component
public class TransactionalEventListener {
    
    // 事务提交后执行（默认）
    @TransactionalEventListener
    public void handleAfterCommit(OrderCreatedEvent event) {
        System.out.println("事务提交后处理订单: " + event.getOrderNumber());
        // 发送通知、更新缓存等
    }
    
    // 事务提交前执行
    @TransactionalEventListener(phase = TransactionPhase.BEFORE_COMMIT)
    public void handleBeforeCommit(OrderCreatedEvent event) {
        System.out.println("事务提交前检查");
    }
    
    // 事务回滚后执行
    @TransactionalEventListener(phase = TransactionPhase.AFTER_ROLLBACK)
    public void handleAfterRollback(OrderCreatedEvent event) {
        System.out.println("事务回滚，记录日志");
    }
    
    // 事务完成后执行（无论提交还是回滚）
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMPLETION)
    public void handleAfterCompletion(OrderCreatedEvent event) {
        System.out.println("事务完成");
    }
}
```

### 5.2 事务阶段说明

| 阶段 | 说明 | 使用场景 |
|------|------|----------|
| **AFTER_COMMIT** | 事务提交后（默认） | 发送通知、更新缓存 |
| **BEFORE_COMMIT** | 事务提交前 | 最后的验证检查 |
| **AFTER_ROLLBACK** | 事务回滚后 | 记录错误日志 |
| **AFTER_COMPLETION** | 事务完成后 | 清理资源 |

### 5.3 实际示例

```java
@Service
public class OrderService {
    
    @Autowired
    private ApplicationEventPublisher eventPublisher;
    
    @Autowired
    private OrderRepository orderRepository;
    
    @Transactional
    public void createOrder(Order order) {
        // 保存订单
        orderRepository.save(order);
        
        // 发布事件
        OrderCreatedEvent event = new OrderCreatedEvent(
            order.getId(), 
            order.getOrderNumber(), 
            order.getAmount()
        );
        eventPublisher.publishEvent(event);
        
        // 如果后续代码抛异常，事务会回滚
        // 但 @TransactionalEventListener 会在回滚后执行
    }
}

@Component
public class OrderEventHandler {
    
    @TransactionalEventListener
    public void sendOrderConfirmation(OrderCreatedEvent event) {
        // 只有在事务成功提交后才发送确认邮件
        System.out.println("发送订单确认邮件: " + event.getOrderNumber());
        emailService.sendOrderConfirmation(event);
    }
    
    @TransactionalEventListener(phase = TransactionPhase.AFTER_ROLLBACK)
    public void handleOrderFailure(OrderCreatedEvent event) {
        // 事务回滚后记录失败
        System.out.println("订单创建失败: " + event.getOrderNumber());
        logService.logFailure(event);
    }
}
```

## 6. 内置系统事件

Spring 提供了一些内置事件：

### 6.1 应用生命周期事件

```java
@Component
public class ApplicationLifecycleListener {
    
    // 应用启动完成
    @EventListener
    public void onApplicationReady(ApplicationReadyEvent event) {
        System.out.println("应用已启动，可以接收请求");
    }
    
    // 容器刷新
    @EventListener
    public void onContextRefreshed(ContextRefreshedEvent event) {
        System.out.println("Spring 容器已刷新");
    }
    
    // 容器启动
    @EventListener
    public void onContextStarted(ContextStartedEvent event) {
        System.out.println("Spring 容器已启动");
    }
    
    // 容器停止
    @EventListener
    public void onContextStopped(ContextStoppedEvent event) {
        System.out.println("Spring 容器已停止");
    }
    
    // 容器关闭
    @EventListener
    public void onContextClosed(ContextClosedEvent event) {
        System.out.println("Spring 容器已关闭");
    }
}
```

### 6.2 Web 应用事件

```java
@Component
public class WebApplicationListener {
    
    // Servlet 容器初始化
    @EventListener
    public void onServletContainerReady(ServletWebServerInitializedEvent event) {
        int port = event.getWebServer().getPort();
        System.out.println("Web 服务器启动在端口: " + port);
    }
}
```

## 7. 实际应用场景

### 7.1 用户注册流程

```java
// 事件定义
public class UserRegisteredEvent {
    private Long userId;
    private String username;
    private String email;
    
    public UserRegisteredEvent(Long userId, String username, String email) {
        this.userId = userId;
        this.username = username;
        this.email = email;
    }
    
    // getters
}

// 发布事件
@Service
public class UserService {
    
    @Autowired
    private ApplicationEventPublisher eventPublisher;
    
    @Transactional
    public User registerUser(String username, String email, String password) {
        // 1. 创建用户
        User user = new User(username, email, password);
        userRepository.save(user);
        
        // 2. 发布事件
        eventPublisher.publishEvent(
            new UserRegisteredEvent(user.getId(), username, email)
        );
        
        return user;
    }
}

// 多个监听器处理不同任务
@Component
public class UserRegistrationHandlers {
    
    @Autowired
    private EmailService emailService;
    
    @Autowired
    private CouponService couponService;
    
    @Autowired
    private AnalyticsService analyticsService;
    
    // 发送欢迎邮件
    @Async
    @TransactionalEventListener
    public void sendWelcomeEmail(UserRegisteredEvent event) {
        emailService.sendWelcomeEmail(event.getEmail());
    }
    
    // 赠送新人优惠券
    @TransactionalEventListener
    public void grantNewUserCoupon(UserRegisteredEvent event) {
        couponService.grantCoupon(event.getUserId(), "NEWUSER100");
    }
    
    // 记录用户行为分析
    @Async
    @TransactionalEventListener
    public void trackRegistration(UserRegisteredEvent event) {
        analyticsService.trackEvent("user_registered", event.getUserId());
    }
}
```

### 7.2 订单状态变更

```java
// 订单状态变更事件
public class OrderStatusChangedEvent {
    private Long orderId;
    private OrderStatus oldStatus;
    private OrderStatus newStatus;
    
    public OrderStatusChangedEvent(Long orderId, OrderStatus oldStatus, OrderStatus newStatus) {
        this.orderId = orderId;
        this.oldStatus = oldStatus;
        this.newStatus = newStatus;
    }
    
    // getters
}

// 订单服务
@Service
public class OrderService {
    
    @Autowired
    private ApplicationEventPublisher eventPublisher;
    
    @Transactional
    public void changeOrderStatus(Long orderId, OrderStatus newStatus) {
        Order order = orderRepository.findById(orderId).orElseThrow();
        OrderStatus oldStatus = order.getStatus();
        
        order.setStatus(newStatus);
        orderRepository.save(order);
        
        // 发布状态变更事件
        eventPublisher.publishEvent(
            new OrderStatusChangedEvent(orderId, oldStatus, newStatus)
        );
    }
}

// 处理不同状态的监听器
@Component
public class OrderStatusHandlers {
    
    // 订单完成时发送确认
    @TransactionalEventListener(condition = "#event.newStatus.name() == 'COMPLETED'")
    public void handleOrderCompleted(OrderStatusChangedEvent event) {
        System.out.println("订单已完成: " + event.getOrderId());
        notificationService.sendOrderCompletedNotification(event.getOrderId());
    }
    
    // 订单取消时退款
    @TransactionalEventListener(condition = "#event.newStatus.name() == 'CANCELLED'")
    public void handleOrderCancelled(OrderStatusChangedEvent event) {
        System.out.println("订单已取消: " + event.getOrderId());
        refundService.processRefund(event.getOrderId());
    }
}
```

## 8. 最佳实践

### 8.1 事件粒度

```java
// ✅ 推荐：细粒度事件
public class UserEmailChangedEvent { }
public class UserPasswordChangedEvent { }
public class UserProfileUpdatedEvent { }

// ❌ 避免：粗粒度事件
public class UserUpdatedEvent { 
    // 包含所有更新类型，难以区分
}
```

### 8.2 事件不可变性

```java
// ✅ 推荐：使用 final 字段，保证不可变
public class OrderCreatedEvent {
    private final Long orderId;
    private final String orderNumber;
    
    public OrderCreatedEvent(Long orderId, String orderNumber) {
        this.orderId = orderId;
        this.orderNumber = orderNumber;
    }
    
    public Long getOrderId() {
        return orderId;
    }
    
    public String getOrderNumber() {
        return orderNumber;
    }
}
```

### 8.3 异常处理

```java
@Component
public class SafeEventListener {
    
    @EventListener
    public void handleEvent(UserRegisteredEvent event) {
        try {
            // 业务逻辑
            processEvent(event);
        } catch (Exception e) {
            // 记录异常，不影响其他监听器
            log.error("处理事件失败: " + event.getUsername(), e);
        }
    }
}
```

### 8.4 避免循环事件

```java
// ❌ 避免：事件监听器中发布相同类型的事件
@EventListener
public void handleUserEvent(UserEvent event) {
    // 危险：可能导致无限循环
    eventPublisher.publishEvent(new UserEvent(...));
}

// ✅ 推荐：使用不同类型的事件
@EventListener
public void handleUserCreated(UserCreatedEvent event) {
    // 发布不同类型的事件
    eventPublisher.publishEvent(new UserWelcomeEvent(...));
}
```

## 9. 总结

| 特性 | 说明 | 使用场景 |
|------|------|----------|
| @EventListener | 事件监听注解 | 简单事件处理 |
| @Async | 异步处理 | 耗时操作 |
| @TransactionalEventListener | 事务事件 | 需要事务保证 |
| condition | 条件监听 | 过滤特定事件 |
| @Order | 执行顺序 | 控制监听器顺序 |

---

**关键要点**：

- 事件机制实现组件解耦
- 使用 @EventListener 简化监听器
- 耗时操作使用 @Async 异步处理
- 事务相关用 @TransactionalEventListener
- 保持事件对象不可变

**下一步**：学习 [AOP 详解](/docs/spring/aop)
