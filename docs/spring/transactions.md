---
id: transactions
title: 事务管理
sidebar_label: 事务管理
sidebar_position: 6
---

# Spring 事务管理

> [!WARNING]
> **事务失效场景**: @Transactional 在类内部调用和 private 方法上会失效。理解事务传播行为和隔离级别至关重要，错误配置可能导致数据不一致。

### 1.1 事务的特性（ACID）

| 特性 | 说明 |
|------|------|
| **A (Atomicity)** | 原子性 - 事务中的所有操作要么全部成功，要么全部失败 |
| **C (Consistency)** | 一致性 - 事务执行前后数据必须保持一致 |
| **I (Isolation)** | 隔离性 - 并发事务之间互不影响 |
| **D (Durability)** | 持久性 - 事务成功后，数据永久保存 |

### 1.2 Spring事务管理方式

1. **编程式事务** - 手动编写代码管理事务
2. **声明式事务** - 使用注解或XML配置（推荐）

## 2. 编程式事务管理

### 2.1 使用TransactionTemplate

```java
@Service
public class UserService {
    
    @Autowired
    private TransactionTemplate transactionTemplate;
    
    @Autowired
    private UserRepository userRepository;
    
    public void createUser(User user) {
        transactionTemplate.execute(new TransactionCallback<Void>() {
            @Override
            public Void doInTransaction(TransactionStatus status) {
                try {
                    userRepository.save(user);
                    // 其他操作
                    return null;
                } catch (Exception ex) {
                    status.setRollbackOnly();  // 手动回滚
                    throw ex;
                }
            }
        });
    }
    
    // Lambda表达式简化
    public void createUserLambda(User user) {
        transactionTemplate.execute(status -> {
            userRepository.save(user);
            return null;
        });
    }
}
```

### 2.2 使用PlatformTransactionManager

```java
@Service
public class UserService {
    
    @Autowired
    private PlatformTransactionManager transactionManager;
    
    @Autowired
    private UserRepository userRepository;
    
    public void createUser(User user) {
        // 开始事务
        TransactionStatus status = transactionManager.getTransaction(
            new DefaultTransactionDefinition()
        );
        
        try {
            userRepository.save(user);
            // 提交事务
            transactionManager.commit(status);
        } catch (Exception ex) {
            // 回滚事务
            transactionManager.rollback(status);
            throw ex;
        }
    }
}
```

## 3. 声明式事务管理（推荐）

### 3.1 基本使用

```java
@Service
public class UserService {
    
    @Autowired
    private UserRepository userRepository;
    
    // 最简单的事务管理
    @Transactional
    public void saveUser(User user) {
        userRepository.save(user);
    }
}
```

### 3.2 配置

需要在配置类中启用事务管理：

```java
@Configuration
@EnableTransactionManagement  // 启用事务管理
public class AppConfig {
    
    @Bean
    public PlatformTransactionManager transactionManager(DataSource dataSource) {
        return new DataSourceTransactionManager(dataSource);
    }
}
```

或在XML配置中：

```xml
<tx:annotation-driven transaction-manager="transactionManager" />

<bean id="transactionManager" 
      class="org.springframework.jdbc.datasource.DataSourceTransactionManager">
    <property name="dataSource" ref="dataSource" />
</bean>
```

## 4. @Transactional详解

### 4.1 常用属性

```java
@Transactional(
    // 事务的传播行为
    propagation = Propagation.REQUIRED,
    
    // 事务的隔离级别
    isolation = Isolation.READ_COMMITTED,
    
    // 是否只读
    readOnly = false,
    
    // 超时时间（秒）
    timeout = 30,
    
    // 指定哪些异常需要回滚
    rollbackFor = Exception.class,
    
    // 指定哪些异常不需要回滚
    noRollbackFor = {IllegalArgumentException.class},
    
    // 指定使用的事务管理器
    value = "transactionManager"
)
public void saveUser(User user) {
    // 业务逻辑
}
```

### 4.2 传播行为（Propagation）

传播行为定义了事务方法调用另一个事务方法时如何处理事务。

| 传播行为 | 说明 | 场景 |
|---------|------|------|
| **REQUIRED** | 有则用，无则创建（默认） | 大多数业务方法 |
| **REQUIRES_NEW** | 总是创建新事务 | 需要独立事务的操作 |
| **NESTED** | 嵌套事务 | 子操作失败不影响主事务 |
| **SUPPORTS** | 有则用，无则不用 | 可选的事务操作 |
| **NOT_SUPPORTED** | 非事务执行 | 明确不需要事务 |
| **MANDATORY** | 必须有事务 | 必须在事务中调用 |
| **NEVER** | 必须无事务 | 不能在事务中调用 |

```java
@Service
public class UserService {
    
    @Autowired
    private UserRepository userRepository;
    
    @Autowired
    private LogService logService;
    
    // REQUIRED - 共享外层事务（默认）
    @Transactional(propagation = Propagation.REQUIRED)
    public void saveUser(User user) {
        userRepository.save(user);
        logService.log("User saved");  // 共享同一个事务
    }
    
    // REQUIRES_NEW - 创建新事务
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void logOperation(String operation) {
        // 即使外层事务回滚，日志也会被保存
    }
    
    // NESTED - 嵌套事务
    @Transactional(propagation = Propagation.NESTED)
    public void saveUserWithNested(User user) {
        userRepository.save(user);
        try {
            updateStatistics(user);
        } catch (Exception ex) {
            // 统计更新失败，但用户保存成功
        }
    }
}

@Service
public class LogService {
    
    // 如果在事务中调用，会使用外层事务
    // 如果不在事务中调用，则创建新事务
    @Transactional(propagation = Propagation.SUPPORTS)
    public void log(String message) {
        // 记录日志
    }
}
```

### 4.3 隔离级别（Isolation）

隔离级别定义了并发事务之间的隔离程度。

| 隔离级别 | 说明 | 脏读 | 不可重复读 | 幻读 | 性能 |
|---------|------|------|----------|------|------|
| **READ_UNCOMMITTED** | 读未提交 | ✓ | ✓ | ✓ | 最高 |
| **READ_COMMITTED** | 读提交 | ✗ | ✓ | ✓ | 高 |
| **REPEATABLE_READ** | 可重复读 | ✗ | ✗ | ✓ | 中 |
| **SERIALIZABLE** | 串行化 | ✗ | ✗ | ✗ | 最低 |

```java
@Service
public class UserService {
    
    // 最常用的隔离级别
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void saveUser(User user) {
        // 可以看到其他事务已提交的数据
        // 但不会看到未提交的数据
    }
    
    // 最严格的隔离级别，性能最低
    @Transactional(isolation = Isolation.SERIALIZABLE)
    public void criticalOperation() {
        // 完全隔离，不受其他事务影响
    }
}
```

### 4.4 只读事务

```java
// 只读事务（优化查询性能）
@Transactional(readOnly = true)
public User getUserById(Long id) {
    return userRepository.findById(id);
}

// 非只读事务（默认）
@Transactional(readOnly = false)
public void updateUser(User user) {
    userRepository.save(user);
}
```

### 4.5 超时设置

```java
// 设置超时时间为30秒
@Transactional(timeout = 30)
public void longRunningOperation() {
    // 如果超过30秒未完成，会抛出异常
}
```

### 4.6 异常处理

```java
@Service
public class UserService {
    
    @Autowired
    private UserRepository userRepository;
    
    // 所有异常都回滚
    @Transactional(rollbackFor = Exception.class)
    public void saveUser(User user) {
        userRepository.save(user);
        throw new RuntimeException("Failed");  // 会回滚
    }
    
    // 指定回滚异常
    @Transactional(rollbackFor = {IOException.class, SQLException.class})
    public void createUserWithFile(User user, File file) throws IOException {
        userRepository.save(user);
        // 如果抛出IOException或SQLException，则回滚
    }
    
    // 指定不回滚的异常
    @Transactional(noRollbackFor = {IllegalArgumentException.class})
    public void saveUserWithValidation(User user) {
        if (user.getAge() < 0) {
            throw new IllegalArgumentException("Invalid age");  // 不会回滚
        }
        userRepository.save(user);
    }
}
```

## 5. 事务的回滚

### 5.1 自动回滚

```java
@Service
public class UserService {
    
    @Autowired
    private UserRepository userRepository;
    
    @Transactional
    public void saveUser(User user) {
        userRepository.save(user);
        
        // ✅ RuntimeException自动回滚
        throw new RuntimeException("Error");
    }
    
    @Transactional(rollbackFor = Exception.class)
    public void saveUserWithChecked(User user) throws Exception {
        userRepository.save(user);
        
        // ✅ 检查异常也会回滚
        throw new Exception("Error");
    }
}
```

### 5.2 手动回滚

```java
@Service
public class UserService {
    
    @Autowired
    private UserRepository userRepository;
    
    @Transactional
    public void saveUserWithManualRollback(User user) {
        userRepository.save(user);
        
        // 获取事务状态
        TransactionInterceptor.currentTransactionStatus().setRollbackOnly();
        
        // 或者通过抛出异常
        throw new RuntimeException("Rollback");
    }
}
```

## 6. 常见问题

### 6.1 self-invocation问题

```java
@Service
public class UserService {
    
    @Transactional
    public void register(User user) {
        saveUser(user);  // ❌ saveUser的@Transactional失效
        notifyUser(user);
    }
    
    @Transactional
    public void saveUser(User user) {
        // 事务失效
    }
    
    // ✅ 解决方案1：注入自己
    @Autowired
    private UserService userService;
    
    public void registerFixed(User user) {
        userService.saveUser(user);  // 通过代理调用
    }
}
```

### 6.2 private方法不生效

```java
@Service
public class UserService {
    
    // ❌ private方法的@Transactional不生效
    @Transactional
    private void saveUserPrivate(User user) {
    }
    
    // ✅ public方法才能被拦截
    @Transactional
    public void saveUser(User user) {
    }
}
```

### 6.3 注解在接口上

```java
// ✅ 推荐：注解在实现类上
@Service
public class UserServiceImpl implements UserService {
    @Transactional
    public void save(User user) {
    }
}

// 或者在接口上（但不推荐）
public interface UserService {
    @Transactional
    void save(User user);
}
```

## 7. 实战示例

### 7.1 转账示例

```java
@Service
public class AccountService {
    
    @Autowired
    private AccountRepository accountRepository;
    
    @Transactional
    public void transfer(Long fromId, Long toId, BigDecimal amount) {
        // 获取账户
        Account fromAccount = accountRepository.findById(fromId);
        Account toAccount = accountRepository.findById(toId);
        
        // 验证余额
        if (fromAccount.getBalance().compareTo(amount) < 0) {
            throw new IllegalArgumentException("Insufficient balance");
        }
        
        // 执行转账
        fromAccount.setBalance(fromAccount.getBalance().subtract(amount));
        toAccount.setBalance(toAccount.getBalance().add(amount));
        
        // 更新账户
        accountRepository.save(fromAccount);
        accountRepository.save(toAccount);
        
        // 如果任何操作失败，整个事务回滚
    }
}
```

### 7.2 订单处理示例

```java
@Service
public class OrderService {
    
    @Autowired
    private OrderRepository orderRepository;
    
    @Autowired
    private InventoryService inventoryService;
    
    @Autowired
    private PaymentService paymentService;
    
    @Transactional(
        propagation = Propagation.REQUIRED,
        isolation = Isolation.READ_COMMITTED,
        rollbackFor = Exception.class
    )
    public void placeOrder(Order order) {
        // 1. 保存订单
        orderRepository.save(order);
        
        // 2. 扣减库存
        for (OrderItem item : order.getItems()) {
            inventoryService.decreaseStock(item.getProductId(), item.getQuantity());
        }
        
        // 3. 处理支付
        try {
            paymentService.processPayment(order);
        } catch (PaymentException ex) {
            // 支付失败，事务回滚
            throw new OrderException("Payment failed", ex);
        }
    }
}
```

## 8. 总结

| 特性 | 说明 |
|------|------|
| 传播行为 | 决定事务如何传递 |
| 隔离级别 | 决定并发隔离程度 |
| 只读事务 | 优化查询性能 |
| 超时时间 | 防止长时间占用资源 |
| 异常处理 | 控制何时回滚 |

---

**最佳实践**：

1. 使用声明式事务管理
2. 事务方法应该尽可能短
3. 避免在事务中执行长时间操作
4. 合理选择隔离级别和传播行为
5. 只读事务可以优化性能

**下一步**：学习[Spring MVC](/docs/spring/spring-mvc)
