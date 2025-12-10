---
id: best-practices
title: Spring Cloud Alibaba 最佳实践
sidebar_label: 最佳实践
sidebar_position: 11
---

# Spring Cloud Alibaba 最佳实践

> [!IMPORTANT]
> **生产级应用指南**: 本文总结了 Spring Cloud Alibaba 在生产环境中的最佳实践,帮助你构建稳定、高效的微服务系统。

## 1. 服务拆分原则

### 单一职责原则

**每个服务专注于一个业务领域**:

```
❌ 不好的拆分:
- user-order-service (混合用户和订单)

✅ 好的拆分:
- user-service (用户管理)
- order-service (订单管理)
```

### 合理的粒度

**避免过度拆分**:

- **太粗**: 服务过大,难以维护
- **太细**: 服务过多,增加运维复杂度

**建议**: 3-7 人的团队维护一个服务

### 数据独立性

**每个服务拥有独立的数据库**:

```
✅ 正确:
user-service → user_db
order-service → order_db

❌ 错误:
user-service \
              → shared_db
order-service /
```

## 2. Nacos 最佳实践

### 命名空间隔离

**使用 Namespace 隔离不同环境**:

```yaml
# 开发环境
spring:
  cloud:
    nacos:
      discovery:
        namespace: dev
      config:
        namespace: dev

# 生产环境
spring:
  cloud:
    nacos:
      discovery:
        namespace: prod
      config:
        namespace: prod
```

### 配置分层管理

**三层配置结构**:

```
1. 公共配置 (shared-config)
   └── common.yaml (数据库、Redis 等)

2. 应用配置 (application-level)
   └── user-service.yaml

3. 环境配置 (profile-level)
   └── user-service-prod.yaml
```

**配置示例**:

```yaml
spring:
  cloud:
    nacos:
      config:
        # 共享配置
        shared-configs:
          - data-id: common.yaml
            group: COMMON_GROUP
            refresh: true
          - data-id: redis.yaml
            group: MIDDLEWARE_GROUP
            refresh: true
        
        # 扩展配置
        extension-configs:
          - data-id: mysql.yaml
            group: DATABASE_GROUP
            refresh: true
```

### 配置加密

**敏感信息加密**:

```yaml
# 在 Nacos 中加密存储
datasource:
  password: ENC(encrypted_password)
```

使用 Jasypt 加密:

```xml
<dependency>
    <groupId>com.github.ulisesbocchio</groupId>
    <artifactId>jasypt-spring-boot-starter</artifactId>
</dependency>
```

```yaml
jasypt:
  encryptor:
    password: ${JASYPT_PASSWORD}
```

### 服务分组

**使用 Group 隔离不同业务**:

```yaml
spring:
  cloud:
    nacos:
      discovery:
        group: PAYMENT_GROUP  # 支付业务组
```

### 元数据管理

**添加服务元数据**:

```yaml
spring:
  cloud:
    nacos:
      discovery:
        metadata:
          version: 1.0.0
          region: cn-hangzhou
          env: prod
          team: backend
```

## 3. Sentinel 最佳实践

### 资源定义规范

**使用有意义的资源名**:

```java
// ❌ 不好
@SentinelResource("api1")

// ✅ 好
@SentinelResource("user:getById")
```

### 限流规则配置

**针对不同场景选择策略**:

| 场景           | 策略       | 配置               |
| -------------- | ---------- | ------------------ |
| 突发流量       | Warm Up    | 预热时间 10s       |
| 削峰填谷       | 排队等待   | 超时时间 5s        |
| 普通场景       | 快速失败   | QPS 阈值           |
| 慢调用保护     | 并发线程数 | 线程数阈值 50      |

**示例配置**:

```java
// 预热模式 - 应对突发流量
@SentinelResource(
    value = "hot-api",
    blockHandler = "handleBlock"
)

// 在控制台配置:
// - 阈值类型: QPS
// - 阈值: 100
// - 流控效果: Warm Up
// - 预热时长: 10
```

### 熔断降级规则

**慢调用熔断**:

```yaml
# 超过 1 秒的调用视为慢调用
# 慢调用比例超过 50% 触发熔断
# 熔断时长 10 秒
```

**异常比例熔断**:

```yaml
# 异常比例超过 50% 触发熔断
# 最小请求数 10
# 熔断时长 10 秒
```

### 规则持久化

**推荐使用 Nacos 持久化**:

```yaml
spring:
  cloud:
    sentinel:
      datasource:
        # 流控规则
        flow:
          nacos:
            server-addr: localhost:8848
            dataId: ${spring.application.name}-flow-rules
            groupId: SENTINEL_GROUP
            rule-type: flow
        
        # 熔断规则
        degrade:
          nacos:
            server-addr: localhost:8848
            dataId: ${spring.application.name}-degrade-rules
            groupId: SENTINEL_GROUP
            rule-type: degrade
```

## 4. Seata 最佳实践

### 事务模式选择

| 场景               | 推荐模式 | 理由               |
| ------------------ | -------- | ------------------ |
| 普通业务           | AT       | 简单,无侵入        |
| 高性能要求         | TCC      | 性能好             |
| 长流程业务         | SAGA     | 适合长事务         |
| 强一致性要求       | XA       | 强一致             |

### AT 模式优化

**避免大事务**:

```java
// ❌ 不好 - 事务太大
@GlobalTransactional
public void createOrder() {
    // 100 行业务代码
    // ...
}

// ✅ 好 - 拆分事务
@GlobalTransactional
public void createOrder() {
    // 核心业务
    orderService.create();
    accountService.deduct();
    stockService.reduce();
}
```

### 幂等性设计

**关键操作必须幂等**:

```java
@GlobalTransactional
public void createOrder(Order order) {
    // 1. 检查幂等性
    if (orderRepository.existsByOrderNo(order.getOrderNo())) {
        return;
    }
    
    // 2. 创建订单
    orderRepository.save(order);
    
    // 3. 其他操作...
}
```

### 超时配置

```yaml
seata:
  client:
    tm:
      # 默认全局事务超时时间
      default-global-transaction-timeout: 60000
    rm:
      # 分支事务超时时间
      report-success-enable: true
```

## 5. RocketMQ 最佳实践

### 消息幂等

**使用唯一ID去重**:

```java
@RocketMQMessageListener(topic = "order-topic", consumerGroup = "order-group")
public class OrderConsumer implements RocketMQListener<Order> {

    @Autowired
    private RedisTemplate redisTemplate;

    @Override
    public void onMessage(Order order) {
        String msgId = order.getMsgId();
        
        // 检查是否已处理
        Boolean success = redisTemplate.opsForValue()
            .setIfAbsent(msgId, "1", 24, TimeUnit.HOURS);
        
        if (!success) {
            return;  // 已处理过
        }
        
        // 处理消息
        processOrder(order);
    }
}
```

### 消息可靠性

**生产者确认**:

```java
// 同步发送 - 确保消息发送成功
SendResult result = rocketMQTemplate.syncSend("topic", message);
if (result.getSendStatus() == SendStatus.SEND_OK) {
    // 发送成功
}
```

**消费者ACK**:

```java
// 消费成功自动ACK
// 消费失败抛异常自动重试
@Override
public void onMessage(Order order) {
    try {
        processOrder(order);
    } catch (Exception e) {
        throw new RuntimeException("处理失败,自动重试", e);
    }
}
```

### 消息重试和死信

**合理配置重试次数**:

```java
@RocketMQMessageListener(
    topic = "order-topic",
    consumerGroup = "order-group",
    maxReconsumeTimes = 3  // 最多重试 3 次
)
```

**处理死信队列**:

```java
@RocketMQMessageListener(
    topic = "%DLQ%order-group",  // 死信队列
    consumerGroup = "dlq-handler"
)
public class DLQConsumer implements RocketMQListener<Order> {
    @Override
    public void onMessage(Order order) {
        // 记录日志
        log.error("死信消息: {}", order);
        // 人工处理或告警
    }
}
```

## 6. Dubbo 最佳实践

### 接口设计

**遵循接口设计原则**:

```java
// ✅ 好的接口设计
public interface UserService {
    User getById(Long id);
    List<User> listByIds(List<Long> ids);
    PageResult<User> page(UserQuery query);
}

// ❌ 不好的接口设计
public interface UserService {
    Object doSomething(Map params);  // 参数不明确
}
```

### 超时配置

**合理设置超时时间**:

```java
@DubboReference(
    timeout = 3000,      // 调用超时 3 秒
    retries = 2,         // 重试 2 次
    loadbalance = "roundrobin"
)
private UserService userService;
```

### 异步调用

**高性能场景使用异步**:

```java
@DubboReference
private UserService userService;

public void demo() {
    // 异步调用
    CompletableFuture<User> future = 
        RpcContext.getContext().asyncCall(() -> 
            userService.getUser(1L)
        );
    
    future.thenAccept(user -> {
        System.out.println(user);
    });
}
```

## 7. 微服务通用最佳实践

### 统一异常处理

```java
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(BusinessException.class)
    public Result handleBusinessException(BusinessException e) {
        return Result.fail(e.getCode(), e.getMessage());
    }

    @ExceptionHandler(Exception.class)
    public Result handleException(Exception e) {
        log.error("系统异常", e);
        return Result.fail(500, "系统异常");
    }
}
```

### 统一返回格式

```java
public class Result<T> {
    private int code;
    private String message;
    private T data;
    
    public static <T> Result<T> success(T data) {
        return new Result<>(200, "success", data);
    }
    
    public static <T> Result<T> fail(int code, String message) {
        return new Result<>(code, message, null);
    }
}
```

### 日志规范

**使用结构化日志**:

```java
log.info("用户登录, userId={}, ip={}", userId, ip);

// ❌ 避免
log.info("用户" + userId + "从" + ip + "登录");
```

**日志级别使用**:

- **ERROR**: 影响业务的错误
- **WARN**: 潜在问题
- **INFO**: 重要业务流程
- **DEBUG**: 调试信息

### 健康检查

```java
@Component
public class CustomHealthIndicator implements HealthIndicator {

    @Override
    public Health health() {
        // 检查数据库连接
        if (checkDatabase()) {
            return Health.up().build();
        }
        return Health.down().withDetail("reason", "数据库连接失败").build();
    }
}
```

## 8. 性能优化

### 连接池配置

**数据库连接池 (HikariCP)**:

```yaml
spring:
  datasource:
    hikari:
      maximum-pool-size: 20
      minimum-idle: 5
      connection-timeout: 3000
      idle-timeout: 600000
      max-lifetime: 1800000
```

**Redis 连接池**:

```yaml
spring:
  redis:
    lettuce:
      pool:
        max-active: 20
        max-idle: 10
        min-idle: 5
        max-wait: 3000
```

### 缓存策略

**合理使用缓存**:

```java
@Cacheable(value = "users", key = "#id")
public User getUser(Long id) {
    return userRepository.findById(id);
}

@CacheEvict(value = "users", key = "#user.id")
public void updateUser(User user) {
    userRepository.save(user);
}
```

### 批量操作

**避免循环调用**:

```java
// ❌ 不好
for (Long id : ids) {
    userService.getById(id);
}

// ✅ 好
List<User> users = userService.listByIds(ids);
```

## 9. 安全最佳实践

### 接口鉴权

**使用 JWT**:

```java
@Component
public class JwtAuthFilter extends OncePerRequestFilter {

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                    HttpServletResponse response,
                                    FilterChain chain) {
        String token = request.getHeader("Authorization");
        
        if (token != null && JwtUtil.validate(token)) {
            // 设置认证信息
            SecurityContextHolder.getContext()
                .setAuthentication(JwtUtil.getAuthentication(token));
        }
        
        chain.doFilter(request, response);
    }
}
```

### 敏感信息脱敏

**日志脱敏**:

```java
public class MaskUtil {
    public static String maskPhone(String phone) {
        if (phone == null || phone.length() != 11) {
            return phone;
        }
        return phone.substring(0, 3) + "****" + phone.substring(7);
    }
}
```

### 限流防刷

**接口限流**:

```java
@SentinelResource(value = "login", blockHandler = "handleBlock")
@PostMapping("/login")
public Result login(@RequestBody LoginRequest request) {
    // 登录逻辑
}
```

## 10. 监控和运维

### 核心指标监控

**必须监控的指标**:

- **QPS/TPS**: 每秒请求/事务数
- **响应时间**: P50, P95, P99
- **错误率**: 4xx, 5xx 错误比例
- **服务可用性**: UP/DOWN 状态

### 告警配置

**关键告警**:

```yaml
alerts:
  - name: 错误率告警
    condition: error_rate > 1%
    duration: 5m
    
  - name: 响应时间告警
    condition: p95_latency > 1000ms
    duration: 5m
    
  - name: 服务下线告警
    condition: service_down
    duration: 1m
```

### 链路追踪

**集成 Sleuth + Zipkin**:

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-sleuth</artifactId>
</dependency>

<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-sleuth-zipkin</artifactId>
</dependency>
```

```yaml
spring:
  zipkin:
    base-url: http://localhost:9411
  sleuth:
    sampler:
      probability: 1.0  # 采样率 100%
```

## 11. 总结检查清单

### 服务设计 ✓

- [ ] 服务职责单一
- [ ] 数据库独立
- [ ] 接口设计合理

### 配置管理 ✓

- [ ] 使用 Namespace 隔离环境
- [ ] 配置分层管理
- [ ] 敏感信息加密

### 流量控制 ✓

- [ ] 配置限流规则
- [ ] 配置熔断规则
- [ ] 规则持久化到 Nacos

### 分布式事务 ✓

- [ ] 选择合适的事务模式
- [ ] 实现幂等性
- [ ] 配置超时时间

### 消息队列 ✓

- [ ] 消息幂等处理
- [ ] 配置重试策略
- [ ] 处理死信队列

### 性能优化 ✓

- [ ] 连接池配置合理
- [ ] 合理使用缓存
- [ ] 避免循环调用

### 安全 ✓

- [ ] 接口鉴权
- [ ] 敏感信息脱敏
- [ ] 接口限流

### 监控运维 ✓

- [ ] 配置健康检查
- [ ] 核心指标监控
- [ ] 告警配置
- [ ] 链路追踪

---

**关键要点**:

- 遵循最佳实践可以避免常见问题
- 生产环境必须做好监控和告警
- 安全和性能同样重要
- 持续优化和改进

**下一步**: 学习 [FAQ 常见问题](/docs/springcloud-alibaba/faq)
