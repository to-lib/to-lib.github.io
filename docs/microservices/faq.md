---
sidebar_position: 9
title: 常见问题
description: 微服务常见问题解答 - FAQ
---

# 常见问题

## 架构设计

### 1. 什么时候应该使用微服务架构？

**适合使用微服务的场景：**
- 大型复杂应用，需要多团队协作
- 需要独立扩展不同业务模块
- 需要快速迭代和持续交付
- 团队具备 DevOps 能力

**不适合使用微服务的场景：**
- 小型简单应用
- 团队规模小，缺乏运维能力
- 业务边界不清晰
- 初创项目，需求不稳定

### 2. 如何确定服务的粒度？

服务粒度没有标准答案，但可以参考以下原则：

```
✅ 合适的粒度
- 2-3 人团队可以维护
- 可以在 2 周内重写
- 有明确的业务边界
- 可以独立部署和扩展

❌ 粒度过细的信号
- 服务间调用过于频繁
- 简单操作需要跨多个服务
- 分布式事务过于复杂

❌ 粒度过粗的信号
- 服务代码量过大
- 多个团队修改同一服务
- 部署频率受限
```

### 3. 如何从单体应用迁移到微服务？

推荐渐进式迁移策略：

```java
// 步骤 1: 识别边界 - 使用模块化重构单体
@Module("user")
public class UserModule {
    // 用户相关代码
}

@Module("order")
public class OrderModule {
    // 订单相关代码
}

// 步骤 2: 抽取服务 - 从边缘服务开始
// 先抽取依赖较少的服务，如通知服务、日志服务

// 步骤 3: 使用 Strangler Fig 模式
// 新功能用微服务实现，逐步替换旧功能
```

## 服务通信

### 4. REST 和 gRPC 如何选择？

| 场景 | 推荐 | 原因 |
| ---- | ---- | ---- |
| 对外公开 API | REST | 通用性好，浏览器支持 |
| 内部高频调用 | gRPC | 高性能，强类型 |
| 需要流式传输 | gRPC | 原生支持双向流 |
| 快速原型开发 | REST | 简单易用 |

### 5. 同步调用和异步消息如何选择？

```java
// 同步调用 - 需要立即响应
public Order createOrder(OrderDTO dto) {
    // 需要立即知道用户是否存在
    User user = userClient.getUser(dto.getUserId());
    if (user == null) {
        throw new UserNotFoundException();
    }
    return orderRepository.save(new Order(dto));
}

// 异步消息 - 可以延迟处理
public Order createOrder(OrderDTO dto) {
    Order order = orderRepository.save(new Order(dto));
    // 发送通知可以异步处理
    eventPublisher.publish(new OrderCreatedEvent(order));
    return order;
}
```

### 6. 如何处理服务调用超时？

```java
// 设置合理的超时时间
@FeignClient(name = "user-service", configuration = FeignConfig.class)
public interface UserClient {
    @GetMapping("/api/users/{id}")
    User getUser(@PathVariable Long id);
}

@Configuration
public class FeignConfig {
    @Bean
    public Request.Options options() {
        return new Request.Options(
            5, TimeUnit.SECONDS,   // 连接超时
            10, TimeUnit.SECONDS,  // 读取超时
            true
        );
    }
}

// 配合断路器使用
@CircuitBreaker(name = "userService", fallbackMethod = "fallback")
public User getUser(Long id) {
    return userClient.getUser(id);
}

public User fallback(Long id, Exception e) {
    log.warn("获取用户失败，使用降级数据: {}", e.getMessage());
    return new User(id, "Unknown", "降级用户");
}
```

## 数据管理

### 7. 如何处理分布式事务？

**方案对比：**

| 方案 | 一致性 | 性能 | 复杂度 | 适用场景 |
| ---- | ------ | ---- | ------ | ------- |
| 2PC | 强一致 | 低 | 高 | 金融核心 |
| Saga | 最终一致 | 高 | 中 | 电商订单 |
| 本地消息表 | 最终一致 | 高 | 低 | 通用场景 |
| 事务消息 | 最终一致 | 高 | 低 | 消息驱动 |

```java
// 本地消息表方案
@Transactional
public Order createOrder(OrderDTO dto) {
    // 1. 保存订单
    Order order = orderRepository.save(new Order(dto));
    
    // 2. 保存本地消息（同一事务）
    OutboxMessage message = new OutboxMessage(
        "ORDER_CREATED",
        JsonUtils.toJson(new OrderCreatedEvent(order))
    );
    outboxRepository.save(message);
    
    return order;
}

// 定时任务发送消息
@Scheduled(fixedDelay = 1000)
public void publishMessages() {
    List<OutboxMessage> messages = outboxRepository.findUnpublished();
    for (OutboxMessage msg : messages) {
        try {
            kafkaTemplate.send(msg.getTopic(), msg.getPayload());
            msg.setPublished(true);
            outboxRepository.save(msg);
        } catch (Exception e) {
            log.error("发送消息失败", e);
        }
    }
}
```

### 8. 如何实现跨服务数据查询？

```java
// 方案 1: API 组合
@Service
public class OrderQueryService {
    
    public OrderDetailDTO getOrderDetail(Long orderId) {
        Order order = orderClient.getOrder(orderId);
        User user = userClient.getUser(order.getUserId());
        List<Product> products = productClient.getProducts(order.getProductIds());
        
        return new OrderDetailDTO(order, user, products);
    }
}

// 方案 2: 数据冗余
@Entity
public class Order {
    private Long id;
    private Long userId;
    private String userName;  // 冗余用户名
    // ...
}

// 方案 3: CQRS 读模型
@Entity
@Table(name = "order_view")
public class OrderView {
    private Long orderId;
    private String userName;
    private String productNames;
    private BigDecimal totalAmount;
    // 预聚合的查询视图
}
```

### 9. 如何保证数据一致性？

```java
// 使用幂等性保证
@Service
public class PaymentService {
    
    public PaymentResult pay(String orderId, BigDecimal amount) {
        // 幂等性检查
        Payment existing = paymentRepository.findByOrderId(orderId);
        if (existing != null) {
            return new PaymentResult(existing);
        }
        
        // 执行支付
        Payment payment = new Payment(orderId, amount);
        payment = paymentRepository.save(payment);
        
        return new PaymentResult(payment);
    }
}

// 使用乐观锁
@Entity
public class Inventory {
    @Id
    private Long productId;
    private Integer quantity;
    
    @Version
    private Long version;
}
```

## 服务治理

### 10. 服务注册中心如何选择？

| 特性 | Nacos | Consul | Eureka |
| ---- | ----- | ------ | ------ |
| 一致性 | AP/CP 可切换 | CP | AP |
| 配置管理 | ✅ | ✅ | ❌ |
| 健康检查 | TCP/HTTP | TCP/HTTP/gRPC | 心跳 |
| 管理界面 | ✅ | ✅ | ✅ |
| 推荐场景 | 国内首选 | 多语言环境 | Spring Cloud |

### 11. 如何实现服务降级？

```java
@Service
public class ProductService {
    
    @Autowired
    private ProductClient productClient;
    
    // 方法级降级
    @SentinelResource(value = "getProduct", 
        fallback = "getProductFallback",
        blockHandler = "getProductBlockHandler")
    public Product getProduct(Long id) {
        return productClient.getProduct(id);
    }
    
    // 业务异常降级
    public Product getProductFallback(Long id, Throwable t) {
        log.warn("获取商品失败: {}", t.getMessage());
        return getDefaultProduct(id);
    }
    
    // 限流降级
    public Product getProductBlockHandler(Long id, BlockException e) {
        log.warn("请求被限流");
        return getDefaultProduct(id);
    }
    
    private Product getDefaultProduct(Long id) {
        return new Product(id, "商品信息暂不可用", BigDecimal.ZERO);
    }
}
```

### 12. 如何处理服务雪崩？

```java
// 1. 设置超时
@FeignClient(name = "user-service")
public interface UserClient {
    @GetMapping("/api/users/{id}")
    @Timeout(value = 3000)
    User getUser(@PathVariable Long id);
}

// 2. 使用断路器
@CircuitBreaker(name = "userService", fallbackMethod = "fallback")
public User getUser(Long id) {
    return userClient.getUser(id);
}

// 3. 限流保护
@RateLimiter(name = "userService")
public User getUser(Long id) {
    return userClient.getUser(id);
}

// 4. 舱壁隔离
@Bulkhead(name = "userService", type = Bulkhead.Type.THREADPOOL)
public User getUser(Long id) {
    return userClient.getUser(id);
}
```

## 部署运维

### 13. 如何实现零停机部署？

```yaml
# Kubernetes 滚动更新
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    spec:
      containers:
        - name: app
          readinessProbe:
            httpGet:
              path: /actuator/health/readiness
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 5
```

```java
// 优雅停机
@Component
public class GracefulShutdown implements ApplicationListener<ContextClosedEvent> {
    
    @Override
    public void onApplicationEvent(ContextClosedEvent event) {
        // 1. 从注册中心注销
        // 2. 等待正在处理的请求完成
        // 3. 关闭连接池
    }
}
```

### 14. 如何进行灰度发布？

```yaml
# Istio 灰度发布
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: user-service
spec:
  hosts:
    - user-service
  http:
    # 特定用户走新版本
    - match:
        - headers:
            x-user-id:
              regex: "^(1|2|3)$"
      route:
        - destination:
            host: user-service
            subset: v2
    # 其他用户走旧版本
    - route:
        - destination:
            host: user-service
            subset: v1
```

### 15. 如何排查分布式系统问题？

```java
// 1. 统一日志格式
{
    "timestamp": "2024-01-15T10:30:00.000Z",
    "traceId": "abc123",
    "spanId": "def456",
    "service": "order-service",
    "level": "ERROR",
    "message": "创建订单失败",
    "exception": "UserNotFoundException",
    "userId": 12345
}

// 2. 使用分布式追踪
// 在 Jaeger/Zipkin 中查看完整调用链

// 3. 查看指标
// Prometheus 查询: rate(http_requests_total{status="500"}[5m])

// 4. 检查健康状态
// GET /actuator/health
```

## 安全相关

### 16. 如何保护微服务 API？

```java
// 1. API 网关统一认证
@Component
public class AuthFilter implements GlobalFilter {
    
    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        String token = exchange.getRequest().getHeaders().getFirst("Authorization");
        if (!validateToken(token)) {
            return unauthorized(exchange);
        }
        return chain.filter(exchange);
    }
}

// 2. 服务间 mTLS
// 使用 Istio 自动管理证书

// 3. 接口权限控制
@PreAuthorize("hasRole('ADMIN')")
@GetMapping("/admin/users")
public List<User> getAllUsers() {
    return userService.findAll();
}
```

### 17. 如何安全存储敏感配置？

```yaml
# 使用 Kubernetes Secret
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
data:
  password: cGFzc3dvcmQ=  # base64 编码

# 使用 HashiCorp Vault
# 动态获取数据库凭证
```

```java
// 使用 Jasypt 加密配置
@Value("${db.password}")
private String dbPassword;  // ENC(加密后的密码)
```
