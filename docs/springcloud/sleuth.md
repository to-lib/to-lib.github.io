---
title: Sleuth 链路追踪
sidebar_label: Sleuth
sidebar_position: 9
---

# Sleuth 链路追踪

> [!TIP] > **分布式追踪**: Spring Cloud Sleuth 为分布式系统提供链路追踪功能，帮助理解服务间的调用关系和性能瓶颈。

## 1. Sleuth 简介

### 什么是 Sleuth？

**Spring Cloud Sleuth** 是 Spring Cloud 的分布式链路追踪解决方案，兼容 Zipkin、HTrace 等追踪系统。

### 解决的问题

```
用户请求 → Gateway → 订单服务 → 用户服务 → 数据库
                             ↓
                          库存服务 → 数据库
                             ↓
                          支付服务 → 第三方支付
```

- 如何追踪一个请求的完整链路？
- 哪个服务出现了性能问题？
- 服务间的依赖关系是什么？

### 核心概念

- **Trace** - 一次完整的请求链路，包含多个 Span
- **Span** - 一个服务调用，是 Trace 的基本单位
- **Trace ID** - 全局唯一的追踪 ID
- **Span ID** - 当前 Span 的 ID
- **Parent ID** - 父 Span 的 ID

## 2. 快速开始

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-sleuth</artifactId>
</dependency>
```

### 自动配置

Sleuth 会自动：

- 为每个请求生成 Trace ID
- 在日志中添加追踪信息
- 在服务间传递追踪信息

### 查看日志

```
[服务名,TraceId,SpanId,是否输出到追踪系统]
[order-service,f8a9c5d2b1e3a456,a1b2c3d4e5f6g7h8,true] 处理订单请求
```

- **服务名**: order-service
- **TraceId**: f8a9c5d2b1e3a456
- **SpanId**: a1b2c3d4e5f6g7h8
- **是否输出**: true

## 3. 集成 Zipkin

### 什么是 Zipkin？

**Zipkin** 是一个分布式追踪系统，用于收集、存储和展示追踪数据。

### 启动 Zipkin Server

```bash
# 下载 Zipkin
curl -sSL https://zipkin.io/quickstart.sh | bash -s

# 启动
java -jar zipkin.jar
```

访问：`http://localhost:9411`

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-zipkin</artifactId>
</dependency>
```

### 配置

```yaml
spring:
  application:
    name: order-service
  zipkin:
    # Zipkin 服务地址
    base-url: http://localhost:9411
    # 发送方式：web(HTTP) 或 rabbit(RabbitMQ)
    sender:
      type: web
  sleuth:
    sampler:
      # 采样率（0.0-1.0）
      probability: 1.0
```

## 4. 追踪信息传递

### HTTP 请求（自动）

Sleuth 会自动在 HTTP 请求头中添加追踪信息：

```
X-B3-TraceId: f8a9c5d2b1e3a456
X-B3-SpanId: a1b2c3d4e5f6g7h8
X-B3-ParentSpanId: 1234567890abcdef
X-B3-Sampled: 1
```

### 消息队列（RabbitMQ/Kafka）

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-sleuth-zipkin</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.amqp</groupId>
    <artifactId>spring-rabbit</artifactId>
</dependency>
```

Sleuth 会自动在消息头中传递追踪信息。

### 手动传递（异步线程）

```java
@Service
public class OrderService {

    @Autowired
    private Tracer tracer;

    @Autowired
    private ExecutorService executorService;

    public void processOrder() {
        // 获取当前 Span
        Span span = tracer.currentSpan();

        executorService.submit(() -> {
            // 在新线程中继续追踪
            try (Tracer.SpanInScope ws = tracer.withSpanInScope(span)) {
                // 业务逻辑
                System.out.println("处理订单");
            }
        });
    }
}
```

## 5. 自定义 Span

### 创建新 Span

```java
@Service
public class UserService {

    @Autowired
    private Tracer tracer;

    public User getUser(Long id) {
        // 创建新 Span
        Span span = tracer.nextSpan().name("getUser").start();

        try (Tracer.SpanInScope ws = tracer.withSpanInScope(span)) {
            // 添加标签
            span.tag("userId", String.valueOf(id));
            span.tag("method", "GET");

            // 添加事件
            span.event("查询数据库");

            // 业务逻辑
            User user = userRepository.findById(id);

            span.event("查询完成");

            return user;
        } finally {
            span.end();
        }
    }
}
```

### 使用注解

```java
@Service
public class OrderService {

    @NewSpan(name = "createOrder")
    public Order createOrder(OrderRequest request) {
        // 业务逻辑
        return new Order();
    }

    @ContinueSpan
    public void updateOrder(@SpanTag("orderId") Long id, Order order) {
        // 业务逻辑
    }
}
```

## 6. 采样策略

### 固定采样率

```yaml
spring:
  sleuth:
    sampler:
      # 采样率：1.0 表示 100%，0.1 表示 10%
      probability: 0.1
```

### 基于速率的采样

```yaml
spring:
  sleuth:
    sampler:
      # 每秒采样的请求数
      rate: 10
```

### 自定义采样器

```java
@Configuration
public class SamplerConfig {

    @Bean
    public Sampler customSampler() {
        return new Sampler() {
            @Override
            public boolean isSampled(long traceId) {
                // 自定义采样逻辑
                // 例如：只采样特定用户的请求
                return shouldSample();
            }

            private boolean shouldSample() {
                // 从请求上下文获取用户信息
                String userId = UserContext.getUserId();
                return "admin".equals(userId);
            }
        };
    }
}
```

## 7. 数据存储

### 内存存储（默认）

适合开发环境，重启后数据丢失。

### MySQL 存储

```yaml
# Zipkin Server 配置
storage:
  type: mysql
  mysql:
    host: localhost
    port: 3306
    username: root
    password: password
    db: zipkin
```

### Elasticsearch 存储

```yaml
# Zipkin Server 配置
storage:
  type: elasticsearch
  elasticsearch:
    hosts: http://localhost:9200
    index: zipkin
```

## 8. 与日志系统集成

### Logback 集成

```xml
<!-- logback-spring.xml -->
<configuration>
    <include resource="org/springframework/boot/logging/logback/defaults.xml"/>

    <springProperty scope="context" name="springAppName" source="spring.application.name"/>

    <appender name="CONSOLE" class="ch.qos.logback.core.ConsoleAppender">
        <encoder>
            <pattern>
                %d{yyyy-MM-dd HH:mm:ss.SSS} [${springAppName},%X{traceId},%X{spanId}] %-5level %logger{36} - %msg%n
            </pattern>
        </encoder>
    </appender>

    <root level="INFO">
        <appender-ref ref="CONSOLE"/>
    </root>
</configuration>
```

### MDC 上下文

```java
@Service
public class OrderService {

    private static final Logger log = LoggerFactory.getLogger(OrderService.class);

    public void createOrder() {
        // Sleuth 会自动将 traceId 和 spanId 放入 MDC
        log.info("创建订单"); // 日志会包含 traceId 和 spanId
    }
}
```

## 9. 性能分析

### 查找慢服务

通过 Zipkin UI 可以：

- 查看完整的调用链路
- 分析每个服务的耗时
- 找出性能瓶颈

### 依赖关系分析

Zipkin 的 Dependencies 页面展示：

- 服务间的调用关系
- 调用频率
- 错误率

## 10. 与其他追踪系统集成

### Brave

Sleuth 底层使用 Brave 实现追踪：

```java
@Autowired
private Tracer tracer;

@Autowired
private brave.Tracer braveTracer;
```

### OpenTelemetry

```xml
<dependency>
    <groupId>io.opentelemetry</groupId>
    <artifactId>opentelemetry-spring-boot-starter</artifactId>
</dependency>
```

### Jaeger

```xml
<dependency>
    <groupId>io.opentracing.contrib</groupId>
    <artifactId>opentracing-spring-jaeger-cloud-starter</artifactId>
</dependency>
```

## 11. 实战示例

### 完整链路追踪

```java
// Gateway
@RestController
public class GatewayController {
    @GetMapping("/api/orders/{id}")
    public Order getOrder(@PathVariable Long id) {
        // Sleuth 自动生成 TraceId
        return orderClient.getOrder(id);
    }
}

// Order Service
@Service
public class OrderService {

    @Autowired
    private UserClient userClient;

    @Autowired
    private InventoryClient inventoryClient;

    @NewSpan("getOrder")
    public Order getOrder(Long id) {
        Order order = orderRepository.findById(id);

        // 调用用户服务（自动关联到同一个 Trace）
        User user = userClient.getUser(order.getUserId());

        // 调用库存服务
        Inventory inventory = inventoryClient.getInventory(order.getProductId());

        return order;
    }
}
```

在 Zipkin 中可以看到完整的调用链：

```
Gateway (100ms)
  └─ Order Service (80ms)
       ├─ User Service (30ms)
       └─ Inventory Service (50ms)
```

## 12. 最佳实践

### 采样率配置

- **生产环境**: 0.01 - 0.1（1%-10%）
- **测试环境**: 0.5 - 1.0（50%-100%）
- **开发环境**: 1.0（100%）

### Span 命名

- 使用有意义的名称
- 包含操作类型：`getUserById`, `createOrder`
- 避免使用动态值：❌ `getUser_123`

### 标签使用

```java
span.tag("http.method", "GET");
span.tag("http.path", "/api/users/1");
span.tag("http.status_code", "200");
span.tag("userId", "123");
```

### 异常记录

```java
try {
    // 业务逻辑
} catch (Exception e) {
    span.tag("error", "true");
    span.tag("error.message", e.getMessage());
    throw e;
}
```

## 13. 监控与告警

### 关键指标

- **响应时间** - P50, P95, P99
- **错误率** - 各服务的错误率
- **调用量** - 各服务的 QPS
- **依赖关系** - 服务间的依赖图

### 告警规则

- 响应时间超过阈值
- 错误率超过阈值
- 服务不可用

## 14. 常见问题

### TraceId 丢失

**原因**:

- 异步线程未传递上下文
- 使用了不支持的 HTTP 客户端

**解决**:

- 使用 `tracer.withSpanInScope()`
- 使用 Sleuth 支持的客户端

### 性能影响

**优化**:

- 降低采样率
- 使用异步发送
- 使用消息队列传输数据

### 数据量过大

**解决**:

- 降低采样率
- 设置数据保留时间
- 使用 Elasticsearch 存储

## 15. 总结

| 功能     | 说明                 |
| -------- | -------------------- |
| 链路追踪 | 追踪请求的完整调用链 |
| 性能分析 | 分析服务响应时间     |
| 依赖关系 | 展示服务间的依赖     |
| 日志关联 | 将日志与追踪关联     |

---

**关键要点**：

- Sleuth 提供自动化的链路追踪
- 集成 Zipkin 可视化追踪数据
- 合理设置采样率平衡性能和监控
- 使用追踪数据分析性能瓶颈

**下一步**：探索 [Spring Cloud Alibaba](/docs/springcloud-alibaba/index)
