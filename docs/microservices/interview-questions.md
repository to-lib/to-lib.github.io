---
sidebar_position: 10
title: 面试题
description: 微服务面试题 - 基础概念、设计模式、实践经验
---

# 微服务面试题

## 基础概念（初级）

### 1. 什么是微服务架构？

**答案：**
微服务架构是一种将应用程序构建为一组小型、独立服务的架构风格。每个服务：
- 运行在自己的进程中
- 围绕业务能力组织
- 通过轻量级机制（通常是 HTTP API）通信
- 可以独立部署和扩展
- 可以使用不同的技术栈

### 2. 微服务和单体架构的区别？

**答案：**

| 特性 | 微服务 | 单体架构 |
| ---- | ------ | ------- |
| 部署 | 独立部署 | 整体部署 |
| 扩展 | 按需扩展 | 整体扩展 |
| 技术栈 | 多样化 | 统一 |
| 团队 | 小团队自治 | 大团队协作 |
| 复杂度 | 分布式复杂度 | 代码复杂度 |
| 故障隔离 | 服务级别 | 全局影响 |

### 3. 微服务有哪些优缺点？

**答案：**

**优点：**
- 独立部署和扩展
- 技术栈灵活
- 故障隔离
- 团队自治
- 易于持续交付

**缺点：**
- 分布式系统复杂度
- 网络延迟和故障
- 数据一致性挑战
- 运维成本高
- 测试复杂

### 4. 什么是服务注册与发现？

**答案：**
服务注册与发现是微服务架构的核心组件：

- **服务注册**：服务启动时向注册中心注册自己的地址信息
- **服务发现**：消费者从注册中心获取服务提供者的地址列表
- **健康检查**：注册中心定期检查服务健康状态，剔除不健康实例

常用实现：Nacos、Consul、Eureka、Zookeeper

### 5. 什么是 API 网关？

**答案：**
API 网关是微服务架构的入口点，主要功能：

- **请求路由**：将请求转发到对应服务
- **认证授权**：统一的身份验证
- **限流熔断**：保护后端服务
- **协议转换**：HTTP/gRPC/WebSocket
- **请求聚合**：合并多个服务响应
- **日志监控**：统一的访问日志

常用实现：Spring Cloud Gateway、Kong、Nginx

## 设计模式（中级）

### 6. 什么是断路器模式？如何实现？

**答案：**
断路器模式用于防止级联故障，有三种状态：

- **Closed**：正常状态，请求正常通过
- **Open**：熔断状态，请求快速失败
- **Half-Open**：半开状态，允许部分请求探测

```java
@CircuitBreaker(name = "userService", fallbackMethod = "fallback")
public User getUser(Long id) {
    return userClient.getUser(id);
}

public User fallback(Long id, Exception e) {
    return new User(id, "降级用户");
}
```

### 7. 什么是 Saga 模式？

**答案：**
Saga 模式用于管理分布式事务，将长事务拆分为多个本地事务：

**编排式（Choreography）：**
- 服务通过事件相互协调
- 无中心协调者
- 松耦合但难以追踪

**协调式（Orchestration）：**
- 中心协调者管理流程
- 易于追踪和管理
- 协调者可能成为瓶颈

每个本地事务都有对应的补偿操作，失败时执行补偿回滚。

### 8. 什么是 CQRS 模式？

**答案：**
CQRS（命令查询职责分离）将读操作和写操作分离：

- **命令模型**：处理写操作，保证数据一致性
- **查询模型**：处理读操作，优化查询性能

优点：
- 读写分离，独立扩展
- 查询模型可以针对特定场景优化
- 适合读写比例悬殊的场景

缺点：
- 复杂度增加
- 数据同步延迟

### 9. 什么是服务网格（Service Mesh）？

**答案：**
服务网格是处理服务间通信的基础设施层：

- **数据平面**：Sidecar 代理（如 Envoy）处理所有网络流量
- **控制平面**：管理配置和策略（如 Istio）

功能：
- 流量管理（路由、负载均衡）
- 安全（mTLS、认证授权）
- 可观测性（追踪、指标、日志）

### 10. 如何设计微服务的 API？

**答案：**

**RESTful 设计原则：**
```
GET    /api/users          # 获取列表
GET    /api/users/{id}     # 获取单个
POST   /api/users          # 创建
PUT    /api/users/{id}     # 更新
DELETE /api/users/{id}     # 删除
```

**最佳实践：**
- 使用名词复数表示资源
- 使用 HTTP 状态码表示结果
- 版本控制（/api/v1/users）
- 分页、过滤、排序
- HATEOAS（可选）

## 实践经验（高级）

### 11. 如何处理分布式事务？

**答案：**

**方案选择：**

| 方案 | 一致性 | 性能 | 适用场景 |
| ---- | ------ | ---- | ------- |
| 2PC | 强一致 | 低 | 金融核心 |
| Saga | 最终一致 | 高 | 电商订单 |
| 本地消息表 | 最终一致 | 高 | 通用场景 |
| 事务消息 | 最终一致 | 高 | 消息驱动 |

**本地消息表实现：**
```java
@Transactional
public Order createOrder(OrderDTO dto) {
    Order order = orderRepository.save(new Order(dto));
    // 同一事务保存消息
    outboxRepository.save(new OutboxMessage("ORDER_CREATED", order));
    return order;
}
```

### 12. 如何保证服务的高可用？

**答案：**

1. **多实例部署**：至少 2 个实例
2. **负载均衡**：分发请求到健康实例
3. **健康检查**：及时剔除不健康实例
4. **断路器**：防止级联故障
5. **限流**：保护服务不被压垮
6. **降级**：核心功能优先
7. **多机房部署**：容灾

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          averageUtilization: 70
```

### 13. 如何进行服务拆分？

**答案：**

**拆分原则：**
- 单一职责
- 高内聚低耦合
- 按业务领域划分（DDD）
- 团队自治

**拆分步骤：**
1. 识别业务领域和限界上下文
2. 定义服务边界和接口
3. 规划数据模型和存储
4. 渐进式拆分（Strangler Fig 模式）

**注意事项：**
- 避免过度拆分
- 考虑数据一致性
- 评估团队能力

### 14. 如何设计微服务的数据库？

**答案：**

**原则：**
- 每个服务独立数据库
- 数据通过 API 或事件同步
- 避免跨服务 JOIN

**数据同步方案：**
```java
// 方案 1: 事件驱动
@EventListener
public void handleUserUpdated(UserUpdatedEvent event) {
    // 更新本地冗余数据
    orderRepository.updateUserName(event.getUserId(), event.getName());
}

// 方案 2: API 调用
public OrderDTO getOrder(Long orderId) {
    Order order = orderRepository.findById(orderId);
    User user = userClient.getUser(order.getUserId());
    return new OrderDTO(order, user);
}
```

### 15. 如何实现微服务的可观测性？

**答案：**

**三大支柱：**

1. **日志（Logging）**
   - 结构化日志（JSON）
   - 统一日志格式
   - 集中收集（ELK）

2. **指标（Metrics）**
   - RED 指标：Rate、Errors、Duration
   - USE 指标：Utilization、Saturation、Errors
   - Prometheus + Grafana

3. **追踪（Tracing）**
   - 分布式追踪
   - TraceID 贯穿调用链
   - Jaeger/Zipkin

```java
// 统一日志格式
{
    "timestamp": "2024-01-15T10:30:00Z",
    "traceId": "abc123",
    "service": "order-service",
    "level": "INFO",
    "message": "订单创建成功"
}
```

### 16. 如何进行微服务测试？

**答案：**

**测试金字塔：**
- **单元测试**（大量）：测试单个组件
- **集成测试**（适量）：测试组件交互
- **端到端测试**（少量）：测试完整流程

**契约测试：**
```java
// 消费者定义契约
@Pact(consumer = "order-service", provider = "user-service")
public RequestResponsePact getUserPact(PactDslWithProvider builder) {
    return builder
        .given("用户存在")
        .uponReceiving("获取用户请求")
        .path("/api/users/1")
        .willRespondWith()
        .status(200)
        .body(new PactDslJsonBody().integerType("id", 1))
        .toPact();
}
```

### 17. 如何处理服务版本兼容？

**答案：**

**API 版本控制：**
```java
// URL 版本
@RequestMapping("/api/v1/users")
@RequestMapping("/api/v2/users")

// Header 版本
@GetMapping(headers = "X-API-Version=1")
```

**兼容性原则：**
- 新增字段向后兼容
- 删除字段先标记废弃
- 重大变更使用新版本
- 保持旧版本一段时间

### 18. 微服务架构中如何处理配置管理？

**答案：**

**配置中心方案：**
- Nacos Config
- Apollo
- Spring Cloud Config

**最佳实践：**
```yaml
# 分环境配置
spring:
  profiles:
    active: ${SPRING_PROFILES_ACTIVE:dev}
  cloud:
    nacos:
      config:
        server-addr: ${NACOS_SERVER:localhost:8848}
        namespace: ${NACOS_NAMESPACE:dev}
```

- 敏感配置加密
- 配置热更新
- 配置版本管理
- 配置审计

### 19. 如何设计微服务的安全架构？

**答案：**

**安全层次：**
1. **边界安全**：API 网关认证
2. **传输安全**：HTTPS/mTLS
3. **服务安全**：服务间认证
4. **数据安全**：加密存储

```java
// JWT 认证
@Component
public class JwtFilter implements GlobalFilter {
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        String token = extractToken(exchange);
        if (!validateToken(token)) {
            return unauthorized(exchange);
        }
        return chain.filter(exchange);
    }
}
```

### 20. 微服务架构的常见陷阱有哪些？

**答案：**

1. **分布式单体**：服务紧耦合，必须一起部署
2. **共享数据库**：多服务共享同一数据库
3. **过度拆分**：服务粒度过细
4. **忽略网络问题**：假设网络总是可靠
5. **缺乏监控**：无法追踪分布式请求
6. **数据一致性**：忽视最终一致性问题

**避免方法：**
- 异步解耦
- 数据库拆分
- 合理粒度
- 断路器和重试
- 完善可观测性
- 设计补偿机制
