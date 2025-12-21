---
title: Spring Cloud 面试题
sidebar_position: 23
---

# Spring Cloud 面试题集

> [!TIP]
> 本页面整理了 Spring Cloud 微服务架构相关的常见面试问题，帮助你准备技术面试。

## 1. 基础概念

### Q: 什么是微服务架构？与单体架构相比有什么优缺点？

**答案**：

微服务架构是将应用程序拆分为一组小型、自治的服务，每个服务独立运行、独立部署。

| 对比项     | 单体架构       | 微服务架构     |
| ---------- | -------------- | -------------- |
| 部署       | 整体部署       | 独立部署       |
| 扩展       | 整体扩展       | 按需扩展       |
| 技术栈     | 统一           | 可异构         |
| 故障隔离   | 差             | 好             |
| 开发效率   | 初期高，后期低 | 初期低，后期高 |
| 运维复杂度 | 低             | 高             |

---

### Q: Spring Cloud 的核心组件有哪些？分别解决什么问题？

**答案**：

| 组件              | 作用                 |
| ----------------- | -------------------- |
| **Eureka/Consul** | 服务注册与发现       |
| **Config**        | 分布式配置中心       |
| **Gateway**       | API 网关、路由、限流 |
| **OpenFeign**     | 声明式 HTTP 客户端   |
| **LoadBalancer**  | 客户端负载均衡       |
| **Resilience4j**  | 熔断、限流、重试     |
| **Sleuth**        | 链路追踪             |
| **Stream**        | 消息驱动             |

---

### Q: Spring Cloud 和 Spring Cloud Alibaba 有什么区别？

**答案**：

| 功能       | Spring Cloud  | Spring Cloud Alibaba |
| ---------- | ------------- | -------------------- |
| 服务发现   | Eureka/Consul | Nacos                |
| 配置中心   | Config        | Nacos                |
| 熔断限流   | Resilience4j  | Sentinel             |
| 分布式事务 | -             | Seata                |
| 消息       | Stream        | RocketMQ             |
| RPC        | OpenFeign     | Dubbo                |

Spring Cloud Alibaba 更适合国内生产环境，提供了开箱即用的全套解决方案。

---

## 2. 服务注册与发现

### Q: Eureka 的工作原理是什么？

**答案**：

1. **服务注册**: 服务启动时向 Eureka Server 注册自己的信息
2. **心跳续约**: 默认每 30 秒发送一次心跳
3. **服务发现**: 客户端定期拉取服务列表并缓存
4. **服务下线**: 主动调用 API 或超时（默认 90 秒）被剔除

**关键参数**:

- `lease-renewal-interval-in-seconds: 30` - 心跳间隔
- `lease-expiration-duration-in-seconds: 90` - 过期时间

---

### Q: 什么是 Eureka 的自我保护模式？

**答案**：

当 Eureka Server 在短时间内丢失过多客户端心跳时（默认 15 分钟内心跳比例低于 85%），会进入自我保护模式：

- **不再剔除**任何服务实例
- 仍然接受新服务注册
- 保护现有服务列表

**目的**: 防止因网络分区导致误删健康的服务。

---

### Q: CAP 理论中，Eureka 和 Zookeeper 分别满足哪些？

**答案**：

- **Eureka**: AP（可用性 + 分区容忍性）
  - 分区时仍可提供服务
  - 数据可能不一致（自我保护）
- **Zookeeper**: CP（一致性 + 分区容忍性）
  - 保证数据一致
  - Leader 选举期间不可用

微服务场景通常更倾向于 **AP**，因为服务发现的短暂不一致可以接受。

---

## 3. 配置中心

### Q: Spring Cloud Config 如何实现配置动态刷新？

**答案**：

**方式一：手动刷新**

1. 添加 `@RefreshScope` 注解
2. POST 请求 `/actuator/refresh`

**方式二：Spring Cloud Bus 自动刷新**

1. 集成消息队列（RabbitMQ/Kafka）
2. 配置变更后调用 `/actuator/bus-refresh`
3. 通过消息广播通知所有实例刷新

---

### Q: @RefreshScope 的原理是什么？

**答案**：

`@RefreshScope` 是一个自定义的 Bean 作用域：

1. 将 Bean 放入特殊的 **RefreshScope** 缓存
2. 收到刷新事件时，清除缓存中的 Bean
3. 下次访问时重新创建 Bean，获取最新配置

**注意**: `@RefreshScope` 使用了代理模式，可能影响性能。

---

## 4. 负载均衡

### Q: Ribbon 和 Spring Cloud LoadBalancer 有什么区别？

**答案**：

| 特性     | Ribbon   | LoadBalancer      |
| -------- | -------- | ----------------- |
| 状态     | 维护模式 | **活跃开发**      |
| 编程模型 | 阻塞     | 阻塞 + **响应式** |
| 依赖     | Netflix  | Spring 原生       |
| 配置     | 复杂     | 简单              |

**新项目推荐 LoadBalancer**。

---

### Q: LoadBalancer 支持哪些负载均衡策略？

**答案**：

**内置策略**:

- `RoundRobinLoadBalancer` - 轮询（默认）
- `RandomLoadBalancer` - 随机

**自定义策略**:

```java
@Bean
public ReactorLoadBalancer<ServiceInstance> customLoadBalancer(
        Environment environment,
        LoadBalancerClientFactory factory) {
    String name = environment.getProperty(
        LoadBalancerClientFactory.PROPERTY_NAME);
    return new RandomLoadBalancer(
        factory.getLazyProvider(name, ServiceInstanceListSupplier.class),
        name);
}
```

---

## 5. 熔断与限流

### Q: 什么是熔断？熔断器有哪些状态？

**答案**：

**熔断**: 当下游服务故障时，快速失败，避免级联故障。

**熔断器状态**:

1. **CLOSED（关闭）**: 正常调用，统计失败率
2. **OPEN（打开）**: 失败率超阈值，拒绝请求，直接返回降级结果
3. **HALF_OPEN（半开）**: 等待一段时间后，允许少量请求尝试恢复

```
CLOSED --失败率超阈值--> OPEN --等待时间到--> HALF_OPEN
  ↑                                            |
  +----------恢复正常/失败继续熔断--------------+
```

---

### Q: Hystrix 和 Resilience4j 有什么区别？

**答案**：

| 特性      | Hystrix      | Resilience4j   |
| --------- | ------------ | -------------- |
| 状态      | **停止维护** | 活跃开发       |
| Java 版本 | 6+           | 8+（函数式）   |
| 隔离方式  | 主要线程池   | 主要信号量     |
| 模块化    | 耦合         | **高度模块化** |
| 响应式    | RxJava 1     | Reactor        |

**新项目必须使用 Resilience4j**。

---

### Q: 服务降级和服务熔断的区别？

**答案**：

| 概念     | 服务降级             | 服务熔断     |
| -------- | -------------------- | ------------ |
| 触发方   | 主动                 | 被动         |
| 触发条件 | 资源紧张、保核心功能 | 故障率超阈值 |
| 作用     | 返回兜底数据         | 快速失败     |
| 范围     | 非核心功能           | 故障服务     |

**联系**: 熔断后通常会触发降级处理。

---

## 6. API 网关

### Q: Spring Cloud Gateway 的工作原理？

**答案**：

**核心组件**:

1. **Route（路由）**: 定义请求转发规则
2. **Predicate（断言）**: 匹配 HTTP 请求条件
3. **Filter（过滤器）**: 处理请求和响应

**请求流程**:

```
请求 --> Gateway Handler Mapping (匹配路由)
     --> Gateway Web Handler (执行过滤器链)
     --> 前置过滤器
     --> 代理请求到后端服务
     --> 后置过滤器
     --> 返回响应
```

---

### Q: Gateway 和 Zuul 有什么区别？

**答案**：

| 特性   | Gateway               | Zuul 1.x        |
| ------ | --------------------- | --------------- |
| 底层   | WebFlux（**响应式**） | Servlet（阻塞） |
| 性能   | 高                    | 中              |
| 限流   | 内置                  | 需自己实现      |
| 长连接 | 支持 WebSocket        | 不支持          |
| 状态   | 活跃                  | 停止维护        |

---

## 7. 链路追踪

### Q: 什么是分布式链路追踪？核心概念有哪些？

**答案**：

**定义**: 追踪请求在分布式系统中的完整调用链路。

**核心概念**:

- **Trace**: 一次完整请求的追踪记录（唯一 TraceId）
- **Span**: 一个工作单元（如一次服务调用）
- **Annotation**: 事件标记（cs、sr、ss、cr）

```
TraceId: abc123
├── SpanId: 001 (Gateway)
├── SpanId: 002 (OrderService)
│   └── SpanId: 003 (UserService)
└── SpanId: 004 (PaymentService)
```

---

### Q: Sleuth 如何实现跨服务追踪？

**答案**：

1. **自动注入**: 在请求中注入 TraceId 和 SpanId
2. **自动传播**: 通过 HTTP Header（如 `X-B3-TraceId`）传递
3. **自动采集**: 拦截 RestTemplate、Feign、WebClient 等

```
服务A                    服务B
[TraceId: abc, Span: 1]
         ---HTTP Header--->  [TraceId: abc, Span: 2, Parent: 1]
```

---

## 8. 综合问题

### Q: 如何设计一个高可用的微服务架构？

**答案**：

1. **服务注册发现**: 3+ 节点集群
2. **配置中心**: 高可用 + Git 版本控制
3. **API 网关**: 限流、熔断、认证
4. **服务调用**: 超时设置、重试机制
5. **容错设计**: 熔断降级、舱壁隔离
6. **链路追踪**: 全链路监控
7. **日志收集**: ELK/EFK
8. **优雅停机**: 无损发布

---

### Q: 微服务间如何保证数据一致性？

**答案**：

1. **最终一致性**（推荐）: 消息队列 + 事件驱动
2. **Saga 模式**: 本地事务 + 补偿机制
3. **分布式事务**: Seata（适合强一致性场景）

```
// 最终一致性示例
OrderService:
  1. 创建订单
  2. 发送 "订单已创建" 消息

InventoryService:
  1. 消费消息
  2. 扣减库存
  3. 失败则重试或补偿
```

---

### Q: 服务拆分的原则是什么？

**答案**：

1. **单一职责**: 一个服务只做一件事
2. **高内聚低耦合**: 服务内紧密，服务间松散
3. **数据自治**: 每个服务独立数据库
4. **独立部署**: 可单独升级
5. **业务边界**: 按领域划分（DDD）

---

**相关文档**：

- [快速参考](/docs/springcloud/quick-reference)
- [最佳实践](/docs/springcloud/best-practices)
- [常见问题](/docs/springcloud/faq)
