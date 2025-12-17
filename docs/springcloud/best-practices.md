---
title: Spring Cloud 最佳实践
sidebar_label: 最佳实践
sidebar_position: 21
---

# Spring Cloud 最佳实践

> [!TIP]
> 本文总结了 Spring Cloud 微服务架构的最佳实践，帮助你构建高可用、可维护的微服务系统。

## 1. 服务拆分原则

### 拆分标准

- **单一职责**: 每个服务只负责一个业务领域
- **高内聚低耦合**: 服务内部紧密关联，服务之间松散耦合
- **数据自治**: 每个服务拥有自己的数据存储
- **独立部署**: 服务可以独立升级和扩展

### 拆分粒度

```
✅ 推荐
- 用户服务、订单服务、商品服务、支付服务

❌ 避免
- 过细：用户登录服务、用户注册服务、用户信息服务
- 过粗：电商服务（包含用户、订单、商品）
```

### 服务命名规范

```yaml
# 推荐：语义明确的服务名
spring:
  application:
    name: user-service      # ✅ 清晰
    name: order-service     # ✅ 清晰

# 避免：无意义的命名
    name: service1          # ❌ 不清晰
    name: app               # ❌ 不清晰
```

## 2. 服务注册发现

### Eureka 最佳实践

```yaml
eureka:
  client:
    # 生产环境：启用服务注册和发现
    register-with-eureka: true
    fetch-registry: true
    registry-fetch-interval-seconds: 30
  instance:
    # 使用 IP 注册，避免 DNS 问题
    prefer-ip-address: true
    # 启用健康检查
    health-check-url-path: /actuator/health
  server:
    # 生产环境：启用自我保护
    enable-self-preservation: true
```

### 高可用部署

- **至少 3 个 Eureka 节点**
- **跨可用区部署**
- **客户端配置多个 Server 地址**

## 3. 配置管理

### 敏感信息处理

```yaml
# ❌ 错误：明文密码
spring:
  datasource:
    password: my-secret-password

# ✅ 正确：加密或使用环境变量
spring:
  datasource:
    password: '{cipher}AQA8F1a2b3c4d5...'
    # 或
    password: ${DB_PASSWORD}
```

### 配置分层

```
config-repo/
├── application.yml           # 全局配置（日志、监控等）
├── application-dev.yml       # 开发环境全局配置
├── application-prod.yml      # 生产环境全局配置
├── user-service.yml          # 服务特有配置
└── user-service-prod.yml     # 服务+环境配置
```

### 配置刷新策略

| 配置类型   | 刷新策略                     |
| ---------- | ---------------------------- |
| 业务开关   | @RefreshScope + Bus 自动刷新 |
| 数据库连接 | 重启服务                     |
| 日志级别   | @RefreshScope 动态刷新       |
| 密钥       | 重启服务                     |

## 4. 服务调用

### Feign 最佳实践

```java
// ✅ 推荐：定义服务契约，设置超时和降级
@FeignClient(
    name = "user-service",
    fallbackFactory = UserClientFallbackFactory.class,
    configuration = FeignConfig.class
)
public interface UserClient {
    @GetMapping("/users/{id}")
    User getUser(@PathVariable Long id);
}

// 配置超时
@Configuration
public class FeignConfig {
    @Bean
    public Request.Options options() {
        return new Request.Options(
            5, TimeUnit.SECONDS,    // 连接超时
            10, TimeUnit.SECONDS,   // 读取超时
            true                    // 跟随重定向
        );
    }
}
```

### 避免循环依赖

```
❌ 避免
服务A → 服务B → 服务A

✅ 推荐
服务A → 服务B → 服务C
```

## 5. 容错设计

### 熔断器配置

```yaml
resilience4j:
  circuitbreaker:
    instances:
      default:
        # 滑动窗口大小
        slidingWindowSize: 10
        # 最小请求数
        minimumNumberOfCalls: 5
        # 失败率阈值
        failureRateThreshold: 50
        # 熔断持续时间
        waitDurationInOpenState: 10s
        # 半开状态请求数
        permittedNumberOfCallsInHalfOpenState: 3
```

### 降级策略

```java
// ✅ 提供有意义的降级响应
public User fallback(Long id, Throwable t) {
    log.warn("获取用户失败，使用默认值: {}", t.getMessage());
    return User.builder()
        .id(id)
        .name("未知用户")
        .status("UNKNOWN")
        .build();
}

// ❌ 避免返回 null 或空对象
public User badFallback(Long id, Throwable t) {
    return null;  // 可能导致 NPE
}
```

### 超时设置原则

```
上游超时 > 下游超时

Gateway (30s) > Service A (10s) > Service B (5s)
```

## 6. API 网关

### 路由配置

```yaml
spring:
  cloud:
    gateway:
      routes:
        - id: user-service
          uri: lb://user-service
          predicates:
            - Path=/api/users/**
          filters:
            # 移除路径前缀
            - StripPrefix=1
            # 添加请求头
            - AddRequestHeader=X-Gateway-Request, true
            # 限流
            - name: RequestRateLimiter
              args:
                redis-rate-limiter.replenishRate: 10
                redis-rate-limiter.burstCapacity: 20
```

### 统一错误处理

```java
@Component
public class GlobalErrorHandler implements ErrorWebExceptionHandler {
    @Override
    public Mono<Void> handle(ServerWebExchange exchange, Throwable ex) {
        // 统一错误响应格式
        ErrorResponse error = new ErrorResponse(
            HttpStatus.INTERNAL_SERVER_ERROR.value(),
            "服务暂时不可用"
        );
        // ...
    }
}
```

## 7. 链路追踪

### Sleuth 配置

```yaml
spring:
  sleuth:
    sampler:
      # 生产环境采样率
      probability: 0.1 # 10% 采样
    propagation-keys:
      - x-request-id
      - x-user-id
```

### 日志关联

```java
// 在日志中自动包含 traceId
log.info("处理订单: orderId={}", orderId);
// 输出: [order-service,abc123,def456] 处理订单: orderId=12345
```

## 8. 安全实践

### API 安全

```yaml
# 网关统一认证
spring:
  cloud:
    gateway:
      default-filters:
        - name: TokenRelay
```

```java
// 白名单配置
private static final List<String> WHITE_LIST = List.of(
    "/api/auth/login",
    "/api/auth/register",
    "/api/public/**"
);
```

### 服务间通信安全

- **内网隔离**: 服务之间在内网通信
- **mTLS**: 关键服务使用双向 TLS
- **服务认证**: 使用 JWT 或 OAuth2

## 9. 监控告警

### 关键指标

| 指标         | 告警阈值   |
| ------------ | ---------- |
| 服务可用性   | < 99.9%    |
| 平均响应时间 | > 500ms    |
| 错误率       | > 1%       |
| 熔断器打开   | 触发即告警 |
| CPU 使用率   | > 80%      |
| 内存使用率   | > 85%      |

### Actuator 配置

```yaml
management:
  endpoints:
    web:
      exposure:
        include: health,info,metrics,circuitbreakers
  endpoint:
    health:
      show-details: when_authorized
```

## 10. 部署实践

### 滚动发布

```yaml
# Kubernetes 配置
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 25%
      maxSurge: 25%
```

### 优雅停机

```yaml
server:
  shutdown: graceful

spring:
  lifecycle:
    timeout-per-shutdown-phase: 30s
```

### 健康检查

```yaml
# 就绪探针
readinessProbe:
  httpGet:
    path: /actuator/health/readiness
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10

# 存活探针
livenessProbe:
  httpGet:
    path: /actuator/health/liveness
    port: 8080
  initialDelaySeconds: 60
  periodSeconds: 30
```

---

## 速查清单

### 生产就绪检查

- [ ] 服务注册发现高可用（3+ 节点）
- [ ] 配置中心高可用
- [ ] 敏感配置已加密
- [ ] 所有服务配置熔断降级
- [ ] API 网关配置限流
- [ ] 链路追踪已启用
- [ ] 监控告警已配置
- [ ] 优雅停机已配置
- [ ] 健康检查端点已暴露

---

**相关文档**：

- [快速参考](/docs/springcloud/quick-reference)
- [常见问题](/docs/springcloud/faq)
- [面试题集](/docs/springcloud/interview-questions)
