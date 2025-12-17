---
title: Spring Cloud 常见问题
sidebar_label: 常见问题
sidebar_position: 22
---

# Spring Cloud 常见问题 (FAQ)

> [!TIP]
> 本页面整理了 Spring Cloud 开发中常见的问题及解决方案。

## 1. 服务注册发现

### Q: 服务注册后无法被发现？

**可能原因与解决方案**：

1. **服务名不一致**

```yaml
# 确保服务名配置正确
spring:
  application:
    name: user-service # 调用方使用这个名称
```

2. **未启用服务发现**

```java
// 确保启用了相关注解
@EnableEurekaClient  // 或 @EnableDiscoveryClient
```

3. **注册中心地址错误**

```yaml
eureka:
  client:
    service-url:
      defaultZone: http://localhost:8761/eureka/ # 注意末尾的 /eureka/
```

### Q: 服务下线后仍然被调用？

**原因**: 客户端缓存了服务列表

**解决方案**:

```yaml
eureka:
  client:
    # 缩短获取服务列表间隔
    registry-fetch-interval-seconds: 5
  instance:
    # 缩短心跳间隔和过期时间
    lease-renewal-interval-in-seconds: 10
    lease-expiration-duration-in-seconds: 30
```

### Q: Eureka 控制台显示 "EMERGENCY! EUREKA MAY BE INCORRECTLY..."？

**原因**: 自我保护模式触发

**开发环境解决方案**:

```yaml
eureka:
  server:
    enable-self-preservation: false # 仅开发环境
```

**生产环境**: 保持开启，检查网络连接和服务健康状态。

---

## 2. 配置中心

### Q: 配置无法加载，启动失败？

**解决方案**:

1. **确保使用 bootstrap.yml**

```yaml
# bootstrap.yml (不是 application.yml)
spring:
  cloud:
    config:
      uri: http://localhost:8888
```

2. **Config Server 不可用时允许降级**

```yaml
spring:
  cloud:
    config:
      fail-fast: false
```

3. **添加重试机制**

```xml
<dependency>
    <groupId>org.springframework.retry</groupId>
    <artifactId>spring-retry</artifactId>
</dependency>
```

### Q: @RefreshScope 不生效？

**检查项**:

1. 确保添加了 `spring-boot-starter-actuator` 依赖
2. 确保暴露了 refresh 端点

```yaml
management:
  endpoints:
    web:
      exposure:
        include: refresh
```

3. 调用刷新端点

```bash
curl -X POST http://localhost:8080/actuator/refresh
```

### Q: 配置刷新后部分配置未更新？

**原因**: 某些配置需要重启才能生效

**不支持动态刷新的配置**:

- 数据库连接池配置
- 端口配置
- 上下文路径

---

## 3. API 网关

### Q: Gateway 路由不生效？

**检查项**:

1. **路由 ID 唯一**

```yaml
spring:
  cloud:
    gateway:
      routes:
        - id: user-route # 必须唯一
          uri: lb://user-service
```

2. **断言语法正确**

```yaml
predicates:
  - Path=/api/users/** # 注意大小写和通配符
```

3. **检查服务是否注册**

```bash
curl http://localhost:8761/eureka/apps
```

### Q: Gateway 报 503 Service Unavailable？

**可能原因**:

1. **后端服务未启动**
2. **使用 lb:// 但未添加 LoadBalancer 依赖**

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-loadbalancer</artifactId>
</dependency>
```

3. **服务名称不匹配**

### Q: Gateway 出现跨域问题？

**解决方案**:

```yaml
spring:
  cloud:
    gateway:
      globalcors:
        cors-configurations:
          "[/**]":
            allowed-origins: "*"
            allowed-methods:
              - GET
              - POST
              - PUT
              - DELETE
            allowed-headers: "*"
```

---

## 4. 服务调用

### Q: Feign 调用超时？

**解决方案**:

```yaml
# 方式 1：全局配置
spring:
  cloud:
    openfeign:
      client:
        config:
          default:
            connectTimeout: 5000
            readTimeout: 10000

# 方式 2：针对特定服务
spring:
  cloud:
    openfeign:
      client:
        config:
          user-service:
            connectTimeout: 3000
            readTimeout: 5000
```

### Q: Feign 请求头丢失？

**解决方案**: 配置请求拦截器

```java
@Configuration
public class FeignConfig {
    @Bean
    public RequestInterceptor requestInterceptor() {
        return template -> {
            ServletRequestAttributes attrs = (ServletRequestAttributes)
                RequestContextHolder.getRequestAttributes();
            if (attrs != null) {
                HttpServletRequest request = attrs.getRequest();
                template.header("Authorization",
                    request.getHeader("Authorization"));
            }
        };
    }
}
```

### Q: 使用 @LoadBalanced 的 RestTemplate 调用失败？

**检查项**:

1. 确保使用服务名而非 IP

```java
// ✅ 正确
restTemplate.getForObject("http://user-service/users/1", User.class);

// ❌ 错误
restTemplate.getForObject("http://localhost:8081/users/1", User.class);
```

2. 确保添加了 LoadBalancer 依赖

---

## 5. 熔断与限流

### Q: 熔断器一直处于 OPEN 状态？

**解决方案**:

```yaml
resilience4j:
  circuitbreaker:
    instances:
      backendA:
        # 增加等待时间
        waitDurationInOpenState: 60s
        # 自动切换到半开
        automaticTransitionFromOpenToHalfOpenEnabled: true
```

**排查步骤**:

1. 检查 Actuator 端点: `/actuator/circuitbreakers`
2. 检查失败率是否超过阈值
3. 检查后端服务是否恢复

### Q: @CircuitBreaker 注解不生效？

**检查项**:

1. 添加 AOP 依赖

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-aop</artifactId>
</dependency>
```

2. 确保方法是 public 的
3. 确保不是在同一个类内部调用

### Q: 限流配置不生效？

**对于 Gateway + Redis 限流**:

```yaml
spring:
  redis:
    host: localhost
    port: 6379 # 确保 Redis 可用

  cloud:
    gateway:
      routes:
        - id: user-service
          filters:
            - name: RequestRateLimiter
              args:
                redis-rate-limiter.replenishRate: 10
                redis-rate-limiter.burstCapacity: 20
                key-resolver: "#{@ipKeyResolver}" # 确保 Bean 存在
```

---

## 6. 消息驱动

### Q: Stream 消费者收不到消息？

**检查项**:

1. **函数名与 binding 匹配**

```yaml
spring:
  cloud:
    stream:
      function:
        definition: process # 函数名
      bindings:
        process-in-0: # <函数名>-in-<索引>
          destination: my-topic
```

2. **消费者组配置**

```yaml
spring:
  cloud:
    stream:
      bindings:
        process-in-0:
          group: my-group # 设置消费者组
```

### Q: 消息消费失败如何重试？

**解决方案**:

```yaml
spring:
  cloud:
    stream:
      bindings:
        process-in-0:
          consumer:
            max-attempts: 3 # 最大重试次数
      rabbit:
        bindings:
          process-in-0:
            consumer:
              autoBindDlq: true # 启用死信队列
```

---

## 7. 链路追踪

### Q: TraceId 跨服务不一致？

**解决方案**:

1. 确保所有服务都添加了 Sleuth 依赖
2. 使用 Feign/RestTemplate 自动传播（确保使用 @LoadBalanced）
3. 如果使用 WebClient，需要注入配置好的实例

### Q: Zipkin 收集不到 Span？

**检查项**:

```yaml
spring:
  sleuth:
    sampler:
      probability: 1.0 # 开发环境设为 1.0 (100% 采样)
  zipkin:
    base-url: http://localhost:9411
    sender:
      type: web # 确保发送方式正确
```

---

## 8. 版本兼容

### Q: 升级 Spring Boot 后 Cloud 组件报错？

**解决方案**: 确保版本兼容

```xml
<!-- Spring Boot 3.2.x 使用 -->
<spring-cloud.version>2023.0.0</spring-cloud.version>

<!-- Spring Boot 2.7.x 使用 -->
<spring-cloud.version>2021.0.8</spring-cloud.version>
```

### Q: 找不到某些类或注解？

**常见变更**:

- Spring Boot 3.x 需要 Java 17+
- `@EnableEurekaClient` 可以省略
- Sleuth -> Micrometer Tracing (Spring Boot 3.x)

---

**相关文档**：

- [快速参考](/docs/springcloud/quick-reference)
- [最佳实践](/docs/springcloud/best-practices)
- [面试题集](/docs/springcloud/interview-questions)
