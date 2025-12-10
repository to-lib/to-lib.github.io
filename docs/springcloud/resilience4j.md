---
title: Resilience4j 容错
sidebar_label: Resilience4j
sidebar_position: 8.1
---

# Resilience4j 容错保护

> [!TIP]
> **Hystrix 的继任者**: Resilience4j 是一个轻量级、专为 Java 8+ 和函数式编程设计的容错库。它是 Spring Cloud Circuit Breaker 官方推荐的 Hystrix 替代方案。

## 1. 简介

Resilience4j 受 Netflix Hystrix 启发，但专为 Java 8 和函数式编程设计。它更加轻量，模块化程度更高。你可以只选择你需要的部分（如只用熔断器，不用限流）。

### 核心模块

- **CircuitBreaker**: 熔断器
- **RateLimiter**: 限流器
- **Retry**: 自动重试
- **Bulkhead**: 舱壁隔离 (并发限制)
- **TimeLimiter**: 超时控制
- **Cache**: 结果缓存

## 2. 快速开始

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-circuitbreaker-resilience4j</artifactId>
</dependency>
<!-- 如果需要监控，添加 actuator -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-aop</artifactId>
</dependency>
```

### 使用注解

Resilience4j 提供了丰富的注解支持。

```java
@Service
public class OrderService {

    @CircuitBreaker(name = "backendA", fallbackMethod = "fallback")
    @RateLimiter(name = "backendA")
    @Bulkhead(name = "backendA")
    public String doSomething() {
        // 可能会失败的调用
        return restTemplate.getForObject("http://backend-service/api", String.class);
    }

    public String fallback(Throwable t) {
        return "服务暂时不可用: " + t.getMessage();
    }
}
```

## 3. 配置详解 (application.yml)

### CircuitBreaker (熔断器)

```yaml
resilience4j.circuitbreaker:
  instances:
    backendA:
      registerHealthIndicator: true
      # 滑动窗口类型：COUNT_BASED (基于计数) 或 TIME_BASED (基于时间)
      slidingWindowType: COUNT_BASED
      # 滑动窗口大小
      slidingWindowSize: 10
      # 最小请求数，少于该值不会触发熔断
      minimumNumberOfCalls: 5
      # 失败率阈值 (百分比)，超过该值触发熔断
      failureRateThreshold: 50
      # 慢调用比例阈值
      slowCallRateThreshold: 100
      # 慢调用时间阈值
      slowCallDurationThreshold: 2000ms
      # 半开状态允许的请求数
      permittedNumberOfCallsInHalfOpenState: 3
      # 熔断器打开状态等待时间 (在此期间拒绝请求)
      waitDurationInOpenState: 5s
      # 自动从 Open 切换到 Half-Open
      automaticTransitionFromOpenToHalfOpenEnabled: true
```

### RateLimiter (限流器)

```yaml
resilience4j.ratelimiter:
  instances:
    backendA:
      # 限制刷新周期
      limitRefreshPeriod: 1s
      # 周期内允许的最大请求数
      limitForPeriod: 10
      # 请求等待超时时间
      timeoutDuration: 500ms
```

### Retry (重试)

```yaml
resilience4j.retry:
  instances:
    backendA:
      # 最大重试次数
      maxAttempts: 3
      # 重试间隔
      waitDuration: 500ms
      # 指数退避 (重试间隔逐渐增加)
      enableExponentialBackoff: true
      exponentialBackoffMultiplier: 2
```

### Bulkhead (舱壁隔离)

Resilience4j 提供两种舱壁隔离：

1. **SemaphoreBulkhead** (信号量，默认): 限制并发请求数。
2. **ThreadPoolBulkhead** (线程池): 使用独立线程池执行。

```yaml
resilience4j.bulkhead:
  instances:
    backendA:
      # 最大并发调用数
      maxConcurrentCalls: 10
      # 当由于并发限制而被阻塞时的最大等待时间
      maxWaitDuration: 0
```

## 4. 动态配置

Resilience4j 支持通过 Spring Cloud Config 动态刷新配置。

## 5. 监控

Resilience4j 通过 Micrometer 暴露指标，可以与 Prometheus + Grafana 完美集成。

访问 Actuator 端点：`/actuator/metrics/resilience4j.circuitbreaker.calls`

## 6. Resilience4j vs Hystrix

| 特性 | Hystrix | Resilience4j |
| :--- | :--- | :--- |
| **状态** | 维护模式 | **活跃开发** |
| **Java 版本** | Java 6+ | Java 8+ (函数式风格) |
| **依赖** | Guava, Apache Commons | Vavr (可选) |
| **模块化** | 强耦合 | **高度模块化** |
| **配置** | Archaius (复杂) | Spring Boot Config (标准) |
| **隔离** | 主要是线程池 | 主要是信号量，也支持线程池 |

> [!WARNING]
> Hystrix 使用 RxJava 1，过度依赖线程池隔离导致上下文传递困难。Resilience4j 基于装饰器模式，更轻量，更易于与 Reactor/WebFlux 集成。
