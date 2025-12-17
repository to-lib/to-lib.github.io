---
title: Spring Cloud 快速参考
sidebar_label: 快速参考
sidebar_position: 20
---

# Spring Cloud 快速参考

> [!TIP]
> 本页面提供 Spring Cloud 常用配置、注解、依赖的速查表，方便快速查阅。

## 1. 版本兼容对照表

| Spring Cloud       | Spring Boot  | Java      |
| ------------------ | ------------ | --------- |
| 2023.0.x (Leyton)  | 3.2.x, 3.3.x | 17+       |
| 2022.0.x (Kilburn) | 3.0.x, 3.1.x | 17+       |
| 2021.0.x (Jubilee) | 2.6.x, 2.7.x | 8, 11, 17 |
| 2020.0.x (Ilford)  | 2.4.x, 2.5.x | 8, 11     |

## 2. 常用注解速查

### 服务注册发现

| 注解                     | 说明                 | 所属组件     |
| ------------------------ | -------------------- | ------------ |
| `@EnableEurekaServer`    | 启用 Eureka 服务端   | Eureka       |
| `@EnableEurekaClient`    | 启用 Eureka 客户端   | Eureka       |
| `@EnableDiscoveryClient` | 启用服务发现（通用） | Spring Cloud |

### 配置中心

| 注解                  | 说明               | 所属组件 |
| --------------------- | ------------------ | -------- |
| `@EnableConfigServer` | 启用配置中心服务端 | Config   |
| `@RefreshScope`       | 支持配置动态刷新   | Config   |

### 服务调用

| 注解                  | 说明              | 所属组件     |
| --------------------- | ----------------- | ------------ |
| `@EnableFeignClients` | 启用 Feign 客户端 | OpenFeign    |
| `@FeignClient`        | 声明 Feign 客户端 | OpenFeign    |
| `@LoadBalanced`       | 启用负载均衡      | LoadBalancer |

### 容错保护

| 注解                    | 说明         | 所属组件       |
| ----------------------- | ------------ | -------------- |
| `@CircuitBreaker`       | 声明熔断     | Resilience4j   |
| `@RateLimiter`          | 声明限流     | Resilience4j   |
| `@Retry`                | 声明重试     | Resilience4j   |
| `@Bulkhead`             | 声明舱壁隔离 | Resilience4j   |
| `@HystrixCommand`       | 定义熔断方法 | Hystrix (维护) |
| `@EnableCircuitBreaker` | 启用熔断器   | Hystrix (维护) |

### 消息驱动

| 注解              | 说明              | 所属组件 |
| ----------------- | ----------------- | -------- |
| `@EnableBinding`  | 启用消息绑定 (旧) | Stream   |
| `@StreamListener` | 消息监听器 (旧)   | Stream   |

## 3. 常用依赖速查

### Spring Cloud BOM

```xml
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>org.springframework.cloud</groupId>
            <artifactId>spring-cloud-dependencies</artifactId>
            <version>2023.0.0</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>
```

### 核心组件依赖

| 组件              | 依赖                                               |
| ----------------- | -------------------------------------------------- |
| Eureka Server     | `spring-cloud-starter-netflix-eureka-server`       |
| Eureka Client     | `spring-cloud-starter-netflix-eureka-client`       |
| Config Server     | `spring-cloud-config-server`                       |
| Config Client     | `spring-cloud-starter-config`                      |
| Gateway           | `spring-cloud-starter-gateway`                     |
| OpenFeign         | `spring-cloud-starter-openfeign`                   |
| LoadBalancer      | `spring-cloud-starter-loadbalancer`                |
| Resilience4j      | `spring-cloud-starter-circuitbreaker-resilience4j` |
| Stream (RabbitMQ) | `spring-cloud-starter-stream-rabbit`               |
| Stream (Kafka)    | `spring-cloud-starter-stream-kafka`                |
| Sleuth            | `spring-cloud-starter-sleuth`                      |
| Bus (RabbitMQ)    | `spring-cloud-starter-bus-amqp`                    |
| Consul Discovery  | `spring-cloud-starter-consul-discovery`            |

## 4. 常用配置速查

### Eureka Server

```yaml
eureka:
  server:
    enable-self-preservation: true # 自我保护模式
    eviction-interval-timer-in-ms: 60000 # 清理间隔
```

### Eureka Client

```yaml
eureka:
  client:
    service-url:
      defaultZone: http://localhost:8761/eureka/
    registry-fetch-interval-seconds: 30 # 获取服务列表间隔
  instance:
    prefer-ip-address: true # 使用 IP 注册
    lease-renewal-interval-in-seconds: 30 # 心跳间隔
    lease-expiration-duration-in-seconds: 90 # 过期时间
```

### Config Client

```yaml
spring:
  cloud:
    config:
      uri: http://localhost:8888
      profile: dev
      label: main
      fail-fast: true
```

### Gateway

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
            - StripPrefix=1
```

### Resilience4j CircuitBreaker

```yaml
resilience4j:
  circuitbreaker:
    instances:
      backendA:
        slidingWindowSize: 10
        minimumNumberOfCalls: 5
        failureRateThreshold: 50
        waitDurationInOpenState: 5s
```

### Resilience4j RateLimiter

```yaml
resilience4j:
  ratelimiter:
    instances:
      backendA:
        limitRefreshPeriod: 1s
        limitForPeriod: 10
        timeoutDuration: 500ms
```

### LoadBalancer

```yaml
spring:
  cloud:
    loadbalancer:
      cache:
        enabled: true
        ttl: 35s
        capacity: 256
```

### Stream

```yaml
spring:
  cloud:
    stream:
      function:
        definition: process
      bindings:
        process-in-0:
          destination: input-topic
          group: my-group
        process-out-0:
          destination: output-topic
```

## 5. 端口规划建议

| 服务          | 建议端口  |
| ------------- | --------- |
| Eureka Server | 8761      |
| Config Server | 8888      |
| API Gateway   | 8080      |
| Zipkin        | 9411      |
| 业务服务      | 8001-8999 |

## 6. Actuator 常用端点

| 端点                        | 说明                        |
| --------------------------- | --------------------------- |
| `/actuator/health`          | 健康检查                    |
| `/actuator/info`            | 应用信息                    |
| `/actuator/refresh`         | 刷新配置 (需 @RefreshScope) |
| `/actuator/bus-refresh`     | 全局配置刷新 (需 Bus)       |
| `/actuator/circuitbreakers` | 熔断器状态                  |
| `/actuator/metrics`         | 应用指标                    |

## 7. 组件选型对照

| 功能     | Netflix (旧)  | Spring Cloud (新)  | 阿里巴巴   |
| -------- | ------------- | ------------------ | ---------- |
| 服务注册 | Eureka        | Consul             | Nacos      |
| 配置中心 | Config        | Config             | Nacos      |
| 负载均衡 | Ribbon        | **LoadBalancer**   | Dubbo      |
| 服务调用 | Feign         | **OpenFeign**      | Dubbo      |
| 熔断降级 | Hystrix       | **Resilience4j**   | Sentinel   |
| API 网关 | Zuul          | **Gateway**        | Gateway    |
| 链路追踪 | Sleuth+Zipkin | Micrometer Tracing | SkyWalking |

---

**相关文档**：

- [核心概念](/docs/springcloud/core-concepts)
- [最佳实践](/docs/springcloud/best-practices)
- [常见问题](/docs/springcloud/faq)
