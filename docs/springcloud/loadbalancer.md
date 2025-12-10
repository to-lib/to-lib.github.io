---
title: LoadBalancer 负载均衡
sidebar_label: LoadBalancer
sidebar_position: 7.1
---

# Spring Cloud LoadBalancer

> [!TIP]
> **新一代负载均衡器**: Spring Cloud LoadBalancer 是 Spring Cloud 官方提供的客户端负载均衡器，用于替代 Netflix Ribbon。它基于 Spring WebFlux (Reactor) 实现，支持响应式编程。

## 1. 简介

### 为什么取代 Ribbon？

- **Ribbon 停止更新**: Netflix Ribbon 已进入维护模式。
- **技术栈演进**: Spring Cloud 全面拥抱响应式编程，Ribbon 基于阻塞 IO，不适合 Reactor 栈。
- **轻量级**: Spring Cloud LoadBalancer 更加轻量，移除对 Netflix 库的依赖。

### 核心特性

- **客户端负载均衡**: 在客户端选择服务实例。
- **支持响应式**: 原生支持 Reactor 模式。
- **可插拔策略**: 支持轮询、随机等多种策略，易于扩展。
- **缓存支持**: 支持 Caffeine 缓存服务列表。

## 2. 快速开始

### 添加依赖

如果在 classpath 中没有 Ribbon，Spring Cloud 会自动使用 LoadBalancer。如果有 Ribbon，需要排除它。

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-loadbalancer</artifactId>
</dependency>
```

### 使用 @LoadBalanced

用法与 Ribbon 完全一致，通过 `RestTemplate` 或 `WebClient` 调用。

```java
@Configuration
public class WebClientConfig {

    @Bean
    @LoadBalanced
    public WebClient.Builder webClientBuilder() {
        return WebClient.builder();
    }
    
    @Bean
    @LoadBalanced
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
}
```

### 服务调用

```java
@Service
public class OrderService {

    @Autowired
    private WebClient.Builder webClientBuilder;

    public Mono<User> getUser(Long userId) {
        return webClientBuilder.build()
            .get()
            .uri("http://user-service/users/{id}", userId)
            .retrieve()
            .bodyToMono(User.class);
    }
}
```

## 3. 负载均衡策略

默认使用 **RoundRobin (轮询)** 策略。

### 切换为随机策略 (Random)

需要自定义配置类：

```java
public class RandomLoadBalancerConfig {

    @Bean
    public ReactorLoadBalancer<ServiceInstance> randomLoadBalancer(Environment environment,
            LoadBalancerClientFactory loadBalancerClientFactory) {
        String name = environment.getProperty(LoadBalancerClientFactory.PROPERTY_NAME);
        return new RandomLoadBalancer(loadBalancerClientFactory
                .getLazyProvider(name, ServiceInstanceListSupplier.class),
                name);
    }
}
```

在启动类或配置类上使用注解应用配置：

```java
@Configuration
@LoadBalancerClient(name = "user-service", configuration = RandomLoadBalancerConfig.class)
// 或者对所有服务生效
// @LoadBalancerClients(defaultConfiguration = RandomLoadBalancerConfig.class)
public class AppConfig {
}
```

### 基于 Nacos 权重的策略

如果你使用 Nacos 作为注册中心，可以使用 Nacos 提供的基于权重的负载均衡策略。

```java
@Configuration
public class NacosLoadBalancerConfig {
    @Bean
    public ReactorLoadBalancer<ServiceInstance> nacosLoadBalancer(Environment environment,
            LoadBalancerClientFactory loadBalancerClientFactory, NacosDiscoveryProperties nacosDiscoveryProperties) {
        String name = environment.getProperty(LoadBalancerClientFactory.PROPERTY_NAME);
        return new NacosLoadBalancer(loadBalancerClientFactory
                .getLazyProvider(name, ServiceInstanceListSupplier.class),
                name, nacosDiscoveryProperties);
    }
}
```

## 4. 缓存服务列表

默认情况下，LoadBalancer 每次都会去注册中心获取服务列表。为了提高性能，可以使用 Caffeine 进行缓存。

### 添加依赖

```xml
<dependency>
    <groupId>com.github.ben-manes.caffeine</groupId>
    <artifactId>caffeine</artifactId>
</dependency>
```

### 配置缓存

```yaml
spring:
  cloud:
    loadbalancer:
      cache:
        enabled: true
        # 缓存过期时间
        ttl: 35s
        # 缓存容量
        capacity: 256
```

## 5. 自定义负载均衡器

实现 `ReactorServiceInstanceLoadBalancer` 接口即可自定义逻辑。

```java
public class MyCustomLoadBalancer implements ReactorServiceInstanceLoadBalancer {
    // ... 实现 choose 方法
}
```

## 6. 总结

| 特性 | Ribbon | Spring Cloud LoadBalancer |
| :--- | :--- | :--- |
| **维护状态** | 维护模式 (Deprecated) | **活跃开发 (推荐)** |
| **编程模型** | 阻塞 IO | 阻塞 & **响应式 (Reactor)** |
| **依赖** | Netflix Libraries | Spring Framework |
| **配置** | 复杂 (XML/Properties) | 简单 (Java Config/YAML) |

> [!IMPORTANT]
> 新项目中建议直接使用 Spring Cloud LoadBalancer。对于老项目，如果迁移成本较高，可以继续使用 Ribbon 但需注意其不再会有新功能。
