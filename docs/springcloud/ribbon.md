---
id: ribbon
title: Ribbon 负载均衡
sidebar_label: Ribbon
sidebar_position: 7
---

# Ribbon 负载均衡

> [!TIP] > **客户端负载均衡**: Ribbon 提供客户端负载均衡功能，支持多种负载均衡策略，与 Eureka、Feign 无缝集成。

## 1. Ribbon 简介

### 什么是 Ribbon？

**Ribbon** 是 Netflix 开源的客户端负载均衡器，在微服务架构中用于实现服务间调用的负载均衡。

### 客户端 vs 服务端负载均衡

| 类型           | 实现方式               | 示例               |
| -------------- | ---------------------- | ------------------ |
| 服务端负载均衡 | 独立的负载均衡器       | Nginx, HAProxy, F5 |
| 客户端负载均衡 | 客户端内置负载均衡逻辑 | Ribbon             |

### 核心概念

- **Rule（规则）** - 负载均衡策略
- **Ping** - 检测服务实例是否存活
- **ServerList** - 服务实例列表
- **ServerListFilter** - 服务实例过滤器

## 2. 快速开始

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-ribbon</artifactId>
</dependency>
```

### 使用 @LoadBalanced

```java
@Configuration
public class RestTemplateConfig {

    @Bean
    @LoadBalanced  // 启用负载均衡
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
}
```

### 调用服务

```java
@Service
public class OrderService {

    @Autowired
    private RestTemplate restTemplate;

    public User getUser(Long userId) {
        // 使用服务名调用，Ribbon 自动负载均衡
        String url = "http://user-service/users/" + userId;
        return restTemplate.getForObject(url, User.class);
    }
}
```

## 3. 负载均衡策略

### 内置策略

| 策略                      | 说明         |
| ------------------------- | ------------ |
| RoundRobinRule            | 轮询（默认） |
| RandomRule                | 随机         |
| RetryRule                 | 重试         |
| WeightedResponseTimeRule  | 响应时间加权 |
| BestAvailableRule         | 最小并发     |
| AvailabilityFilteringRule | 可用性过滤   |
| ZoneAvoidanceRule         | 区域感知     |

### 全局配置

```java
@Configuration
public class RibbonConfig {

    @Bean
    public IRule ribbonRule() {
        // 使用随机策略
        return new RandomRule();
    }
}
```

### 针对服务配置（YAML）

```yaml
# 针对 user-service 的配置
user-service:
  ribbon:
    # 负载均衡策略
    NFLoadBalancerRuleClassName: com.netflix.loadbalancer.RandomRule
```

### 针对服务配置（Java）

```java
// 注意：不要加 @Configuration，否则会成为全局配置
public class UserServiceRibbonConfig {

    @Bean
    public IRule ribbonRule() {
        return new RandomRule();
    }
}

@Configuration
@RibbonClient(name = "user-service", configuration = UserServiceRibbonConfig.class)
public class RibbonClientConfig {
}
```

## 4. 负载均衡策略详解

### RoundRobinRule - 轮询

```java
// 按顺序依次选择服务实例
// 实例1 -> 实例2 -> 实例3 -> 实例1 -> ...
public class RoundRobinRule extends AbstractLoadBalancerRule {
    // 简单轮询，适合服务实例性能相近的场景
}
```

### RandomRule - 随机

```java
// 随机选择服务实例
public class RandomRule extends AbstractLoadBalancerRule {
    // 完全随机，长期来看请求分布均匀
}
```

### WeightedResponseTimeRule - 响应时间加权

```java
// 根据响应时间分配权重，响应时间越短权重越高
public class WeightedResponseTimeRule extends RoundRobinRule {
    // 自动计算每个实例的平均响应时间
    // 适合服务实例性能差异较大的场景
}
```

### RetryRule - 重试

```java
// 在指定时间内重试获取可用实例
public class RetryRule extends AbstractLoadBalancerRule {
    // 默认使用 RoundRobinRule 选择实例
    // 如果选择失败，在超时时间内重试
}

// 配置
@Bean
public IRule ribbonRule() {
    return new RetryRule(new RandomRule(), 500); // 500ms 超时
}
```

### BestAvailableRule - 最小并发

```java
// 选择并发请求数最小的实例
public class BestAvailableRule extends ClientConfigEnabledRoundRobinRule {
    // 跳过熔断的实例
    // 选择当前并发连接最少的实例
}
```

### AvailabilityFilteringRule - 可用性过滤

```java
// 过滤掉故障实例和高并发实例
public class AvailabilityFilteringRule extends PredicateBasedRule {
    // 过滤规则：
    // 1. 跳过连接失败次数过多的实例
    // 2. 跳过并发连接数过高的实例
}
```

### ZoneAvoidanceRule - 区域感知

```java
// Spring Cloud 默认策略
public class ZoneAvoidanceRule extends PredicateBasedRule {
    // 综合考虑服务器所在区域的性能和可用性
    // 选择最优区域内的实例
}
```

## 5. 自定义负载均衡策略

```java
public class CustomRule extends AbstractLoadBalancerRule {

    @Override
    public Server choose(Object key) {
        ILoadBalancer lb = getLoadBalancer();
        List<Server> servers = lb.getAllServers();

        if (servers.isEmpty()) {
            return null;
        }

        // 自定义选择逻辑
        // 例如：根据版本号选择
        String targetVersion = "v2";
        for (Server server : servers) {
            Map<String, String> metadata =
                ((DiscoveryEnabledServer) server).getInstanceInfo().getMetadata();
            if (targetVersion.equals(metadata.get("version"))) {
                return server;
            }
        }

        // 没有匹配的版本，使用轮询
        return new RoundRobinRule().choose(key);
    }
}
```

## 6. Ribbon 配置参数

### 全局配置

```yaml
ribbon:
  # 连接超时时间（毫秒）
  ConnectTimeout: 1000
  # 读取超时时间（毫秒）
  ReadTimeout: 3000
  # 是否对所有操作进行重试
  OkToRetryOnAllOperations: false
  # 同一实例最大重试次数（不包括首次请求）
  MaxAutoRetries: 0
  # 切换实例的最大重试次数
  MaxAutoRetriesNextServer: 1
  # 是否启用 Eureka
  eureka:
    enabled: true
```

### 针对服务配置

```yaml
user-service:
  ribbon:
    NFLoadBalancerRuleClassName: com.netflix.loadbalancer.RandomRule
    NFLoadBalancerPingClassName: com.netflix.loadbalancer.PingUrl
    NIWSServerListClassName: com.netflix.loadbalancer.ConfigurationBasedServerList
    NIWSServerListFilterClassName: com.netflix.loadbalancer.ZoneAffinityServerListFilter
    ConnectTimeout: 2000
    ReadTimeout: 5000
    MaxAutoRetries: 1
    MaxAutoRetriesNextServer: 2
```

## 7. 重试机制

### 启用重试

```xml
<dependency>
    <groupId>org.springframework.retry</groupId>
    <artifactId>spring-retry</artifactId>
</dependency>
```

```yaml
spring:
  cloud:
    loadbalancer:
      retry:
        enabled: true

user-service:
  ribbon:
    # 同一实例重试次数
    MaxAutoRetries: 1
    # 切换实例重试次数
    MaxAutoRetriesNextServer: 2
    # 是否对所有请求进行重试
    OkToRetryOnAllOperations: false
```

### 重试计算

```
总请求次数 = (MaxAutoRetries + 1) * (MaxAutoRetriesNextServer + 1)
```

例如：`MaxAutoRetries=1, MaxAutoRetriesNextServer=2`

```
总请求次数 = (1 + 1) * (2 + 1) = 6 次
```

## 8. 饥饿加载

默认情况下，Ribbon 是懒加载的（首次请求时才创建客户端），可能导致首次请求超时。

### 启用饥饿加载

```yaml
ribbon:
  eager-load:
    # 启用饥饿加载
    enabled: true
    # 需要饥饿加载的服务
    clients: user-service, order-service
```

## 9. 结合 Eureka 使用

Ribbon 可以从 Eureka 获取服务实例列表：

```yaml
eureka:
  client:
    service-url:
      defaultZone: http://localhost:8761/eureka/

user-service:
  ribbon:
    # 使用 Eureka 提供的服务列表
    NIWSServerListClassName: com.netflix.niws.loadbalancer.DiscoveryEnabledNIWSServerList
    # 不启用 Eureka（如果要手动指定服务列表）
    # eureka:
    #   enabled: false
    # listOfServers: localhost:8081,localhost:8082
```

## 10. 手动指定服务列表

```yaml
user-service:
  ribbon:
    eureka:
      enabled: false
    # 手动指定服务实例列表
    listOfServers: localhost:8081,localhost:8082,localhost:8083
```

## 11. Ping 机制

Ribbon 定期 Ping 服务实例以检测可用性。

### Ping 策略

| 策略              | 说明               |
| ----------------- | ------------------ |
| NoOpPing          | 不检测（默认）     |
| PingUrl           | 通过 URL 检测      |
| PingConstant      | 认为所有实例都可用 |
| NIWSDiscoveryPing | 基于 Eureka 的检测 |

### 配置 Ping

```yaml
user-service:
  ribbon:
    NFLoadBalancerPingClassName: com.netflix.loadbalancer.PingUrl
    # Ping 间隔（秒）
    NFLoadBalancerPingInterval: 30
```

## 12. 实战示例

### 灰度发布

```java
public class GrayRule extends AbstractLoadBalancerRule {

    @Override
    public Server choose(Object key) {
        ILoadBalancer lb = getLoadBalancer();
        List<Server> servers = lb.getAllServers();

        // 从请求上下文获取灰度标识
        String grayFlag = GrayContext.getGrayFlag();

        if ("true".equals(grayFlag)) {
            // 选择灰度版本
            for (Server server : servers) {
                if (isGrayServer(server)) {
                    return server;
                }
            }
        }

        // 选择正式版本
        for (Server server : servers) {
            if (!isGrayServer(server)) {
                return server;
            }
        }

        return null;
    }

    private boolean isGrayServer(Server server) {
        Map<String, String> metadata =
            ((DiscoveryEnabledServer) server).getInstanceInfo().getMetadata();
        return "true".equals(metadata.get("gray"));
    }
}
```

## 13. 最佳实践

### 策略选择

- **性能相近** - 使用 RoundRobinRule（轮询）
- **性能差异大** - 使用 WeightedResponseTimeRule（响应时间加权）
- **简单场景** - 使用 RandomRule（随机）
- **需要容错** - 使用 RetryRule（重试）

### 超时配置

```yaml
user-service:
  ribbon:
    # 连接超时：尽量短
    ConnectTimeout: 1000
    # 读取超时：根据业务设置
    ReadTimeout: 3000
    # 合理设置重试次数
    MaxAutoRetries: 0
    MaxAutoRetriesNextServer: 1
```

### 监控

- 监控请求响应时间
- 监控失败率
- 监控各实例的负载分布

## 14. 常见问题

### 首次请求超时

**原因**: 懒加载导致

**解决**: 启用饥饿加载

### 负载不均衡

**原因**:

- 策略选择不当
- 实例性能差异大
- 缓存问题

**解决**:

- 选择合适的策略
- 使用 WeightedResponseTimeRule
- 刷新服务列表缓存

### 重试过多

**原因**: MaxAutoRetries 和 MaxAutoRetriesNextServer 设置过大

**解决**: 减小重试次数

## 15. 总结

| 概念           | 说明                     |
| -------------- | ------------------------ |
| 客户端负载均衡 | 在客户端实现负载均衡     |
| 负载均衡策略   | 多种内置策略，支持自定义 |
| 重试机制       | 自动重试失败的请求       |
| 服务列表       | 支持 Eureka 或手动配置   |

---

**关键要点**：

- Ribbon 提供客户端负载均衡
- 支持多种负载均衡策略
- 可以从 Eureka 获取服务列表
- 合理配置超时和重试参数

**下一步**：学习 [Hystrix 熔断器](./hystrix.md)
