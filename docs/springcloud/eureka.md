---
title: Eureka 服务注册与发现
sidebar_label: Eureka
sidebar_position: 3
---

# Eureka 服务注册与发现

> [!TIP] > **服务注册发现是微服务的基础**: Eureka 提供服务注册、发现、健康检查等功能。理解 Eureka 的工作原理对于构建稳定的微服务系统至关重要。

## 1. Eureka 简介

### 什么是 Eureka？

**Eureka** 是 Netflix 开源的服务注册与发现组件，是 Spring Cloud 体系中最常用的服务治理组件。

### 核心概念

- **Eureka Server** - 服务注册中心，提供服务注册和发现功能
- **Eureka Client** - 服务提供者和消费者，向 Server 注册并获取服务列表
- **服务注册** - 服务启动时向 Server 注册自己的信息
- **服务发现** - 从 Server 获取其他服务的信息
- **心跳续约** - 定期发送心跳表明服务健康

## 2. Eureka Server 搭建

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-eureka-server</artifactId>
</dependency>
```

### 启用 Eureka Server

```java
@SpringBootApplication
@EnableEurekaServer
public class EurekaServerApplication {
    public static void main(String[] args) {
        SpringApplication.run(EurekaServerApplication.class, args);
    }
}
```

### 单机配置

```yaml
server:
  port: 8761

spring:
  application:
    name: eureka-server

eureka:
  instance:
    hostname: localhost
  client:
    # 不向注册中心注册自己
    register-with-eureka: false
    # 不从注册中心获取服务列表
    fetch-registry: false
    service-url:
      defaultZone: http://${eureka.instance.hostname}:${server.port}/eureka/
  server:
    # 关闭自我保护模式（开发环境）
    enable-self-preservation: false
    # 清理间隔（毫秒）
    eviction-interval-timer-in-ms: 5000
```

## 3. Eureka Client 配置

### 服务提供者

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
</dependency>
```

```java
@SpringBootApplication
@EnableEurekaClient
public class UserServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(UserServiceApplication.class, args);
    }
}
```

```yaml
server:
  port: 8081

spring:
  application:
    name: user-service

eureka:
  client:
    service-url:
      defaultZone: http://localhost:8761/eureka/
  instance:
    # 使用 IP 地址注册
    prefer-ip-address: true
    # 实例 ID
    instance-id: ${spring.cloud.client.ip-address}:${server.port}
    # 心跳间隔（秒）
    lease-renewal-interval-in-seconds: 30
    # 过期时间（秒）
    lease-expiration-duration-in-seconds: 90
```

### 服务消费者

```java
@SpringBootApplication
@EnableEurekaClient
public class OrderServiceApplication {

    @Bean
    @LoadBalanced  // 启用负载均衡
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }

    public static void main(String[] args) {
        SpringApplication.run(OrderServiceApplication.class, args);
    }
}
```

```java
@Service
public class OrderService {

    @Autowired
    private RestTemplate restTemplate;

    public User getUser(Long userId) {
        // 使用服务名调用
        String url = "http://user-service/users/" + userId;
        return restTemplate.getForObject(url, User.class);
    }
}
```

## 4. Eureka 高可用集群

### 集群架构

```
Eureka Server 1 (8761) ←→ Eureka Server 2 (8762) ←→ Eureka Server 3 (8763)
         ↑                        ↑                         ↑
         └────────────────────────┴─────────────────────────┘
                            服务实例
```

### Server 1 配置

```yaml
server:
  port: 8761

spring:
  application:
    name: eureka-server

eureka:
  instance:
    hostname: eureka-server-1
  client:
    service-url:
      # 向其他节点注册
      defaultZone: http://eureka-server-2:8762/eureka/,http://eureka-server-3:8763/eureka/
```

### Server 2 配置

```yaml
server:
  port: 8762

spring:
  application:
    name: eureka-server

eureka:
  instance:
    hostname: eureka-server-2
  client:
    service-url:
      defaultZone: http://eureka-server-1:8761/eureka/,http://eureka-server-3:8763/eureka/
```

### Client 配置（连接集群）

```yaml
eureka:
  client:
    service-url:
      defaultZone: http://eureka-server-1:8761/eureka/,http://eureka-server-2:8762/eureka/,http://eureka-server-3:8763/eureka/
```

## 5. 核心参数配置

### Server 端参数

```yaml
eureka:
  server:
    # 是否启用自我保护模式
    enable-self-preservation: true
    # 清理无效节点的时间间隔（毫秒）
    eviction-interval-timer-in-ms: 60000
    # 自我保护阈值
    renewal-percent-threshold: 0.85
    # 响应缓存更新间隔（毫秒）
    response-cache-update-interval-ms: 30000
    # 响应缓存过期时间（秒）
    response-cache-auto-expiration-in-seconds: 180
```

### Client 端参数

```yaml
eureka:
  client:
    # 是否向注册中心注册
    register-with-eureka: true
    # 是否从注册中心获取服务列表
    fetch-registry: true
    # 获取服务列表的间隔时间（秒）
    registry-fetch-interval-seconds: 30

  instance:
    # 使用 IP 地址注册
    prefer-ip-address: true
    # 实例 ID
    instance-id: ${spring.cloud.client.ip-address}:${spring.application.name}:${server.port}
    # 心跳间隔（秒）
    lease-renewal-interval-in-seconds: 30
    # 过期时间（秒）
    lease-expiration-duration-in-seconds: 90
    # 实例状态页面路径
    status-page-url-path: /actuator/info
    # 健康检查路径
    health-check-url-path: /actuator/health
```

## 6. 服务下线与剔除

### 主动下线

```java
@RestController
public class ServiceController {

    @Autowired
    private DiscoveryClient discoveryClient;

    @PostMapping("/offline")
    public String offline() {
        // 主动下线
        discoveryClient.deregisterInstance();
        return "服务已下线";
    }
}
```

### 自动剔除

Eureka Server 会定期检查服务实例的心跳：

- 默认 90 秒未收到心跳，剔除实例
- 可通过 `lease-expiration-duration-in-seconds` 配置

## 7. 自我保护模式

### 什么是自我保护？

当网络分区故障发生时，Eureka Server 在短时间内丢失过多客户端心跳，会进入自我保护模式：

- 不再剔除任何服务实例
- 仍然接受新服务的注册和查询请求
- 等待网络恢复

### 触发条件

```
最近一分钟收到的心跳数 < 期望收到的心跳数 * 阈值
```

默认阈值为 0.85（85%）

### 配置

```yaml
eureka:
  server:
    # 关闭自我保护（生产环境建议开启）
    enable-self-preservation: false
    # 自我保护阈值
    renewal-percent-threshold: 0.85
```

## 8. 元数据

### 标准元数据

Eureka 自动收集的元数据：

- hostname
- IP 地址
- 端口
- 状态页面 URL
- 健康检查 URL

### 自定义元数据

```yaml
eureka:
  instance:
    metadata-map:
      zone: zone1
      version: 1.0.0
      env: production
```

```java
@Service
public class ServiceInfoService {

    @Autowired
    private DiscoveryClient discoveryClient;

    public void printMetadata() {
        List<ServiceInstance> instances =
            discoveryClient.getInstances("user-service");

        for (ServiceInstance instance : instances) {
            Map<String, String> metadata = instance.getMetadata();
            System.out.println("Zone: " + metadata.get("zone"));
            System.out.println("Version: " + metadata.get("version"));
        }
    }
}
```

## 9. 健康检查

### 基于心跳的健康检查

默认情况下，Eureka 只检查心跳：

```yaml
eureka:
  client:
    healthcheck:
      enabled: false # 默认关闭
```

### 基于 Actuator 的健康检查

```yaml
eureka:
  client:
    healthcheck:
      enabled: true # 启用健康检查
```

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>
```

## 10. 最佳实践

### 生产环境配置建议

```yaml
eureka:
  server:
    # 启用自我保护
    enable-self-preservation: true
    # 清理间隔
    eviction-interval-timer-in-ms: 60000

  client:
    # 获取服务列表间隔
    registry-fetch-interval-seconds: 30

  instance:
    # 使用 IP 注册
    prefer-ip-address: true
    # 心跳间隔
    lease-renewal-interval-in-seconds: 30
    # 过期时间
    lease-expiration-duration-in-seconds: 90
    # 启用健康检查
    healthcheck:
      enabled: true
```

### 集群部署

- **至少 3 个节点** - 保证高可用
- **跨机房部署** - 提高容灾能力
- **负载均衡** - 客户端配置多个 Server 地址

### 监控告警

- 监控注册实例数量
- 监控心跳失败率
- 监控自我保护模式状态
- 监控服务可用性

## 11. 常见问题

### 服务下线后仍然可以被调用

**原因**: 客户端缓存了服务列表

**解决**:

- 减小 `registry-fetch-interval-seconds`
- 启用健康检查
- 手动刷新服务列表

### 自我保护模式频繁触发

**原因**: 网络不稳定或配置不当

**解决**:

- 检查网络连接
- 调整 `renewal-percent-threshold`
- 调整心跳间隔

### 服务注册慢

**原因**: 默认有缓存和延迟

**解决**:

- 减小 `response-cache-update-interval-ms`
- 减小 `registry-fetch-interval-seconds`

## 12. 总结

| 概念          | 说明                     |
| ------------- | ------------------------ |
| Eureka Server | 服务注册中心             |
| Eureka Client | 服务提供者和消费者       |
| 服务注册      | 服务启动时向 Server 注册 |
| 服务发现      | 从 Server 获取服务列表   |
| 心跳续约      | 定期发送心跳保持注册     |
| 自我保护      | 网络故障时保护服务列表   |

---

**关键要点**：

- Eureka 提供简单易用的服务注册发现功能
- 生产环境建议使用集群部署
- 合理配置心跳和过期时间
- 启用健康检查提高可靠性
- 理解自我保护模式的作用

**下一步**：学习 [Config 配置中心](/docs/springcloud/config)
