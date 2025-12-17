---
title: Consul 服务注册与发现
sidebar_label: Consul
sidebar_position: 12
---

# Consul 服务注册与发现

> [!TIP]
> HashiCorp Consul 是一个多功能的服务网格解决方案，提供服务发现、配置管理、健康检查等功能。它是 Eureka 的替代方案，支持 CP 模式，适合对一致性要求较高的场景。

## 1. 简介

### 什么是 Consul？

Consul 是 HashiCorp 公司出品的开源工具，提供：

- **服务发现**: 注册和发现服务
- **健康检查**: 多种健康检查方式
- **KV 存储**: 分布式键值存储
- **多数据中心**: 原生支持多数据中心

### Consul vs Eureka

| 特性       | Consul           | Eureka       |
| ---------- | ---------------- | ------------ |
| CAP        | **CP（一致性）** | AP（可用性） |
| 健康检查   | 多种方式         | 仅心跳       |
| KV 存储    | ✅ 支持          | ❌ 不支持    |
| 多数据中心 | ✅ 原生支持      | ❌ 不支持    |
| 一致性协议 | Raft             | -            |

## 2. 安装 Consul

### Docker 安装

```bash
# 开发模式（单节点）
docker run -d --name consul \
  -p 8500:8500 \
  -p 8600:8600/udp \
  consul:latest agent -dev -ui -client=0.0.0.0

# 访问 UI: http://localhost:8500
```

### 本地安装

```bash
# macOS
brew install consul

# 开发模式启动
consul agent -dev -ui
```

### 验证安装

```bash
# 查看成员
consul members

# 查看服务
consul catalog services
```

## 3. Spring Cloud 集成

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-consul-discovery</artifactId>
</dependency>
```

### 服务配置

```yaml
spring:
  application:
    name: user-service
  cloud:
    consul:
      # Consul 地址
      host: localhost
      port: 8500
      discovery:
        # 服务名
        service-name: ${spring.application.name}
        # 使用 IP 注册
        prefer-ip-address: true
        # 健康检查路径
        health-check-path: /actuator/health
        # 健康检查间隔
        health-check-interval: 10s
        # 实例 ID
        instance-id: ${spring.application.name}:${server.port}
```

### 启用服务发现

```java
@SpringBootApplication
@EnableDiscoveryClient
public class UserServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(UserServiceApplication.class, args);
    }
}
```

## 4. 健康检查

### HTTP 健康检查

```yaml
spring:
  cloud:
    consul:
      discovery:
        health-check-path: /actuator/health
        health-check-interval: 10s
        health-check-timeout: 5s
        health-check-critical-timeout: 30s
```

### 自定义健康检查

```java
@Component
public class CustomHealthIndicator implements HealthIndicator {

    @Override
    public Health health() {
        // 自定义健康检查逻辑
        boolean isHealthy = checkSomething();

        if (isHealthy) {
            return Health.up()
                .withDetail("status", "服务正常")
                .build();
        } else {
            return Health.down()
                .withDetail("error", "服务异常")
                .build();
        }
    }

    private boolean checkSomething() {
        // 检查数据库连接、外部服务等
        return true;
    }
}
```

### 健康检查失败处理

```yaml
spring:
  cloud:
    consul:
      discovery:
        # 服务实例健康检查失败后多久标记为 critical
        health-check-critical-timeout: 1m
```

## 5. 服务调用

### 使用 LoadBalancer

```java
@Configuration
public class WebClientConfig {

    @Bean
    @LoadBalanced
    public WebClient.Builder webClientBuilder() {
        return WebClient.builder();
    }
}

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

### 使用 Feign

```java
@EnableFeignClients
@SpringBootApplication
public class OrderServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(OrderServiceApplication.class, args);
    }
}

@FeignClient("user-service")
public interface UserClient {

    @GetMapping("/users/{id}")
    User getUser(@PathVariable Long id);
}
```

## 6. 配置管理（KV 存储）

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-consul-config</artifactId>
</dependency>
```

### 配置 Consul Config

```yaml
# bootstrap.yml
spring:
  application:
    name: user-service
  profiles:
    active: dev
  cloud:
    consul:
      host: localhost
      port: 8500
      config:
        enabled: true
        # 配置格式
        format: YAML
        # 配置前缀
        prefix: config
        # 默认上下文
        default-context: application
        # 分隔符
        profile-separator: ","
        # 监听配置变更
        watch:
          enabled: true
          delay: 1000
```

### 在 Consul 中存储配置

```bash
# 存储配置
consul kv put config/user-service/data '
server:
  port: 8081
app:
  message: Hello from Consul
'

# 查看配置
consul kv get config/user-service/data
```

### 配置优先级

```
config/user-service,dev/data    # 最高优先级
config/user-service/data
config/application,dev/data
config/application/data          # 最低优先级
```

## 7. 高可用部署

### 集群架构

```
Consul Server 1 (Leader)
       ↑↓
Consul Server 2 ←→ Consul Server 3
       ↑              ↑
   Client 1        Client 2
```

### Server 配置

```json
{
  "server": true,
  "bootstrap_expect": 3,
  "datacenter": "dc1",
  "data_dir": "/consul/data",
  "encrypt": "your-gossip-key",
  "retry_join": ["consul-server-1", "consul-server-2", "consul-server-3"],
  "ui_config": {
    "enabled": true
  }
}
```

### 客户端连接集群

```yaml
spring:
  cloud:
    consul:
      host: consul-cluster-lb # 负载均衡地址
      port: 8500
      discovery:
        # 优先本地 agent
        prefer-agent-address: true
```

## 8. ACL 安全控制

### 启用 ACL

```json
{
  "acl": {
    "enabled": true,
    "default_policy": "deny",
    "tokens": {
      "master": "your-master-token"
    }
  }
}
```

### Spring Cloud 配置 Token

```yaml
spring:
  cloud:
    consul:
      discovery:
        acl-token: ${CONSUL_ACL_TOKEN}
      config:
        acl-token: ${CONSUL_ACL_TOKEN}
```

## 9. 常用命令

```bash
# 查看集群成员
consul members

# 查看所有服务
consul catalog services

# 查看服务详情
consul catalog service user-service

# 健康检查状态
consul health checks user-service

# KV 操作
consul kv get key
consul kv put key value
consul kv delete key

# 查看 Leader
consul operator raft list-peers
```

## 10. 常见问题

### 服务注册成功但健康检查失败

**检查项**:

1. 确保 `/actuator/health` 端点可访问
2. 检查 Consul 到服务的网络连通性
3. 检查健康检查超时配置

```yaml
spring:
  cloud:
    consul:
      discovery:
        health-check-timeout: 10s # 增加超时时间
```

### 服务无法发现其他服务

**检查项**:

1. 确保使用 `@EnableDiscoveryClient`
2. 确保使用 `@LoadBalanced` 的 RestTemplate/WebClient
3. 检查服务名是否正确

---

**相关文档**：

- [Eureka 服务注册与发现](/docs/springcloud/eureka)
- [LoadBalancer 负载均衡](/docs/springcloud/loadbalancer)
