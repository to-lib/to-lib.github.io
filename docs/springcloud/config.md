---
id: config
title: Config 配置中心
sidebar_label: Config
sidebar_position: 4
---

# Config 配置中心

> [!TIP] > **集中配置管理**: Spring Cloud Config 提供集中化的外部配置管理，支持配置的版本控制、多环境管理和动态刷新。

## 1. Config 简介

### 什么是 Spring Cloud Config？

**Spring Cloud Config** 为分布式系统提供集中化的外部配置支持，配置存储在 Git、SVN 或本地文件系统中。

### 核心特性

- **集中管理** - 所有服务的配置统一管理
- **环境隔离** - 支持开发、测试、生产环境
- **版本控制** - 基于 Git 的配置版本管理
- **动态刷新** - 无需重启即可更新配置
- **配置加密** - 敏感信息加密存储

### 架构

```
Git Repository ← Config Server ← Config Client (微服务)
```

## 2. Config Server 搭建

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-config-server</artifactId>
</dependency>
```

### 启用 Config Server

```java
@SpringBootApplication
@EnableConfigServer
public class ConfigServerApplication {
    public static void main(String[] args) {
        SpringApplication.run(ConfigServerApplication.class, args);
    }
}
```

### Git 仓库配置

```yaml
server:
  port: 8888

spring:
  application:
    name: config-server
  cloud:
    config:
      server:
        git:
          # Git 仓库地址
          uri: https://github.com/your-org/config-repo
          # 默认分支
          default-label: main
          # 搜索路径
          search-paths: "{application}"
          # 用户名和密码（私有仓库）
          username: your-username
          password: your-password
          # 克隆超时时间（秒）
          timeout: 5
          # 强制拉取
          force-pull: true
```

### 本地文件系统配置

```yaml
spring:
  cloud:
    config:
      server:
        native:
          # 配置文件路径
          search-locations: file:///config-repo
  profiles:
    active: native
```

## 3. 配置文件组织

### Git 仓库结构

```
config-repo/
├── application.yml           # 全局配置
├── application-dev.yml       # 开发环境全局配置
├── application-prod.yml      # 生产环境全局配置
├── user-service.yml          # user-service 配置
├── user-service-dev.yml      # user-service 开发环境配置
├── user-service-prod.yml     # user-service 生产环境配置
├── order-service.yml
└── order-service-dev.yml
```

### 配置文件命名规则

```
{application}-{profile}.yml
```

- `application` - 应用名称，对应 `spring.application.name`
- `profile` - 环境名称，对应 `spring.profiles.active`

### 示例配置文件

**application.yml** (全局配置)

```yaml
# 所有服务的通用配置
logging:
  level:
    root: INFO

management:
  endpoints:
    web:
      exposure:
        include: "*"
```

**user-service-dev.yml**

```yaml
server:
  port: 8081

spring:
  datasource:
    url: jdbc:mysql://localhost:3306/user_db_dev
    username: dev_user
    password: dev_password

app:
  message: "开发环境配置"
```

## 4. Config Client 配置

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-config</artifactId>
</dependency>
```

### bootstrap.yml 配置

> [!IMPORTANT]
> Config Client 的配置必须写在 `bootstrap.yml` 中，因为它在 `application.yml` 之前加载。

```yaml
spring:
  application:
    name: user-service
  cloud:
    config:
      # Config Server 地址
      uri: http://localhost:8888
      # 环境
      profile: dev
      # 分支
      label: main
      # 快速失败
      fail-fast: true
      # 重试配置
      retry:
        max-attempts: 6
        initial-interval: 1000
        max-interval: 2000
        multiplier: 1.1
```

### 使用配置

```java
@RestController
@RefreshScope  // 支持动态刷新
public class ConfigController {

    @Value("${app.message}")
    private String message;

    @GetMapping("/message")
    public String getMessage() {
        return message;
    }
}
```

## 5. 配置动态刷新

### 手动刷新

**添加依赖**

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>
```

**开启 refresh 端点**

```yaml
management:
  endpoints:
    web:
      exposure:
        include: refresh
```

**触发刷新**

```bash
curl -X POST http://localhost:8081/actuator/refresh
```

### 使用 Spring Cloud Bus 自动刷新

**添加依赖**

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-bus-amqp</artifactId>
</dependency>
```

**配置 RabbitMQ**

```yaml
spring:
  rabbitmq:
    host: localhost
    port: 5672
    username: guest
    password: guest

management:
  endpoints:
    web:
      exposure:
        include: bus-refresh
```

**触发全局刷新**

```bash
# 刷新所有实例
curl -X POST http://localhost:8888/actuator/bus-refresh

# 刷新指定服务
curl -X POST http://localhost:8888/actuator/bus-refresh/user-service:**
```

## 6. 配置加密

### 安装 JCE

需要安装 Java Cryptography Extension (JCE)

### 配置密钥

```yaml
encrypt:
  # 对称加密密钥
  key: my-secret-key

  # 或使用非对称加密
  # key-store:
  #   location: classpath:/server.jks
  #   password: keystore-password
  #   alias: config-server-key
  #   secret: key-password
```

### 加密配置

**加密**

```bash
curl http://localhost:8888/encrypt -d "mysecret"
# 返回: {cipher}AQA8F1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9
```

**在配置文件中使用**

```yaml
spring:
  datasource:
    username: dev_user
    password: "{cipher}AQA8F1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9"
```

**解密**

```bash
curl http://localhost:8888/decrypt -d "{cipher}AQA8F1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9"
# 返回: mysecret
```

## 7. 多仓库配置

### 不同服务使用不同仓库

```yaml
spring:
  cloud:
    config:
      server:
        git:
          uri: https://github.com/your-org/default-config
          repos:
            # user-service 使用专用仓库
            user:
              pattern: user-service*
              uri: https://github.com/your-org/user-config
            # order-service 使用专用仓库
            order:
              pattern: order-service*
              uri: https://github.com/your-org/order-config
```

## 8. 高可用配置

### Config Server 集群

**部署多个 Config Server 实例**

```yaml
# Server 1
server:
  port: 8888

# Server 2
server:
  port: 8889
```

**Client 配置多个 Server**

```yaml
spring:
  cloud:
    config:
      uri: http://config-server-1:8888,http://config-server-2:8889
      fail-fast: true
```

### 结合 Eureka

**Config Server 注册到 Eureka**

```yaml
eureka:
  client:
    service-url:
      defaultZone: http://localhost:8761/eureka/
```

**Client 通过服务发现**

```yaml
spring:
  cloud:
    config:
      discovery:
        enabled: true
        service-id: config-server
      profile: dev

eureka:
  client:
    service-url:
      defaultZone: http://localhost:8761/eureka/
```

## 9. 配置优先级

配置加载顺序（优先级从高到低）：

```
1. /{application}/{profile}/{label}
2. /{application}/{profile}
3. /{application}/default
4. /application/{profile}/{label}
5. /application/{profile}
6. /application/default
```

示例：

```
user-service-dev-main.yml     # 最高优先级
user-service-dev.yml
user-service.yml
application-dev-main.yml
application-dev.yml
application.yml                # 最低优先级
```

## 10. 访问配置的 REST API

Config Server 提供 REST API 访问配置：

```bash
# /{application}/{profile}/{label}
GET /user-service/dev/main

# /{application}-{profile}.yml
GET /user-service-dev.yml

# /{label}/{application}-{profile}.yml
GET /main/user-service-dev.yml
```

**响应示例**

```json
{
  "name": "user-service",
  "profiles": ["dev"],
  "label": "main",
  "version": "abc123",
  "state": null,
  "propertySources": [
    {
      "name": "https://github.com/your-org/config-repo/user-service-dev.yml",
      "source": {
        "server.port": 8081,
        "app.message": "开发环境配置"
      }
    }
  ]
}
```

## 11. 最佳实践

### 配置文件管理

- **敏感信息加密** - 密码、密钥等使用加密
- **环境隔离** - 不同环境使用不同配置文件
- **版本控制** - 利用 Git 管理配置变更历史
- **Code Review** - 配置变更也需要审核

### 配置刷新策略

- **重要配置** - 使用 Spring Cloud Bus 自动刷新
- **敏感配置** - 需要重启服务
- **频繁变更** - 考虑使用配置中心 UI（如 Apollo、Nacos）

### 高可用

- **集群部署** - 部署多个 Config Server
- **服务发现** - 结合 Eureka 实现自动发现
- **本地缓存** - 配置本地缓存，Server 不可用时使用缓存

## 12. 常见问题

### 配置不生效

**检查项**:

- 配置文件命名是否正确
- profile 和 label 是否匹配
- 是否加了 `@RefreshScope`
- 是否触发了刷新

### 启动失败

**原因**: Config Server 不可用

**解决**:

```yaml
spring:
  cloud:
    config:
      fail-fast: false # 不快速失败
```

### 配置刷新不及时

**解决**:

- 使用 Spring Cloud Bus
- 配置 Git Webhook 触发刷新
- 减小 Config Server 的缓存时间

## 13. 总结

| 特性     | 说明                 |
| -------- | -------------------- |
| 集中管理 | 统一管理所有服务配置 |
| 环境隔离 | 支持多环境配置       |
| 版本控制 | 基于 Git 的版本管理  |
| 动态刷新 | 无需重启更新配置     |
| 配置加密 | 保护敏感信息         |

---

**关键要点**：

- Config Server 提供集中化配置管理
- 配置存储在 Git 仓库中，便于版本控制
- 使用 `@RefreshScope` 支持动态刷新
- 敏感配置需要加密
- 生产环境建议使用集群部署

**下一步**：学习 [Gateway API 网关](./gateway)
