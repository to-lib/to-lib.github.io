---
id: getting-started
title: 快速开始指南
sidebar_label: 快速开始
sidebar_position: 3
---

# Spring Cloud Alibaba 快速开始

> [!TIP]
> **动手实践**: 通过本指南,你将在 30 分钟内搭建一个完整的微服务示例,包括服务注册、配置管理和服务调用。

## 1. 环境准备

### 必需软件

- **JDK**: 17 或更高版本
- **Maven**: 3.6.0 或更高版本
- **IDE**: IntelliJ IDEA 或 Eclipse
- **Docker**: 用于运行 Nacos (可选)

### 版本对应关系

| Spring Cloud Alibaba | Spring Cloud | Spring Boot |
| -------------------- | ------------ | ----------- |
| 2023.0.0.0           | 2023.0.x     | 3.2.x       |
| 2022.0.0.0           | 2022.0.x     | 3.0.x       |
| 2021.0.5.0           | 2021.0.x     | 2.6.x       |

## 2. 启动 Nacos Server

### 使用 Docker (推荐)

```bash
docker run -d \
  --name nacos \
  -p 8848:8848 \
  -p 9848:9848 \
  -e MODE=standalone \
  nacos/nacos-server:v2.3.0
```

### 手动下载启动

```bash
# 下载
wget https://github.com/alibaba/nacos/releases/download/2.3.0/nacos-server-2.3.0.zip

# 解压
unzip nacos-server-2.3.0.zip

# 启动 (Linux/Mac)
cd nacos/bin
sh startup.sh -m standalone

# 启动 (Windows)
cd nacos\bin
startup.cmd -m standalone
```

访问控制台: [http://localhost:8848/nacos](http://localhost:8848/nacos)

默认账号密码: `nacos/nacos`

## 3. 创建父项目

### 创建项目结构

```
springcloud-alibaba-demo
├── pom.xml
├── user-service (用户服务)
└── order-service (订单服务)
```

### 父 POM 配置

```xml title="pom.xml"
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.example</groupId>
    <artifactId>springcloud-alibaba-demo</artifactId>
    <version>1.0.0</version>
    <packaging>pom</packaging>

    <modules>
        <module>user-service</module>
        <module>order-service</module>
    </modules>

    <properties>
        <java.version>17</java.version>
        <spring-boot.version>3.2.0</spring-boot.version>
        <spring-cloud.version>2023.0.0</spring-cloud.version>
        <spring-cloud-alibaba.version>2023.0.0.0</spring-cloud-alibaba.version>
    </properties>

    <dependencyManagement>
        <dependencies>
            <!-- Spring Boot -->
            <dependency>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-dependencies</artifactId>
                <version>${spring-boot.version}</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>

            <!-- Spring Cloud -->
            <dependency>
                <groupId>org.springframework.cloud</groupId>
                <artifactId>spring-cloud-dependencies</artifactId>
                <version>${spring-cloud.version}</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>

            <!-- Spring Cloud Alibaba -->
            <dependency>
                <groupId>com.alibaba.cloud</groupId>
                <artifactId>spring-cloud-alibaba-dependencies</artifactId>
                <version>${spring-cloud-alibaba.version}</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>
        </dependencies>
    </dependencyManagement>

    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
                <version>${spring-boot.version}</version>
            </plugin>
        </plugins>
    </build>
</project>
```

## 4. 创建用户服务 (Provider)

### 添加依赖

```xml title="user-service/pom.xml"
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <parent>
        <groupId>com.example</groupId>
        <artifactId>springcloud-alibaba-demo</artifactId>
        <version>1.0.0</version>
    </parent>

    <artifactId>user-service</artifactId>

    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>

        <!-- Nacos 服务注册发现 -->
        <dependency>
            <groupId>com.alibaba.cloud</groupId>
            <artifactId>spring-cloud-starter-alibaba-nacos-discovery</artifactId>
        </dependency>

        <!-- Nacos 配置管理 -->
        <dependency>
            <groupId>com.alibaba.cloud</groupId>
            <artifactId>spring-cloud-starter-alibaba-nacos-config</artifactId>
        </dependency>

        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-actuator</artifactId>
        </dependency>
    </dependencies>
</project>
```

### 配置文件

```yaml title="user-service/src/main/resources/application.yml"
server:
  port: 8081

spring:
  application:
    name: user-service
  cloud:
    nacos:
      discovery:
        server-addr: localhost:8848
      config:
        server-addr: localhost:8848
        file-extension: yaml

management:
  endpoints:
    web:
      exposure:
        include: "*"
```

### 实体类

```java title="User.java"
package com.example.user.entity;

public class User {
    private Long id;
    private String name;
    private String email;

    public User() {}

    public User(Long id, String name, String email) {
        this.id = id;
        this.name = name;
        this.email = email;
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
}
```

### Controller

```java title="UserController.java"
package com.example.user.controller;

import com.example.user.entity.User;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/users")
public class UserController {

    private static final Map<Long, User> users = new HashMap<>();

    static {
        users.put(1L, new User(1L, "张三", "zhangsan@example.com"));
        users.put(2L, new User(2L, "李四", "lisi@example.com"));
        users.put(3L, new User(3L, "王五", "wangwu@example.com"));
    }

    @GetMapping("/{id}")
    public User getUser(@PathVariable Long id) {
        return users.get(id);
    }

    @GetMapping
    public Map<Long, User> getAllUsers() {
        return users;
    }
}
```

### 启动类

```java title="UserServiceApplication.java"
package com.example.user;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;

@SpringBootApplication
@EnableDiscoveryClient
public class UserServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(UserServiceApplication.class, args);
    }
}
```

## 5. 创建订单服务 (Consumer)

### 添加依赖

```xml title="order-service/pom.xml"
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <parent>
        <groupId>com.example</groupId>
        <artifactId>springcloud-alibaba-demo</artifactId>
        <version>1.0.0</version>
    </parent>

    <artifactId>order-service</artifactId>

    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>

        <!-- Nacos 服务注册发现 -->
        <dependency>
            <groupId>com.alibaba.cloud</groupId>
            <artifactId>spring-cloud-starter-alibaba-nacos-discovery</artifactId>
        </dependency>

        <!-- LoadBalancer -->
        <dependency>
            <groupId>org.springframework.cloud</groupId>
            <artifactId>spring-cloud-starter-loadbalancer</artifactId>
        </dependency>
    </dependencies>
</project>
```

### 配置文件

```yaml title="order-service/src/main/resources/application.yml"
server:
  port: 8082

spring:
  application:
    name: order-service
  cloud:
    nacos:
      discovery:
        server-addr: localhost:8848
```

### 配置类

```java title="RestTemplateConfig.java"
package com.example.order.config;

import org.springframework.cloud.client.loadbalancer.LoadBalanced;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

@Configuration
public class RestTemplateConfig {

    @Bean
    @LoadBalanced
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
}
```

### 实体类

```java title="Order.java"
package com.example.order.entity;

public class Order {
    private Long id;
    private Long userId;
    private String product;
    private Object user;

    public Order() {}

    public Order(Long id, Long userId, String product) {
        this.id = id;
        this.userId = userId;
        this.product = product;
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }
    public String getProduct() { return product; }
    public void setProduct(String product) { this.product = product; }
    public Object getUser() { return user; }
    public void setUser(Object user) { this.user = user; }
}
```

### Controller

```java title="OrderController.java"
package com.example.order.controller;

import com.example.order.entity.Order;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

@RestController
@RequestMapping("/orders")
public class OrderController {

    @Autowired
    private RestTemplate restTemplate;

    @GetMapping("/{id}")
    public Order getOrder(@PathVariable Long id) {
        Order order = new Order(id, 1L, "MacBook Pro");
        
        // 通过服务名调用用户服务
        Object user = restTemplate.getForObject(
            "http://user-service/users/" + order.getUserId(),
            Object.class
        );
        
        order.setUser(user);
        return order;
    }
}
```

### 启动类

```java title="OrderServiceApplication.java"
package com.example.order;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;

@SpringBootApplication
@EnableDiscoveryClient
public class OrderServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(OrderServiceApplication.class, args);
    }
}
```

## 6. 运行和测试

### 启动服务

1. **启动 Nacos Server** (如果还没启动)
2. **启动用户服务** - 运行 `UserServiceApplication`
3. **启动订单服务** - 运行 `OrderServiceApplication`

### 验证服务注册

访问 Nacos 控制台: [http://localhost:8848/nacos](http://localhost:8848/nacos)

在"服务管理" → "服务列表"中应该能看到:

- user-service
- order-service

### 测试接口

**用户服务**:

```bash
curl http://localhost:8081/users/1
```

响应:

```json
{
  "id": 1,
  "name": "张三",
  "email": "zhangsan@example.com"
}
```

**订单服务** (包含用户信息):

```bash
curl http://localhost:8082/orders/1
```

响应:

```json
{
  "id": 1,
  "userId": 1,
  "product": "MacBook Pro",
  "user": {
    "id": 1,
    "name": "张三",
    "email": "zhangsan@example.com"
  }
}
```

## 7. 添加配置管理

### 在 Nacos 中创建配置

1. 登录 Nacos 控制台
2. 进入"配置管理" → "配置列表"
3. 点击"+"创建配置

**配置信息**:

- Data ID: `user-service.yaml`
- Group: `DEFAULT_GROUP`
- 配置格式: `YAML`
- 配置内容:

```yaml
app:
  message: "Hello from Nacos Config!"
  version: "1.0.0"
```

### 使用配置

```java title="ConfigController.java"
package com.example.user.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.cloud.context.config.annotation.RefreshScope;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/config")
@RefreshScope  // 支持动态刷新
public class ConfigController {

    @Value("${app.message:默认消息}")
    private String message;

    @Value("${app.version:1.0.0}")
    private String version;

    @GetMapping
    public String getConfig() {
        return "Message: " + message + ", Version: " + version;
    }
}
```

### 测试配置

```bash
curl http://localhost:8081/config
```

响应:

```
Message: Hello from Nacos Config!, Version: 1.0.0
```

修改 Nacos 中的配置后,配置会自动刷新(无需重启服务)。

## 8. 项目结构总结

```
springcloud-alibaba-demo
├── pom.xml
├── user-service
│   ├── pom.xml
│   └── src
│       └── main
│           ├── java
│           │   └── com.example.user
│           │       ├── UserServiceApplication.java
│           │       ├── controller
│           │       │   ├── UserController.java
│           │       │   └── ConfigController.java
│           │       └── entity
│           │           └── User.java
│           └── resources
│               └── application.yml
└── order-service
    ├── pom.xml
    └── src
        └── main
            ├── java
            │   └── com.example.order
            │       ├── OrderServiceApplication.java
            │       ├── config
            │       │   └── RestTemplateConfig.java
            │       ├── controller
            │       │   └── OrderController.java
            │       └── entity
            │           └── Order.java
            └── resources
                └── application.yml
```

## 9. 常见问题

### 服务注册失败

**检查项**:

- Nacos Server 是否正常运行
- `server-addr` 配置是否正确
- 防火墙是否阻止了端口 8848

### 服务调用失败

**检查项**:

- 服务名是否正确 (`user-service` 而不是 `localhost:8081`)
- `@LoadBalanced` 注解是否添加
- 是否引入了 `spring-cloud-starter-loadbalancer` 依赖

### 配置不生效

**检查项**:

- Data ID 命名是否正确 (`${spring.application.name}.${file-extension}`)
- Group 是否匹配
- 是否添加了 `@RefreshScope` 注解

## 10. 下一步

恭喜!你已经成功搭建了一个基础的 Spring Cloud Alibaba 微服务系统。

**继续学习**:

- [Sentinel 流量控制](./sentinel) - 保护服务稳定性
- [Seata 分布式事务](./seata) - 解决数据一致性
- [RocketMQ 消息队列](./rocketmq) - 异步通信
- [Gateway 网关](./gateway) - 统一入口

**完整代码**: [GitHub 示例仓库](https://github.com/alibaba/spring-cloud-alibaba/tree/2022.x/spring-cloud-alibaba-examples)

---

**关键要点**:

- Spring Cloud Alibaba 与 Spring Boot 完美集成
- Nacos 同时提供服务注册和配置管理
- 服务调用通过服务名而不是 IP:端口
- 配置支持动态刷新,无需重启服务
