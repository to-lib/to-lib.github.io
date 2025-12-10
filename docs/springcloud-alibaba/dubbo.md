---
id: dubbo
title: Dubbo RPC 框架
sidebar_label: Dubbo
sidebar_position: 7
---

# Dubbo RPC 框架

> [!TIP] > **高性能 RPC**: Dubbo 是阿里巴巴开源的高性能 RPC 框架，性能优于 OpenFeign，是微服务间通信的优秀选择。

## 1. Dubbo 简介

**Dubbo** 是一款高性能、轻量级的开源 RPC 框架，提供三大核心能力：

- **面向接口的远程方法调用**
- **智能容错和负载均衡**
- **服务自动注册与发现**

### Dubbo vs Feign

| 特性     | Dubbo                       | Feign  |
| -------- | --------------------------- | ------ |
| 协议     | Dubbo/gRPC/HTTP             | HTTP   |
| 性能     | 高                          | 中     |
| 序列化   | 多种（Hessian/Protobuf 等） | JSON   |
| 负载均衡 | 丰富的策略                  | Ribbon |
| 服务治理 | 完善                        | 基础   |

## 2. 快速开始

### 添加依赖

**Provider 和 Consumer 共同依赖**：

```xml
<dependency>
    <groupId>com.alibaba.cloud</groupId>
    <artifactId>spring-cloud-starter-dubbo</artifactId>
</dependency>

<dependency>
    <groupId>com.alibaba.cloud</groupId>
    <artifactId>spring-cloud-starter-alibaba-nacos-discovery</artifactId>
</dependency>
```

### 定义服务接口（API 模块）

```java
public interface UserService {
    User getUser(Long id);
    List<User> listUsers();
}
```

### 服务提供者（Provider）

```java
@DubboService(version = "1.0.0", group = "user")
public class UserServiceImpl implements UserService {

    @Override
    public User getUser(Long id) {
        return userRepository.findById(id);
    }

    @Override
    public List<User> listUsers() {
        return userRepository.findAll();
    }
}
```

**配置**：

```yaml
dubbo:
  application:
    name: user-service-provider
  protocol:
    name: dubbo
    port: 20880
  registry:
    address: nacos://localhost:8848
  scan:
    base-packages: com.example.service
```

### 服务消费者（Consumer）

```java
@RestController
public class OrderController {

    @DubboReference(version = "1.0.0", group = "user")
    private UserService userService;

    @GetMapping("/orders/{id}")
    public Order getOrder(@PathVariable Long id) {
        User user = userService.getUser(1L);
        return new Order(id, user);
    }
}
```

**配置**：

```yaml
dubbo:
  application:
    name: order-service-consumer
  registry:
    address: nacos://localhost:8848
```

## 3. 负载均衡

Dubbo 提供多种负载均衡策略：

| 策略           | 说明           |
| -------------- | -------------- |
| Random         | 随机（默认）   |
| RoundRobin     | 轮询           |
| LeastActive    | 最少活跃调用数 |
| ConsistentHash | 一致性 Hash    |

```java
@DubboReference(loadbalance = "roundrobin")
private UserService userService;
```

## 4. 集群容错

| 策略      | 说明                 |
| --------- | -------------------- |
| Failover  | 失败自动切换（默认） |
| Failfast  | 快速失败             |
| Failsafe  | 失败安全，忽略异常   |
| Failback  | 失败自动恢复         |
| Forking   | 并行调用多个服务     |
| Broadcast | 广播调用所有提供者   |

```java
@DubboReference(cluster = "failfast", retries = 0)
private UserService userService;
```

## 5. 服务版本

```java
// Provider
@DubboService(version = "1.0.0")
public class UserServiceV1 implements UserService {
    // 旧版本实现
}

@DubboService(version = "2.0.0")
public class UserServiceV2 implements UserService {
    // 新版本实现
}

// Consumer
@DubboReference(version = "2.0.0")
private UserService userService;

// 或使用通配符
@DubboReference(version = "*")
private UserService userService;
```

## 6. 服务分组

```java
// Provider
@DubboService(group = "default")
public class UserServiceDefault implements UserService {
    // 默认实现
}

@DubboService(group = "premium")
public class UserServicePremium implements UserService {
    // 高级实现
}

// Consumer
@DubboReference(group = "premium")
private UserService userService;
```

## 7. 异步调用

```java
// 服务接口
public interface UserService {
    User getUser(Long id);
    CompletableFuture<User> getUserAsync(Long id);
}

// Provider
@DubboService
public class UserServiceImpl implements UserService {

    @Override
    @Async
    public CompletableFuture<User> getUserAsync(Long id) {
        return CompletableFuture.supplyAsync(() -> getUser(id));
    }
}

// Consumer
@DubboReference
private UserService userService;

public void demo() {
    CompletableFuture<User> future = userService.getUserAsync(1L);
    future.thenAccept(user -> {
        System.out.println(user);
    });
}
```

## 8. 泛化调用

不需要服务接口的调用方式：

```java
@Autowired
private GenericService genericService;

public void demo() {
    GenericService userService = new GenericService();
    userService.setInterfaceClass(UserService.class);
    userService.setVersion("1.0.0");

    Object result = userService.$invoke(
        "getUser",
        new String[]{"java.lang.Long"},
        new Object[]{1L}
    );
}
```

## 9. 最佳实践

### 服务接口设计

- 接口应该简单明确
- 避免过度设计
- 定义在独立的 API 模块

### 超时配置

```java
@DubboReference(
    timeout = 3000,     // 调用超时（毫秒）
    retries = 2,        // 重试次数
    check = false       // 启动时不检查服务是否可用
)
private UserService userService;
```

### 线程模型

```yaml
dubbo:
  protocol:
    threads: 200 # 业务线程池大小
    iothreads: 4 # IO 线程池大小
```

### 参数回调

```java
// 回调接口
public interface CallbackListener {
    void onChange(String msg);
}

// 服务接口
public interface CallbackService {
    void addListener(String key, CallbackListener listener);
}
```

## 10. 总结

| 特性     | 说明             |
| -------- | ---------------- |
| 高性能   | 优于 HTTP        |
| 负载均衡 | 丰富的策略       |
| 服务治理 | 版本、分组、路由 |
| 容错机制 | 多种集群容错策略 |

---

**关键要点**：

- Dubbo 性能优于 Feign
- 提供丰富的负载均衡和容错策略
- 支持服务版本和分组管理
- 与 Nacos 无缝集成

**相关文档**：

- [Spring Cloud](../springcloud/index)
- [Nacos](/docs/springcloud-alibaba/nacos)
