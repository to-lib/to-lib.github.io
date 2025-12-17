---
id: quick-reference
title: Spring Cloud Alibaba 快速参考
sidebar_label: 快速参考
sidebar_position: 12
---

# Spring Cloud Alibaba 快速参考

> [!TIP] > **速查手册**: 本文档汇总了 Spring Cloud Alibaba 开发中常用的配置、注解、依赖和命令，方便快速查阅。

## 版本对应关系

| Spring Cloud Alibaba | Spring Cloud | Spring Boot | 状态 |
| -------------------- | ------------ | ----------- | ---- |
| 2023.0.1.0           | 2023.0.x     | 3.2.x       | 推荐 |
| 2023.0.0.0           | 2023.0.x     | 3.2.x       | 稳定 |
| 2022.0.0.0           | 2022.0.x     | 3.0.x       | 维护 |
| 2021.0.5.0           | 2021.0.x     | 2.6.x       | 维护 |
| 2021.0.1.0           | 2021.0.x     | 2.6.x       | 维护 |

## Maven 依赖速查

### 依赖管理 (BOM)

```xml
<dependencyManagement>
    <dependencies>
        <!-- Spring Boot -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-dependencies</artifactId>
            <version>3.2.0</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
        <!-- Spring Cloud -->
        <dependency>
            <groupId>org.springframework.cloud</groupId>
            <artifactId>spring-cloud-dependencies</artifactId>
            <version>2023.0.0</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
        <!-- Spring Cloud Alibaba -->
        <dependency>
            <groupId>com.alibaba.cloud</groupId>
            <artifactId>spring-cloud-alibaba-dependencies</artifactId>
            <version>2023.0.0.0</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>
```

### 核心组件依赖

| 组件       | GroupId             | ArtifactId                                   |
| ---------- | ------------------- | -------------------------------------------- |
| Nacos 注册 | com.alibaba.cloud   | spring-cloud-starter-alibaba-nacos-discovery |
| Nacos 配置 | com.alibaba.cloud   | spring-cloud-starter-alibaba-nacos-config    |
| Sentinel   | com.alibaba.cloud   | spring-cloud-starter-alibaba-sentinel        |
| Seata      | com.alibaba.cloud   | spring-cloud-starter-alibaba-seata           |
| RocketMQ   | org.apache.rocketmq | rocketmq-spring-boot-starter                 |
| Dubbo      | org.apache.dubbo    | dubbo-spring-boot-starter                    |

## 常用注解速查

### 服务注册与发现

| 注解                     | 说明                       | 包                                            |
| ------------------------ | -------------------------- | --------------------------------------------- |
| `@EnableDiscoveryClient` | 启用服务注册发现           | org.springframework.cloud.client.discovery    |
| `@LoadBalanced`          | 开启 RestTemplate 负载均衡 | org.springframework.cloud.client.loadbalancer |

### 配置管理

| 注解                   | 说明              | 包                                                  |
| ---------------------- | ----------------- | --------------------------------------------------- |
| `@RefreshScope`        | 支持配置动态刷新  | org.springframework.cloud.context.config.annotation |
| `@NacosPropertySource` | 指定 Nacos 配置源 | com.alibaba.nacos.spring.context.annotation         |
| `@Value`               | 注入配置项        | org.springframework.beans.factory.annotation        |

### Sentinel 流控

| 注解                | 说明                    | 包                                  |
| ------------------- | ----------------------- | ----------------------------------- |
| `@SentinelResource` | 定义资源点（流控/熔断） | com.alibaba.csp.sentinel.annotation |

**@SentinelResource 属性**：

```java
@SentinelResource(
    value = "resourceName",        // 资源名
    blockHandler = "handleBlock",  // 流控/熔断处理方法
    fallback = "handleFallback",   // 异常降级方法
    blockHandlerClass = Handler.class  // 独立处理类
)
```

### Seata 分布式事务

| 注解                      | 说明         | 包                         |
| ------------------------- | ------------ | -------------------------- |
| `@GlobalTransactional`    | 开启全局事务 | io.seata.spring.annotation |
| `@TwoPhaseBusinessAction` | TCC 模式注解 | io.seata.rm.tcc.api        |

### Dubbo RPC

| 注解              | 说明            | 包                                 |
| ----------------- | --------------- | ---------------------------------- |
| `@DubboService`   | 暴露 Dubbo 服务 | org.apache.dubbo.config.annotation |
| `@DubboReference` | 引用 Dubbo 服务 | org.apache.dubbo.config.annotation |

### RocketMQ 消息

| 注解                       | 说明       | 包                                    |
| -------------------------- | ---------- | ------------------------------------- |
| `@RocketMQMessageListener` | 消息监听器 | org.apache.rocketmq.spring.annotation |

## 配置速查

### Nacos 配置

```yaml
spring:
  cloud:
    nacos:
      # 服务注册
      discovery:
        server-addr: localhost:8848
        namespace: dev
        group: DEFAULT_GROUP
        weight: 1
        cluster-name: DEFAULT
      # 配置中心
      config:
        server-addr: localhost:8848
        file-extension: yaml
        namespace: dev
        group: DEFAULT_GROUP
        refresh-enabled: true
```

### Sentinel 配置

```yaml
spring:
  cloud:
    sentinel:
      transport:
        dashboard: localhost:8080
        port: 8719
      eager: true
      # Nacos 持久化
      datasource:
        flow:
          nacos:
            server-addr: localhost:8848
            dataId: ${spring.application.name}-flow-rules
            groupId: SENTINEL_GROUP
            rule-type: flow
```

### Seata 配置

```yaml
seata:
  enabled: true
  tx-service-group: my_tx_group
  service:
    vgroup-mapping:
      my_tx_group: default
  registry:
    type: nacos
    nacos:
      server-addr: localhost:8848
      namespace: ""
      group: SEATA_GROUP
```

### RocketMQ 配置

```yaml
rocketmq:
  name-server: localhost:9876
  producer:
    group: producer-group
    send-message-timeout: 3000
  consumer:
    group: consumer-group
```

### Dubbo 配置

```yaml
dubbo:
  application:
    name: ${spring.application.name}
  registry:
    address: nacos://localhost:8848
  protocol:
    name: dubbo
    port: 20880
  consumer:
    timeout: 3000
    retries: 2
```

## 常用命令速查

### Nacos Server

```bash
# 启动（单机）
sh startup.sh -m standalone
# Windows
startup.cmd -m standalone

# 停止
sh shutdown.sh
```

### Sentinel Dashboard

```bash
# 启动
java -Dserver.port=8080 -jar sentinel-dashboard.jar

# 指定用户名密码
java -Dserver.port=8080 \
     -Dsentinel.dashboard.auth.username=admin \
     -Dsentinel.dashboard.auth.password=123456 \
     -jar sentinel-dashboard.jar
```

### Seata Server

```bash
# 启动
sh seata-server.sh -p 8091 -m file

# 带配置启动
sh seata-server.sh -p 8091 -m nacos
```

### RocketMQ

```bash
# 启动 NameServer
nohup sh mqnamesrv &

# 启动 Broker
nohup sh mqbroker -n localhost:9876 &

# 停止
sh mqshutdown broker
sh mqshutdown namesrv
```

## 端口速查

| 服务                | 默认端口 | 说明              |
| ------------------- | -------- | ----------------- |
| Nacos               | 8848     | 控制台和服务发现  |
| Nacos gRPC          | 9848     | gRPC 通信端口     |
| Sentinel Dashboard  | 8080     | 控制台            |
| Sentinel Client     | 8719     | 与 Dashboard 通信 |
| Seata Server        | 8091     | TC 事务协调器     |
| RocketMQ NameServer | 9876     | 路由中心          |
| RocketMQ Broker     | 10911    | 消息服务端口      |
| Dubbo               | 20880    | RPC 通信端口      |

## 配置文件命名规则

### Nacos 配置

```
${spring.application.name}-${spring.profiles.active}.${file-extension}
```

**示例**：

- `user-service.yaml` - 无 profile
- `user-service-dev.yaml` - dev 环境
- `user-service-prod.yaml` - prod 环境

### 配置优先级（从高到低）

1. `${application}-${profile}.${ext}`
2. `${application}.${ext}`
3. `shared-configs`
4. `extension-configs`

## 健康检查端点

```yaml
management:
  endpoints:
    web:
      exposure:
        include: health,info,nacos-discovery,sentinel
  endpoint:
    health:
      show-details: always
```

**关键端点**：

| 端点                        | 说明           |
| --------------------------- | -------------- |
| `/actuator/health`          | 健康状态       |
| `/actuator/nacos-discovery` | Nacos 注册信息 |
| `/actuator/sentinel`        | Sentinel 信息  |

## 日志配置

```yaml
logging:
  level:
    com.alibaba.cloud: DEBUG
    com.alibaba.nacos: WARN
    io.seata: DEBUG
    org.apache.rocketmq: INFO
    org.apache.dubbo: INFO
```

---

**相关文档**：

- [核心概念](/docs/springcloud-alibaba/core-concepts) - 了解组件体系
- [快速开始](/docs/springcloud-alibaba/getting-started) - 搭建第一个项目
- [最佳实践](/docs/springcloud-alibaba/best-practices) - 生产级实践经验
- [FAQ](/docs/springcloud-alibaba/faq) - 常见问题解答
