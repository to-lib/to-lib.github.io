---
title: Stream 消息驱动
sidebar_label: Stream 消息驱动
sidebar_position: 10
---

# Spring Cloud Stream

> [!TIP]
> **屏蔽底层差异**: Spring Cloud Stream 是一个用于构建消息驱动微服务的框架。它通过统一的编程模型屏蔽了底层消息中间件（RabbitMQ, Kafka, RocketMQ 等）的差异，让开发者专注于业务逻辑。

## 1. 核心概念

Spring Cloud Stream 通过 **Binder (绑定器)** 将代码与消息中间件连接起来。

- **Binder (绑定器)**: 负责与外部消息中间件集成 (如 RabbitMQ, Kafka)。
- **Binding (绑定)**: 连接应用程序与消息中间件的桥梁 (Input/Output)。
- **Message (消息)**: 标准化的数据结构，包含 Headers 和 Payload。

```mermaid
graph LR
    App[微服务应用] --> Output[Binding <br/> Output Channel]
    Output --> Binder[Binder <br/> (RabbitMQ/Kafka)]
    Binder --> Middleware[消息中间件]
    Middleware --> Binder2[Binder]
    Binder2 --> Input[Binding <br/> Input Channel]
    Input --> App2[微服务应用]
```

## 2. 快速开始 (RabbitMQ 示例)

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-stream-rabbit</artifactId>
</dependency>
```

如果是 Kafka，则替换为 `spring-cloud-starter-stream-kafka`。

### 编写代码 (函数式编程风格)

Spring Cloud Stream 3.x+ 推荐使用 Java Util Function (`Supplier`, `Function`, `Consumer`) 风格。

```java
@Configuration
public class StreamConfig {

    // 生产者: 定时发送消息
    @Bean
    public Supplier<String> produce() {
        return () -> "Hello Stream: " + System.currentTimeMillis();
    }

    // 处理器: 接收并转换消息
    @Bean
    public Function<String, String> process() {
        return message -> {
            System.out.println("Processing: " + message);
            return message.toUpperCase();
        };
    }

    // 消费者: 接收最终消息
    @Bean
    public Consumer<String> consume() {
        return message -> {
            System.out.println("Consumed: " + message);
        };
    }
}
```

### 配置 (application.yml)

配置 Binding 与函数的映射关系。

```yaml
spring:
  cloud:
    stream:
      function:
        # 定义需要生效的函数，多个用分号隔开
        definition: produce;process;consume
      bindings:
        # 格式: <functionName>-<in/out>-<index>
        
        # 生产者输出
        produce-out-0:
          destination: topic-source
        
        # 处理器输入
        process-in-0:
          destination: topic-source
          group: process-group # 消费者组
        # 处理器输出
        process-out-0:
          destination: topic-dest
        
        # 消费者输入
        consume-in-0:
          destination: topic-dest
          group: consume-group
```

## 3. 高级特性

### 消费者组 (Consumer Group)

通过 `group` 属性设置。同一组内的消费者竞争消费消息，实现负载均衡；不同组的消费者广播消费（发布-订阅）。

### 分区 (Partitioning)

支持消息分区，确保具有相同 Key 的消息被发送到同一个分区，保证顺序性。

```yaml
spring:
  cloud:
    stream:
      bindings:
        produce-out-0:
          producer:
            # key 的提取逻辑
            partitionKeyExpression: payload.id
            # 分区数量
            partitionCount: 2
```

### 错误处理 (DLQ)

支持死信队列 (Dead Letter Queue)。当消息消费失败重试多次后，发送到 DLQ。

```yaml
spring:
  cloud:
    stream:
      rabbit:
        bindings:
          consume-in-0:
            consumer:
              autoBindDlq: true
```

## 4. 为什么使用 Stream?

- **解耦**: 开发者无需关注 MQ 客户端 API，只需关注业务逻辑。
- **切换方便**: 切换 MQ 只需更换依赖和配置，代码无需修改。
- **功能增强**: 提供了自动重试、死信队列、分区、消费组等微服务常用功能开箱即用的支持。

## 5. 常用 Binder

- **RabbitMQ**: `spring-cloud-stream-binder-rabbit`
- **Apache Kafka**: `spring-cloud-stream-binder-kafka`
- **RocketMQ**: `spring-cloud-starter-stream-rocketmq` (由 Alibaba 提供)
