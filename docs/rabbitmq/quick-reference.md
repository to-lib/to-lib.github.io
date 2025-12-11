---
sidebar_position: 13
title: "快速参考"
description: "RabbitMQ 命令和配置速查"
---

# RabbitMQ 快速参考

## 服务管理

```bash
# 启动/停止/重启
sudo systemctl start rabbitmq-server
sudo systemctl stop rabbitmq-server
sudo systemctl restart rabbitmq-server

# 状态检查
rabbitmqctl status
rabbitmqctl cluster_status
```

## 用户管理

```bash
# 添加用户
rabbitmqctl add_user <user> <password>

# 删除用户
rabbitmqctl delete_user <user>

# 修改密码
rabbitmqctl change_password <user> <new_password>

# 设置管理员
rabbitmqctl set_user_tags <user> administrator

# 设置权限
rabbitmqctl set_permissions -p / <user> ".*" ".*" ".*"

# 查看用户
rabbitmqctl list_users
rabbitmqctl list_permissions
```

## 队列管理

```bash
# 列出队列
rabbitmqctl list_queues
rabbitmqctl list_queues name messages consumers

# 清空队列
rabbitmqctl purge_queue <queue>

# 删除队列
rabbitmqctl delete_queue <queue>
```

## 交换机管理

```bash
# 列出交换机
rabbitmqctl list_exchanges

# 列出绑定
rabbitmqctl list_bindings
```

## 连接管理

```bash
# 列出连接
rabbitmqctl list_connections

# 列出通道
rabbitmqctl list_channels

# 关闭连接
rabbitmqctl close_connection <name>
```

## 插件管理

```bash
# 列出插件
rabbitmq-plugins list

# 启用/禁用
rabbitmq-plugins enable rabbitmq_management
rabbitmq-plugins disable rabbitmq_management
```

## 策略配置

```bash
# 镜像队列
rabbitmqctl set_policy ha-all ".*" '{"ha-mode":"all"}'

# 消息 TTL
rabbitmqctl set_policy ttl ".*" '{"message-ttl":60000}'

# 列出/删除策略
rabbitmqctl list_policies
rabbitmqctl clear_policy <name>
```

## Java 速查

### 连接

```java
ConnectionFactory factory = new ConnectionFactory();
factory.setHost("localhost");
factory.setPort(5672);
factory.setUsername("guest");
factory.setPassword("guest");
Connection connection = factory.newConnection();
Channel channel = connection.createChannel();
```

### 声明

```java
// 队列
channel.queueDeclare("queue", true, false, false, null);

// 交换机
channel.exchangeDeclare("exchange", "direct", true);

// 绑定
channel.queueBind("queue", "exchange", "routing-key");
```

### 发送

```java
channel.basicPublish("exchange", "key",
    MessageProperties.PERSISTENT_TEXT_PLAIN,
    message.getBytes());
```

### 消费

```java
channel.basicQos(10);
channel.basicConsume("queue", false,
    (tag, delivery) -> {
        process(delivery.getBody());
        channel.basicAck(delivery.getEnvelope().getDeliveryTag(), false);
    },
    tag -> {});
```

### 确认

```java
// 发布确认
channel.confirmSelect();
channel.waitForConfirms(5000);

// 消费确认
channel.basicAck(deliveryTag, false);
channel.basicNack(deliveryTag, false, true);
```

## 默认端口

| 端口  | 用途       |
| ----- | ---------- |
| 5672  | AMQP       |
| 15672 | 管理界面   |
| 25672 | 集群通信   |
| 4369  | epmd       |
| 15692 | Prometheus |

## 常用配置

```ini
# rabbitmq.conf

# 内存限制
vm_memory_high_watermark.relative = 0.6

# 磁盘限制
disk_free_limit.relative = 2.0

# 心跳
heartbeat = 60

# 通道限制
channel_max = 2047

# 日志
log.file.level = info
```

## 队列参数

| 参数                        | 说明                     |
| --------------------------- | ------------------------ |
| `x-message-ttl`             | 消息过期时间 (ms)        |
| `x-expires`                 | 队列过期时间 (ms)        |
| `x-max-length`              | 最大消息数               |
| `x-max-length-bytes`        | 最大字节数               |
| `x-dead-letter-exchange`    | 死信交换机               |
| `x-dead-letter-routing-key` | 死信路由键               |
| `x-max-priority`            | 最大优先级               |
| `x-queue-mode`              | 队列模式 (lazy)          |
| `x-queue-type`              | 队列类型 (quorum/stream) |

## Docker 命令

```bash
# 启动
docker run -d --name rabbitmq \
  -p 5672:5672 -p 15672:15672 \
  rabbitmq:3-management

# 查看日志
docker logs -f rabbitmq

# 进入容器
docker exec -it rabbitmq bash
```

## Spring Boot 配置

```yaml
spring:
  rabbitmq:
    host: localhost
    port: 5672
    username: guest
    password: guest
    virtual-host: /
    listener:
      simple:
        prefetch: 10
        acknowledge-mode: manual
```
