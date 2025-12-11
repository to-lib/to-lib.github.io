---
sidebar_position: 4
title: "快速开始"
description: "快速搭建和使用 RabbitMQ"
---

# RabbitMQ 快速开始

本指南将帮助你快速搭建 RabbitMQ 环境并进行基本操作。

## 环境要求

- **Erlang/OTP 25+**
- **至少 1GB RAM**
- **Linux/MacOS/Windows**

## 安装 RabbitMQ

### 1. macOS 安装

```bash
# 使用 Homebrew 安装
brew install rabbitmq

# 启动 RabbitMQ
brew services start rabbitmq
```

### 2. Ubuntu/Debian 安装

```bash
# 添加 RabbitMQ 仓库
sudo apt-get install curl gnupg apt-transport-https -y

# 添加密钥
curl -1sLf "https://keys.openpgp.org/vks/v1/by-fingerprint/0A9AF2115F4687BD29803A206B73A36E6026DFCA" | sudo gpg --dearmor | sudo tee /usr/share/keyrings/com.rabbitmq.team.gpg > /dev/null

# 安装 RabbitMQ
sudo apt-get update -y
sudo apt-get install rabbitmq-server -y

# 启动服务
sudo systemctl start rabbitmq-server
sudo systemctl enable rabbitmq-server
```

### 3. Docker 安装（推荐）

```bash
# 启动 RabbitMQ（带管理界面）
docker run -d --name rabbitmq \
  -p 5672:5672 \
  -p 15672:15672 \
  -e RABBITMQ_DEFAULT_USER=admin \
  -e RABBITMQ_DEFAULT_PASS=admin123 \
  rabbitmq:3-management

# 查看日志
docker logs -f rabbitmq
```

### 4. 验证安装

```bash
# 检查 RabbitMQ 状态
rabbitmqctl status

# 访问管理界面
# 浏览器打开: http://localhost:15672
# 默认用户: guest / guest（仅限 localhost）
```

## 启用管理插件

```bash
# 启用管理界面插件
rabbitmq-plugins enable rabbitmq_management

# 重启服务（如果需要）
sudo systemctl restart rabbitmq-server
```

## 基本操作

### 用户管理

```bash
# 添加用户
rabbitmqctl add_user myuser mypassword

# 设置用户为管理员
rabbitmqctl set_user_tags myuser administrator

# 设置用户权限（所有 vhost）
rabbitmqctl set_permissions -p / myuser ".*" ".*" ".*"

# 列出用户
rabbitmqctl list_users
```

### 队列管理

```bash
# 列出队列
rabbitmqctl list_queues

# 列出队列详细信息
rabbitmqctl list_queues name messages consumers memory

# 清空队列
rabbitmqctl purge_queue my_queue

# 删除队列
rabbitmqctl delete_queue my_queue
```

### 交换机管理

```bash
# 列出交换机
rabbitmqctl list_exchanges

# 列出绑定关系
rabbitmqctl list_bindings
```

## Java 快速示例

### Maven 依赖

```xml
<dependency>
    <groupId>com.rabbitmq</groupId>
    <artifactId>amqp-client</artifactId>
    <version>5.20.0</version>
</dependency>
```

### 生产者示例

```java
import com.rabbitmq.client.Channel;
import com.rabbitmq.client.Connection;
import com.rabbitmq.client.ConnectionFactory;

public class SimpleProducer {
    private final static String QUEUE_NAME = "hello";

    public static void main(String[] args) throws Exception {
        // 创建连接工厂
        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost("localhost");
        factory.setPort(5672);
        factory.setUsername("guest");
        factory.setPassword("guest");

        // 创建连接和通道
        try (Connection connection = factory.newConnection();
             Channel channel = connection.createChannel()) {

            // 声明队列
            channel.queueDeclare(QUEUE_NAME, false, false, false, null);

            // 发送消息
            String message = "Hello RabbitMQ!";
            channel.basicPublish("", QUEUE_NAME, null, message.getBytes());

            System.out.println(" [x] Sent '" + message + "'");
        }
    }
}
```

### 消费者示例

```java
import com.rabbitmq.client.*;

public class SimpleConsumer {
    private final static String QUEUE_NAME = "hello";

    public static void main(String[] args) throws Exception {
        // 创建连接工厂
        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost("localhost");

        // 创建连接和通道
        Connection connection = factory.newConnection();
        Channel channel = connection.createChannel();

        // 声明队列
        channel.queueDeclare(QUEUE_NAME, false, false, false, null);
        System.out.println(" [*] Waiting for messages...");

        // 创建消费者回调
        DeliverCallback deliverCallback = (consumerTag, delivery) -> {
            String message = new String(delivery.getBody(), "UTF-8");
            System.out.println(" [x] Received '" + message + "'");
        };

        // 开始消费
        channel.basicConsume(QUEUE_NAME, true, deliverCallback, consumerTag -> {});
    }
}
```

## Spring Boot 集成

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-amqp</artifactId>
</dependency>
```

### 配置文件

```yaml
spring:
  rabbitmq:
    host: localhost
    port: 5672
    username: guest
    password: guest
    virtual-host: /
```

### 生产者示例

```java
@Service
public class MessageProducer {

    @Autowired
    private RabbitTemplate rabbitTemplate;

    public void sendMessage(String message) {
        rabbitTemplate.convertAndSend("my-exchange", "routing-key", message);
        System.out.println("Sent: " + message);
    }
}
```

### 消费者示例

```java
@Component
public class MessageConsumer {

    @RabbitListener(queues = "my-queue")
    public void receiveMessage(String message) {
        System.out.println("Received: " + message);
    }
}
```

### 配置类

```java
@Configuration
public class RabbitConfig {

    @Bean
    public Queue myQueue() {
        return new Queue("my-queue", true);
    }

    @Bean
    public DirectExchange myExchange() {
        return new DirectExchange("my-exchange");
    }

    @Bean
    public Binding binding(Queue myQueue, DirectExchange myExchange) {
        return BindingBuilder.bind(myQueue).to(myExchange).with("routing-key");
    }
}
```

## Docker Compose 部署

创建 `docker-compose.yml`：

```yaml
version: "3.8"
services:
  rabbitmq:
    image: rabbitmq:3-management
    container_name: rabbitmq
    hostname: rabbitmq
    ports:
      - "5672:5672" # AMQP 端口
      - "15672:15672" # 管理界面端口
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: admin123
      RABBITMQ_DEFAULT_VHOST: /
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    restart: unless-stopped

volumes:
  rabbitmq_data:
```

启动服务：

```bash
# 启动
docker-compose up -d

# 查看日志
docker-compose logs -f rabbitmq

# 停止
docker-compose down
```

## 常用管理命令

### 服务管理

```bash
# 启动服务
sudo systemctl start rabbitmq-server

# 停止服务
sudo systemctl stop rabbitmq-server

# 重启服务
sudo systemctl restart rabbitmq-server

# 查看状态
sudo systemctl status rabbitmq-server
```

### 节点信息

```bash
# 查看节点状态
rabbitmqctl status

# 查看集群状态
rabbitmqctl cluster_status

# 查看环境信息
rabbitmqctl environment
```

### 连接管理

```bash
# 列出连接
rabbitmqctl list_connections

# 列出通道
rabbitmqctl list_channels

# 关闭指定连接
rabbitmqctl close_connection <connection_name>
```

## 故障排查

### 检查 RabbitMQ 状态

```bash
# 查看进程
ps aux | grep rabbitmq

# 查看端口占用
netstat -tulpn | grep 5672

# 查看日志
tail -f /var/log/rabbitmq/rabbit@hostname.log
```

### 常见问题

#### 1. 无法连接到 RabbitMQ

```bash
# 检查服务是否启动
rabbitmqctl status

# 检查端口是否开放
nc -zv localhost 5672
```

#### 2. 管理界面无法访问

```bash
# 确认插件已启用
rabbitmq-plugins list

# 启用管理插件
rabbitmq-plugins enable rabbitmq_management
```

#### 3. 权限问题

```bash
# 检查用户权限
rabbitmqctl list_permissions

# 设置权限
rabbitmqctl set_permissions -p / username ".*" ".*" ".*"
```

## 下一步

- 📖 [核心概念](/docs/rabbitmq/core-concepts) - 深入理解 RabbitMQ 架构
- 💻 [生产者指南](/docs/rabbitmq/producer) - 学习生产者高级用法
- 📊 [消费者指南](/docs/rabbitmq/consumer) - 学习消费者高级用法
- ⚙️ [集群管理](/docs/rabbitmq/cluster-management) - 了解如何管理 RabbitMQ 集群

## 参考资料

- [RabbitMQ 官方快速开始](https://www.rabbitmq.com/tutorials/tutorial-one-java.html)
- [RabbitMQ Docker 镜像](https://hub.docker.com/_/rabbitmq)
