---
sidebar_position: 4
title: "快速开始"
description: "快速搭建和使用 RocketMQ"
---

# RocketMQ 快速开始

本指南将帮助你快速搭建 RocketMQ 环境并进行基本操作。

## 环境要求

- **Java 8+**（推荐 JDK 11）
- **至少 4GB RAM**
- **Linux/MacOS/Windows**

## 安装 RocketMQ

### 1. 下载 RocketMQ

```bash
# 下载最新版本
wget https://dist.apache.org/repos/dist/release/rocketmq/5.1.4/rocketmq-all-5.1.4-bin-release.zip

# 解压
unzip rocketmq-all-5.1.4-bin-release.zip
cd rocketmq-all-5.1.4-bin-release
```

### 2. 启动 NameServer

```bash
# 启动 NameServer
nohup sh bin/mqnamesrv &

# 查看启动日志
tail -f ~/logs/rocketmqlogs/namesrv.log
```

看到 `The Name Server boot success` 表示启动成功。

### 3. 启动 Broker

```bash
# 启动 Broker
nohup sh bin/mqbroker -n localhost:9876 &

# 查看启动日志
tail -f ~/logs/rocketmqlogs/broker.log
```

看到 `The broker[...] boot success` 表示启动成功。

### 4. 验证安装

```bash
# 查看进程
jps | grep -E "NamesrvStartup|BrokerStartup"
```

## 基本操作

### 发送消息

```bash
# 设置 NameServer 地址
export NAMESRV_ADDR=localhost:9876

# 发送测试消息
sh bin/tools.sh org.apache.rocketmq.example.quickstart.Producer
```

### 消费消息

```bash
# 消费测试消息
sh bin/tools.sh org.apache.rocketmq.example.quickstart.Consumer
```

## Java 快速示例

### Maven 依赖

```xml
<dependency>
    <groupId>org.apache.rocketmq</groupId>
    <artifactId>rocketmq-client</artifactId>
    <version>5.1.4</version>
</dependency>
```

### 生产者示例

```java
import org.apache.rocketmq.client.producer.DefaultMQProducer;
import org.apache.rocketmq.client.producer.SendResult;
import org.apache.rocketmq.common.message.Message;

public class SimpleProducer {
    public static void main(String[] args) throws Exception {
        // 创建生产者，指定生产者组名
        DefaultMQProducer producer = new DefaultMQProducer("ProducerGroup");

        // 设置 NameServer 地址
        producer.setNamesrvAddr("localhost:9876");

        // 启动生产者
        producer.start();

        try {
            for (int i = 0; i < 10; i++) {
                // 创建消息
                Message msg = new Message(
                    "TopicTest",           // Topic
                    "TagA",                // Tag
                    ("Hello RocketMQ " + i).getBytes()  // Body
                );

                // 发送消息
                SendResult result = producer.send(msg);
                System.out.printf("发送结果: %s%n", result);
            }
        } finally {
            // 关闭生产者
            producer.shutdown();
        }
    }
}
```

### 消费者示例

```java
import org.apache.rocketmq.client.consumer.DefaultMQPushConsumer;
import org.apache.rocketmq.client.consumer.listener.ConsumeConcurrentlyStatus;
import org.apache.rocketmq.client.consumer.listener.MessageListenerConcurrently;
import org.apache.rocketmq.common.message.MessageExt;

public class SimpleConsumer {
    public static void main(String[] args) throws Exception {
        // 创建消费者，指定消费者组名
        DefaultMQPushConsumer consumer = new DefaultMQPushConsumer("ConsumerGroup");

        // 设置 NameServer 地址
        consumer.setNamesrvAddr("localhost:9876");

        // 订阅 Topic
        consumer.subscribe("TopicTest", "*");

        // 注册消息监听器
        consumer.registerMessageListener((MessageListenerConcurrently) (msgs, context) -> {
            for (MessageExt msg : msgs) {
                System.out.printf("收到消息: %s%n", new String(msg.getBody()));
            }
            return ConsumeConcurrentlyStatus.CONSUME_SUCCESS;
        });

        // 启动消费者
        consumer.start();
        System.out.println("消费者已启动...");
    }
}
```

## Docker 快速启动

### 使用 Docker Compose

创建 `docker-compose.yml`：

```yaml
version: "3.8"
services:
  namesrv:
    image: apache/rocketmq:5.1.4
    container_name: rocketmq-namesrv
    ports:
      - "9876:9876"
    command: sh mqnamesrv

  broker:
    image: apache/rocketmq:5.1.4
    container_name: rocketmq-broker
    ports:
      - "10911:10911"
      - "10909:10909"
    environment:
      - NAMESRV_ADDR=namesrv:9876
    command: sh mqbroker
    depends_on:
      - namesrv

  dashboard:
    image: apacherocketmq/rocketmq-dashboard:latest
    container_name: rocketmq-dashboard
    ports:
      - "8080:8080"
    environment:
      - JAVA_OPTS=-Drocketmq.namesrv.addr=namesrv:9876
    depends_on:
      - namesrv
```

启动服务：

```bash
# 启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

访问控制台：http://localhost:8080

## 常用管理命令

### Topic 管理

```bash
# 创建 Topic
sh bin/mqadmin updateTopic -n localhost:9876 -b localhost:10911 -t TopicTest

# 查看所有 Topic
sh bin/mqadmin topicList -n localhost:9876

# 查看 Topic 状态
sh bin/mqadmin topicStatus -n localhost:9876 -t TopicTest

# 删除 Topic
sh bin/mqadmin deleteTopic -n localhost:9876 -c DefaultCluster -t TopicTest
```

### 消费者组管理

```bash
# 查看所有消费者组
sh bin/mqadmin consumerProgress -n localhost:9876

# 查看消费者组消费进度
sh bin/mqadmin consumerProgress -n localhost:9876 -g ConsumerGroup
```

### 消息查询

```bash
# 根据 MessageId 查询
sh bin/mqadmin queryMsgById -n localhost:9876 -i <msgId>

# 根据 Key 查询
sh bin/mqadmin queryMsgByKey -n localhost:9876 -t TopicTest -k <key>
```

## Spring Boot 集成

### Maven 依赖

```xml
<dependency>
    <groupId>org.apache.rocketmq</groupId>
    <artifactId>rocketmq-spring-boot-starter</artifactId>
    <version>2.2.3</version>
</dependency>
```

### 配置文件

```yaml
rocketmq:
  name-server: localhost:9876
  producer:
    group: springboot-producer-group
```

### 生产者

```java
@Service
public class MessageProducer {

    @Autowired
    private RocketMQTemplate rocketMQTemplate;

    public void send(String message) {
        rocketMQTemplate.convertAndSend("TopicTest", message);
    }

    public void sendWithTag(String message) {
        rocketMQTemplate.convertAndSend("TopicTest:TagA", message);
    }

    public SendResult syncSend(String message) {
        return rocketMQTemplate.syncSend("TopicTest", message);
    }

    public void asyncSend(String message) {
        rocketMQTemplate.asyncSend("TopicTest", message, new SendCallback() {
            @Override
            public void onSuccess(SendResult result) {
                System.out.println("发送成功: " + result.getMsgId());
            }

            @Override
            public void onException(Throwable e) {
                System.err.println("发送失败: " + e.getMessage());
            }
        });
    }
}
```

### 消费者

```java
@Service
@RocketMQMessageListener(
    topic = "TopicTest",
    consumerGroup = "springboot-consumer-group"
)
public class MessageConsumer implements RocketMQListener<String> {

    @Override
    public void onMessage(String message) {
        System.out.println("收到消息: " + message);
    }
}
```

## 故障排查

### 检查 RocketMQ 状态

```bash
# 查看进程
ps aux | grep rocketmq

# 检查端口
netstat -tlnp | grep -E "9876|10911"

# 查看日志
tail -f ~/logs/rocketmqlogs/namesrv.log
tail -f ~/logs/rocketmqlogs/broker.log
```

### 常见问题

#### 1. NameServer 连接失败

```bash
# 检查 NameServer 是否启动
jps | grep NamesrvStartup

# 检查防火墙
firewall-cmd --list-ports
```

#### 2. 发送消息超时

```java
// 增加超时时间
producer.setSendMsgTimeout(10000);
```

#### 3. 内存不足

```bash
# 修改 JVM 参数
vi bin/runbroker.sh

# 调整内存配置
JAVA_OPT="${JAVA_OPT} -server -Xms2g -Xmx2g"
```

## 关闭服务

```bash
# 关闭 Broker
sh bin/mqshutdown broker

# 关闭 NameServer
sh bin/mqshutdown namesrv
```

## 下一步

- 📖 [核心概念](./core-concepts.md) - 深入理解 RocketMQ 架构
- 💻 [生产者详解](./producer.md) - 学习生产者高级用法
- 📊 [消费者详解](./consumer.md) - 学习消费者高级用法
- 🔄 [消息类型](./message-types.md) - 了解各种消息类型

## 参考资料

- [RocketMQ 官方文档](https://rocketmq.apache.org/docs/)
- [RocketMQ Docker 镜像](https://hub.docker.com/r/apache/rocketmq)
