---
sidebar_position: 4
title: "快速开始"
description: "快速搭建和使用 Kafka"
---

# Kafka 快速开始

本指南将帮助你快速搭建 Kafka 环境并进行基本操作。

## 环境要求

- **Java 8+**
- **至少 2GB RAM**
- **Linux/MacOS/Windows**

## 安装 Kafka

### 1. 下载 Kafka

```bash
# 下载最新版本
wget https://downloads.apache.org/kafka/3.6.0/kafka_2.13-3.6.0.tgz

# 解压
tar -xzf kafka_2.13-3.6.0.tgz
cd kafka_2.13-3.6.0
```

### 2. 启动 Kafka（KRaft 模式）

```bash
# 生成集群 ID
KAFKA_CLUSTER_ID="$(bin/kafka-storage.sh random-uuid)"

# 格式化日志目录
bin/kafka-storage.sh format -t $KAFKA_CLUSTER_ID -c config/kraft/server.properties

# 启动 Kafka 服务器
bin/kafka-server-start.sh config/kraft/server.properties
```

### 3. 验证安装

```bash
# 查看 Kafka 进程
jps | grep Kafka
```

## 基本操作

### 创建 Topic

```bash
# 创建一个名为 quickstart-events 的 Topic
bin/kafka-topics.sh --create \
  --topic quickstart-events \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1
```

### 查看 Topic

```bash
# 列出所有 Topic
bin/kafka-topics.sh --list \
  --bootstrap-server localhost:9092

# 查看 Topic 详情
bin/kafka-topics.sh --describe \
  --topic quickstart-events \
  --bootstrap-server localhost:9092
```

### 发送消息

```bash
# 启动生产者控制台
bin/kafka-console-producer.sh \
  --topic quickstart-events \
  --bootstrap-server localhost:9092

# 输入消息（每行一条）
> Hello Kafka
> This is a test message
> Kafka is awesome
```

### 消费消息

```bash
# 启动消费者控制台（从最早的消息开始）
bin/kafka-console-consumer.sh \
  --topic quickstart-events \
  --from-beginning \
  --bootstrap-server localhost:9092
```

## Java 快速示例

### Maven 依赖

```xml
<dependency>
    <groupId>org.apache.kafka</groupId>
    <artifactId>kafka-clients</artifactId>
    <version>3.6.0</version>
</dependency>
```

### 生产者示例

```java
import org.apache.kafka.clients.producer.*;
import java.util.Properties;

public class SimpleProducer {
    public static void main(String[] args) {
        // 配置
        Properties props = new Properties();
        props.put("bootstrap.servers", "localhost:9092");
        props.put("key.serializer",
            "org.apache.kafka.common.serialization.StringSerializer");
        props.put("value.serializer",
            "org.apache.kafka.common.serialization.StringSerializer");

        // 创建生产者
        KafkaProducer<String, String> producer = new KafkaProducer<>(props);

        try {
            // 发送消息
            for (int i = 0; i < 10; i++) {
                String key = "key-" + i;
                String value = "message-" + i;

                ProducerRecord<String, String> record =
                    new ProducerRecord<>("quickstart-events", key, value);

                // 异步发送
                producer.send(record, (metadata, exception) -> {
                    if (exception == null) {
                        System.out.printf("发送成功: topic=%s, partition=%d, offset=%d%n",
                            metadata.topic(), metadata.partition(), metadata.offset());
                    } else {
                        exception.printStackTrace();
                    }
                });
            }
        } finally {
            producer.close();
        }
    }
}
```

### 消费者示例

```java
import org.apache.kafka.clients.consumer.*;
import java.time.Duration;
import java.util.Collections;
import java.util.Properties;

public class SimpleConsumer {
    public static void main(String[] args) {
        // 配置
        Properties props = new Properties();
        props.put("bootstrap.servers", "localhost:9092");
        props.put("group.id", "test-group");
        props.put("key.deserializer",
            "org.apache.kafka.common.serialization.StringDeserializer");
        props.put("value.deserializer",
            "org.apache.kafka.common.serialization.StringDeserializer");
        props.put("auto.offset.reset", "earliest");

        // 创建消费者
        KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);

        try {
            // 订阅 Topic
            consumer.subscribe(Collections.singletonList("quickstart-events"));

            // 持续拉取消息
            while (true) {
                ConsumerRecords<String, String> records =
                    consumer.poll(Duration.ofMillis(100));

                for (ConsumerRecord<String, String> record : records) {
                    System.out.printf("收到消息: key=%s, value=%s, partition=%d, offset=%d%n",
                        record.key(), record.value(),
                        record.partition(), record.offset());
                }
            }
        } finally {
            consumer.close();
        }
    }
}
```

## Docker 快速启动

### 使用 Docker Compose

创建 `docker-compose.yml`：

```yaml
version: "3"
services:
  kafka:
    image: apache/kafka:3.6.0
    container_name: kafka
    ports:
      - "9092:9092"
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: PLAINTEXT://localhost:9092,CONTROLLER://localhost:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@localhost:9093
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_LOG_DIRS: /tmp/kraft-combined-logs
      CLUSTER_ID: MkU3OEVBNTcwNTJENDM2Qk
```

启动服务：

```bash
# 启动
docker-compose up -d

# 查看日志
docker-compose logs -f kafka

# 停止
docker-compose down
```

## 常用管理命令

### Topic 管理

```bash
# 修改 Topic 分区数
bin/kafka-topics.sh --alter \
  --topic quickstart-events \
  --partitions 5 \
  --bootstrap-server localhost:9092

# 删除 Topic
bin/kafka-topics.sh --delete \
  --topic quickstart-events \
  --bootstrap-server localhost:9092
```

### 消费者组管理

```bash
# 查看所有消费者组
bin/kafka-consumer-groups.sh --list \
  --bootstrap-server localhost:9092

# 查看消费者组详情
bin/kafka-consumer-groups.sh --describe \
  --group test-group \
  --bootstrap-server localhost:9092

# 重置消费位移
bin/kafka-consumer-groups.sh --reset-offsets \
  --group test-group \
  --topic quickstart-events \
  --to-earliest \
  --bootstrap-server localhost:9092 \
  --execute
```

### 性能测试

```bash
# 生产者性能测试
bin/kafka-producer-perf-test.sh \
  --topic test-topic \
  --num-records 1000000 \
  --record-size 1000 \
  --throughput -1 \
  --producer-props bootstrap.servers=localhost:9092

# 消费者性能测试
bin/kafka-consumer-perf-test.sh \
  --topic test-topic \
  --messages 1000000 \
  --bootstrap-server localhost:9092
```

## 故障排查

### 检查 Kafka 状态

```bash
# 查看 Kafka 进程
ps aux | grep kafka

# 查看端口占用
netstat -tulpn | grep 9092

# 查看日志
tail -f logs/server.log
```

### 常见问题

#### 1. 连接被拒绝

```bash
# 检查 Kafka 是否启动
jps | grep Kafka

# 检查配置文件中的监听地址
grep listeners config/kraft/server.properties
```

#### 2. 磁盘空间不足

```bash
# 清理旧日志
bin/kafka-log-dirs.sh --describe \
  --bootstrap-server localhost:9092

# 修改日志保留策略
bin/kafka-configs.sh --alter \
  --bootstrap-server localhost:9092 \
  --entity-type topics \
  --entity-name quickstart-events \
  --add-config retention.ms=86400000
```

## 下一步

- 📖 [核心概念](./core-concepts.md) - 深入理解 Kafka 架构
- 💻 [生产者 API](./producer-api.md) - 学习生产者高级用法
- 📊 [消费者 API](./consumer-api.md) - 学习消费者高级用法
- ⚙️ [集群管理](./cluster-management.md) - 了解如何管理 Kafka 集群

## 参考资料

- [Kafka 快速开始官方文档](https://kafka.apache.org/quickstart)
- [Kafka Docker 镜像](https://hub.docker.com/r/apache/kafka)
