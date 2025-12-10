---
sidebar_position: 10
title: "快速参考"
description: "Kafka 常用命令和配置速查"
---

# Kafka 快速参考

## 常用命令

### Topic 管理

```bash
# 创建 Topic
kafka-topics.sh --create --topic my-topic \
  --bootstrap-server localhost:9092 \
  --partitions 3 --replication-factor 2

# 列出所有 Topic
kafka-topics.sh --list --bootstrap-server localhost:9092

# 查看 Topic 详情
kafka-topics.sh --describe --topic my-topic \
  --bootstrap-server localhost:9092

# 修改分区数
kafka-topics.sh --alter --topic my-topic \
  --partitions 5 --bootstrap-server localhost:9092

# 删除 Topic
kafka-topics.sh --delete --topic my-topic \
  --bootstrap-server localhost:9092
```

### 生产和消费

```bash
# 控制台生产者
kafka-console-producer.sh --topic my-topic \
  --bootstrap-server localhost:9092

# 控制台消费者
kafka-console-consumer.sh --topic my-topic \
  --from-beginning --bootstrap-server localhost:9092

# 指定消费者组
kafka-console-consumer.sh --topic my-topic \
  --group my-group --bootstrap-server localhost:9092
```

### 消费者组管理

```bash
# 列出所有消费者组
kafka-consumer-groups.sh --list \
  --bootstrap-server localhost:9092

# 查看消费者组详情
kafka-consumer-groups.sh --describe \
  --group my-group --bootstrap-server localhost:9092

# 重置位移到最早
kafka-consumer-groups.sh --reset-offsets --to-earliest \
  --group my-group --topic my-topic \
  --bootstrap-server localhost:9092 --execute
```

## 生产者配置速查

| 参数                 | 说明           | 默认值 | 推荐值              |
| -------------------- | -------------- | ------ | ------------------- |
| `bootstrap.servers`  | Kafka 集群地址 | -      | `localhost:9092`    |
| `acks`               | 确认级别       | 1      | `all`               |
| `batch.size`         | 批次大小       | 16384  | `32768`             |
| `linger.ms`          | 等待时间       | 0      | `10-20`             |
| `compression.type`   | 压缩类型       | none   | `lz4`               |
| `retries`            | 重试次数       | 0      | `Integer.MAX_VALUE` |
| `enable.idempotence` | 幂等性         | false  | `true`              |

## 消费者配置速查

| 参数                 | 说明           | 默认值 | 推荐值            |
| -------------------- | -------------- | ------ | ----------------- |
| `bootstrap.servers`  | Kafka 集群地址 | -      | `localhost:9092`  |
| `group.id`           | 消费者组 ID    | -      | 自定义            |
| `enable.auto.commit` | 自动提交       | true   | `false`           |
| `auto.offset.reset`  | 位移重置策略   | latest | `earliest/latest` |
| `max.poll.records`   | 单次拉取数量   | 500    | `100-500`         |
| `session.timeout.ms` | 会话超时       | 10000  | `30000`           |

## Java 代码模板

### 简单生产者

```java
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");

KafkaProducer<String, String> producer = new KafkaProducer<>(props);
producer.send(new ProducerRecord<>("topic", "key", "value"));
producer.close();
```

### 简单消费者

```java
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("group.id", "test-group");
props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
props.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
consumer.subscribe(Collections.singletonList("topic"));
ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
consumer.close();
```

## 参考链接

- [官方文档](https://kafka.apache.org/documentation/)
- [配置参考](https://kafka.apache.org/documentation/#configuration)
