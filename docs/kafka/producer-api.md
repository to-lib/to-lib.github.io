---
sidebar_position: 5
title: "生产者 API"
description: "深入学习 Kafka 生产者 API"
---

# Kafka 生产者 API

## 生产者概述

Kafka Producer 负责将消息发布到 Kafka Topic。生产者会将消息发送到指定的分区，并可以配置各种参数来控制性能和可靠性。

## 基本配置

### 必需配置

```java
Properties props = new Properties();

// Kafka 集群地址
props.put("bootstrap.servers", "localhost:9092");

// Key 序列化器
props.put("key.serializer",
    "org.apache.kafka.common.serialization.StringSerializer");

// Value 序列化器
props.put("value.serializer",
    "org.apache.kafka.common.serialization.StringSerializer");
```

### 创建生产者

```java
KafkaProducer<String, String> producer = new KafkaProducer<>(props);
```

## 发送消息的方式

### 1. 发送并忘记（Fire-and-Forget）

```java
public void fireAndForget() {
    ProducerRecord<String, String> record =
        new ProducerRecord<>("my-topic", "key", "value");

    try {
        producer.send(record); // 不关心结果
    } catch (Exception e) {
        // 只会捕获不可重试的异常
        e.printStackTrace();
    }
}
```

**特点：**

- ✅ 最高性能
- ❌ 可能丢失消息
- ❌ 不知道发送结果

### 2. 同步发送

```java
public void sendSync() throws Exception {
    ProducerRecord<String, String> record =
        new ProducerRecord<>("my-topic", "key", "value");

    try {
        // get() 会阻塞等待结果
        RecordMetadata metadata = producer.send(record).get();
        System.out.printf("发送成功: topic=%s, partition=%d, offset=%d%n",
            metadata.topic(), metadata.partition(), metadata.offset());
    } catch (ExecutionException e) {
        // 处理发送失败
        e.printStackTrace();
    }
}
```

**特点：**

- ✅ 可靠性高
- ✅ 知道发送结果
- ❌ 性能较低（阻塞等待）

### 3. 异步发送（推荐）

```java
public void sendAsync() {
    ProducerRecord<String, String> record =
        new ProducerRecord<>("my-topic", "key", "value");

    producer.send(record, new Callback() {
        @Override
        public void onCompletion(RecordMetadata metadata, Exception exception) {
            if (exception == null) {
                System.out.printf("发送成功: topic=%s, partition=%d, offset=%d%n",
                    metadata.topic(), metadata.partition(), metadata.offset());
            } else {
                exception.printStackTrace();
            }
        }
    });
}

// 使用 Lambda 表达式
public void sendAsyncLambda() {
    ProducerRecord<String, String> record =
        new ProducerRecord<>("my-topic", "key", "value");

    producer.send(record, (metadata, exception) -> {
        if (exception == null) {
            System.out.println("发送成功: " + metadata.offset());
        } else {
            exception.printStackTrace();
        }
    });
}
```

**特点：**

- ✅ 高性能
- ✅ 知道发送结果
- ✅ 不阻塞（推荐使用）

## 重要配置参数

### 性能相关

```java
// 批次大小（默认 16KB）
props.put("batch.size", 16384);

// 等待时间（默认 0ms）
props.put("linger.ms", 10);

// 缓冲区大小（默认 32MB）
props.put("buffer.memory", 33554432);

// 压缩类型
props.put("compression.type", "lz4");  // none, gzip, snappy, lz4, zstd

// 最大请求大小
props.put("max.request.size", 1048576);
```

### 可靠性相关

```java
// ACK 确认机制
props.put("acks", "all");  // 0, 1, all(-1)

// 重试次数
props.put("retries", Integer.MAX_VALUE);

// 重试间隔
props.put("retry.backoff.ms", 100);

// 幂等性（防止重复）
props.put("enable.idempotence", "true");

// 事务 ID（用于事务性发送）
props.put("transactional.id", "my-transactional-id");

// 最大飞行中请求数（保证顺序）
props.put("max.in.flight.requests.per.connection", 5);
```

## ACK 确认机制详解

| acks        | 说明                  | 延迟 | 吞吐量 | 可靠性 | 使用场景             |
| ----------- | --------------------- | ---- | ------ | ------ | -------------------- |
| **0**       | 不等待任何确认        | 最低 | 最高   | 最低   | 日志收集、非关键数据 |
| **1**       | 等待 Leader 确认      | 中等 | 中等   | 中等   | 一般业务消息         |
| **all(-1)** | 等待所有 ISR 副本确认 | 最高 | 最低   | 最高   | 金融交易、关键数据   |

```java
// acks=0: 发送后立即返回
props.put("acks", "0");

// acks=1: Leader 写入成功后返回
props.put("acks", "1");

// acks=all: 所有 ISR 副本写入成功后返回
props.put("acks", "all");
props.put("min.insync.replicas", "2");  // 至少 2 个副本确认
```

## 分区策略

### 默认分区器

```java
// 1. 指定分区
ProducerRecord<String, String> record =
    new ProducerRecord<>("my-topic", 0, "key", "value");

// 2. 指定 key（根据 key 的哈希值分配分区）
ProducerRecord<String, String> record =
    new ProducerRecord<>("my-topic", "key", "value");

// 3. 不指定 key（轮询分配）
ProducerRecord<String, String> record =
    new ProducerRecord<>("my-topic", "value");
```

### 自定义分区器

```java
public class CustomPartitioner implements Partitioner {

    @Override
    public int partition(String topic, Object key, byte[] keyBytes,
                        Object value, byte[] valueBytes, Cluster cluster) {
        List<PartitionInfo> partitions = cluster.partitionsForTopic(topic);
        int numPartitions = partitions.size();

        if (key == null) {
            // 没有 key 时的处理
            return ThreadLocalRandom.current().nextInt(numPartitions);
        }

        // 自定义分区逻辑
        if (key.toString().startsWith("VIP")) {
            return 0; // VIP 用户发送到分区 0
        }

        return Math.abs(key.hashCode()) % numPartitions;
    }

    @Override
    public void close() {}

    @Override
    public void configure(Map<String, ?> configs) {}
}

// 使用自定义分区器
props.put("partitioner.class", "com.example.CustomPartitioner");
```

## 序列化器

### 内置序列化器

```java
// String 序列化器
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");

// Integer 序列化器
props.put("value.serializer", "org.apache.kafka.common.serialization.IntegerSerializer");

// Long 序列化器
props.put("value.serializer", "org.apache.kafka.common.serialization.LongSerializer");

// ByteArray 序列化器
props.put("value.serializer", "org.apache.kafka.common.serialization.ByteArraySerializer");
```

### JSON 序列化器

```java
import com.fasterxml.jackson.databind.ObjectMapper;

public class JsonSerializer<T> implements Serializer<T> {
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Override
    public byte[] serialize(String topic, T data) {
        if (data == null) {
            return null;
        }
        try {
            return objectMapper.writeValueAsBytes(data);
        } catch (Exception e) {
            throw new SerializationException("Error serializing JSON message", e);
        }
    }
}

// 使用
props.put("value.serializer", "com.example.JsonSerializer");
```

### Avro 序列化器

```java
// 使用 Confluent Schema Registry
props.put("value.serializer",
    "io.confluent.kafka.serializers.KafkaAvroSerializer");
props.put("schema.registry.url", "http://localhost:8081");
```

## 消息头（Headers）

```java
ProducerRecord<String, String> record =
    new ProducerRecord<>("my-topic", "key", "value");

// 添加消息头
record.headers()
    .add("correlation-id", "12345".getBytes())
    .add("source", "payment-service".getBytes())
    .add("timestamp", String.valueOf(System.currentTimeMillis()).getBytes());

producer.send(record);
```

## 拦截器

```java
public class ProducerInterceptorDemo implements ProducerInterceptor<String, String> {

    @Override
    public ProducerRecord<String, String> onSend(ProducerRecord<String, String> record) {
        // 发送前拦截
        System.out.println("准备发送: " + record.value());

        // 可以修改消息
        return new ProducerRecord<>(
            record.topic(),
            record.key(),
            record.value() + " [intercepted]"
        );
    }

    @Override
    public void onAcknowledgement(RecordMetadata metadata, Exception exception) {
        // 收到确认后调用
        if (exception == null) {
            System.out.println("发送成功: partition=" + metadata.partition());
        } else {
            System.err.println("发送失败: " + exception.getMessage());
        }
    }

    @Override
    public void close() {}

    @Override
    public void configure(Map<String, ?> configs) {}
}

// 配置拦截器
props.put("interceptor.classes",
    "com.example.ProducerInterceptorDemo");
```

## 事务性发送

```java
public class TransactionalProducer {

    public static void main(String[] args) {
        Properties props = new Properties();
        props.put("bootstrap.servers", "localhost:9092");
        props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");

        // 开启事务
        props.put("enable.idempotence", "true");
        props.put("transactional.id", "my-transaction-id");

        KafkaProducer<String, String> producer = new KafkaProducer<>(props);

        // 初始化事务
        producer.initTransactions();

        try {
            // 开始事务
            producer.beginTransaction();

            // 发送消息
            producer.send(new ProducerRecord<>("topic1", "key1", "value1"));
            producer.send(new ProducerRecord<>("topic2", "key2", "value2"));

            // 提交事务
            producer.commitTransaction();

        } catch (Exception e) {
            // 回滚事务
            producer.abortTransaction();
            e.printStackTrace();
        } finally {
            producer.close();
        }
    }
}
```

## 最佳实践

### 1. 使用异步发送 + 回调

```java
for (int i = 0; i < 1000; i++) {
    final int index = i;
    producer.send(
        new ProducerRecord<>("my-topic", "key-" + i, "value-" + i),
        (metadata, exception) -> {
            if (exception != null) {
                // 记录失败的消息，后续重试
                System.err.println("发送失败: index=" + index);
            }
        }
    );
}
```

### 2. 正确关闭生产者

```java
try {
    // 发送消息
    producer.send(record);
} finally {
    // 确保所有消息发送完成
    producer.flush();
    // 关闭生产者
    producer.close();
}
```

### 3. 合理配置批次大小和延迟

```java
// 高吞吐量场景
props.put("batch.size", 32768);      // 增大批次
props.put("linger.ms", 20);          // 增加等待时间
props.put("compression.type", "lz4"); // 启用压缩

// 低延迟场景
props.put("batch.size", 0);          // 不批量
props.put("linger.ms", 0);           // 立即发送
props.put("compression.type", "none"); // 不压缩
```

### 4. 开启幂等性

```java
// 防止消息重复
props.put("enable.idempotence", "true");
// 此时以下配置会自动设置：
// acks=all
// retries=Integer.MAX_VALUE
// max.in.flight.requests.per.connection=5
```

## 性能优化

### 批量发送

```java
props.put("batch.size", 16384);    // 16KB
props.put("linger.ms", 10);        // 等待 10ms
```

### 压缩

```java
// lz4 平衡了压缩率和 CPU 消耗
props.put("compression.type", "lz4");
```

### 增加缓冲区

```java
props.put("buffer.memory", 67108864); // 64MB
```

## 下一步

- 📊 [消费者 API](./consumer-api.md) - 学习消息消费
- 🔧 [集群管理](./cluster-management.md) - 了解集群管理
- ⚡ [性能优化](./performance-optimization.md) - 深入性能优化

## 参考资料

- [Producer API 官方文档](https://kafka.apache.org/documentation/#producerapi)
- [Producer Configuration](https://kafka.apache.org/documentation/#producerconfigs)
