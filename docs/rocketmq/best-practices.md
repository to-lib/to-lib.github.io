---
sidebar_position: 11
title: "最佳实践"
description: "RocketMQ 生产环境最佳实践指南"
---

# RocketMQ 最佳实践

本文档总结了 RocketMQ 在生产环境中的最佳实践，帮助你构建高可用、高性能的消息系统。

## 生产者最佳实践

### 1. 合理设置 Producer Group

```java
// 同一个应用使用同一个 ProducerGroup
DefaultMQProducer producer = new DefaultMQProducer("OrderService_Producer");

// 不要每次发送都创建新的生产者
// ❌ 错误示例
for (int i = 0; i < 1000; i++) {
    DefaultMQProducer producer = new DefaultMQProducer("Group_" + i);
    producer.start();
    producer.send(msg);
    producer.shutdown();  // 频繁创建销毁
}

// ✅ 正确示例
DefaultMQProducer producer = new DefaultMQProducer("OrderService_Producer");
producer.start();
for (int i = 0; i < 1000; i++) {
    producer.send(msg);
}
```

### 2. 选择合适的发送方式

| 发送方式 | 适用场景                   | 示例       |
| -------- | -------------------------- | ---------- |
| 同步发送 | 重要消息，需要确认结果     | 订单、支付 |
| 异步发送 | 响应时间敏感，允许回调处理 | 通知、日志 |
| 单向发送 | 不关心发送结果             | 日志采集   |

```java
// 重要业务使用同步发送
SendResult result = producer.send(msg);
if (result.getSendStatus() == SendStatus.SEND_OK) {
    // 处理成功逻辑
}

// 高吞吐场景使用异步发送
producer.send(msg, new SendCallback() {
    @Override
    public void onSuccess(SendResult result) {
        // 异步处理成功
    }
    @Override
    public void onException(Throwable e) {
        // 记录失败，后续重试
        saveFailedMessage(msg);
    }
});
```

### 3. 消息 Key 设计

```java
// ✅ 使用有意义的业务 Key
msg.setKeys("ORDER_" + orderId);

// ✅ 多个 Key 用空格分隔
msg.setKeys("ORDER_123 USER_456 PRODUCT_789");

// ❌ 避免使用无意义的 Key
msg.setKeys(UUID.randomUUID().toString());
```

**Key 的作用：**

- 消息查询和追踪
- 顺序消息的路由依据
- 故障排查的关键线索

### 4. 消息体设计

```java
// ✅ 使用紧凑的序列化格式
// JSON（可读性好）或 Protobuf（性能好）
String json = JSON.toJSONString(order);
msg.setBody(json.getBytes(StandardCharsets.UTF_8));

// ✅ 控制消息大小（建议 < 1MB）
if (body.length > 1024 * 1024) {
    // 考虑分片或存储到 OSS
    String fileUrl = uploadToOSS(largeData);
    msg.setBody(fileUrl.getBytes());
}
```

### 5. 重试与超时配置

```java
// 发送超时（根据网络情况调整）
producer.setSendMsgTimeout(5000);

// 同步发送重试次数
producer.setRetryTimesWhenSendFailed(3);

// 异步发送重试次数
producer.setRetryTimesWhenSendAsyncFailed(3);

// 发送失败时切换 Broker
producer.setRetryAnotherBrokerWhenNotStoreOK(true);
```

## 消费者最佳实践

### 1. 消费幂等性

消息可能重复投递，必须实现幂等：

```java
// 方法1：数据库唯一键
@Transactional
public void processOrder(String orderId) {
    try {
        orderDao.insert(order);  // 唯一键冲突会抛异常
    } catch (DuplicateKeyException e) {
        log.info("订单已处理: {}", orderId);
        return;
    }
}

// 方法2：Redis 去重
public boolean tryConsume(String msgId) {
    Boolean success = redis.opsForValue()
        .setIfAbsent("consumed:" + msgId, "1", 24, TimeUnit.HOURS);
    return Boolean.TRUE.equals(success);
}

// 方法3：业务状态检查
public void processPayment(String orderId) {
    Order order = orderDao.findById(orderId);
    if (order.getStatus() == OrderStatus.PAID) {
        log.info("订单已支付: {}", orderId);
        return;
    }
    // 处理支付逻辑
}
```

### 2. 消费线程池配置

```java
// 根据业务特点配置线程数
// CPU 密集型：线程数 = CPU 核心数 + 1
// IO 密集型：线程数 = CPU 核心数 * 2

consumer.setConsumeThreadMin(20);
consumer.setConsumeThreadMax(64);

// 每次消费的消息数量
consumer.setConsumeMessageBatchMaxSize(1);  // 默认 1

// 批量消费场景
consumer.setConsumeMessageBatchMaxSize(10);
```

### 3. 消费失败处理

```java
consumer.registerMessageListener((MessageListenerConcurrently) (msgs, context) -> {
    for (MessageExt msg : msgs) {
        int reconsumeTimes = msg.getReconsumeTimes();

        // 多次重试失败，人工介入
        if (reconsumeTimes >= 3) {
            log.error("消费失败超过3次: {}", msg.getMsgId());
            saveToDeadLetterDB(msg);  // 保存到数据库
            alertService.send("消息消费失败告警", msg);
            return ConsumeConcurrentlyStatus.CONSUME_SUCCESS;  // 不再重试
        }

        try {
            processMessage(msg);
        } catch (Exception e) {
            log.error("消费失败，将重试", e);
            return ConsumeConcurrentlyStatus.RECONSUME_LATER;
        }
    }
    return ConsumeConcurrentlyStatus.CONSUME_SUCCESS;
});
```

### 4. 消费位点管理

```java
// 首次消费位置
// CONSUME_FROM_LAST_OFFSET: 从最新消息开始（推荐）
// CONSUME_FROM_FIRST_OFFSET: 从头开始
consumer.setConsumeFromWhere(ConsumeFromWhere.CONSUME_FROM_LAST_OFFSET);

// 指定时间开始消费
consumer.setConsumeFromWhere(ConsumeFromWhere.CONSUME_FROM_TIMESTAMP);
consumer.setConsumeTimestamp("20240101120000");
```

### 5. 优雅停机

```java
// 添加关闭钩子
Runtime.getRuntime().addShutdownHook(new Thread(() -> {
    log.info("正在关闭消费者...");
    consumer.shutdown();
    log.info("消费者已关闭");
}));

// Spring Boot 中使用 @PreDestroy
@PreDestroy
public void destroy() {
    consumer.shutdown();
}
```

## Topic 设计规范

### 1. 命名规范

```
# 格式：业务域_应用名_功能_环境
order_service_create_prod
payment_service_callback_dev

# 避免使用
TopicTest          # 无意义
myTopic            # 不规范
order-create       # 使用下划线而非连字符
```

### 2. Topic vs Tag 选择

| 场景             | 建议           |
| ---------------- | -------------- |
| 完全不同的业务   | 使用不同 Topic |
| 同一业务不同类型 | 使用不同 Tag   |
| 需要隔离的数据   | 使用不同 Topic |
| 仅需过滤的数据   | 使用不同 Tag   |

```java
// ✅ 订单的不同状态使用 Tag
Message createMsg = new Message("OrderTopic", "create", body);
Message payMsg = new Message("OrderTopic", "pay", body);
Message shipMsg = new Message("OrderTopic", "ship", body);

// ✅ 不同业务使用不同 Topic
Message orderMsg = new Message("OrderTopic", "create", body);
Message paymentMsg = new Message("PaymentTopic", "success", body);
```

### 3. Queue 数量规划

```bash
# Queue 数量 >= 消费者实例数
# 建议：Queue 数量 = 消费者实例数 * 2（预留扩展空间）

# 创建 Topic 时指定 Queue 数量
sh bin/mqadmin updateTopic -n localhost:9876 -t OrderTopic -r 8 -w 8
```

## 高可用部署建议

### 1. NameServer 部署

```bash
# 至少部署 2 个 NameServer
# 各节点配置相同，无状态

# 生产者/消费者配置多个 NameServer
namesrvAddr=192.168.1.1:9876;192.168.1.2:9876
```

### 2. Broker 部署

```
# 推荐：2 Master + 2 Slave 架构
Broker-a-master (192.168.1.1)
  └── Broker-a-slave (192.168.1.2)
Broker-b-master (192.168.1.3)
  └── Broker-b-slave (192.168.1.4)
```

**Broker 配置建议：**

```properties
# broker.conf
brokerClusterName=DefaultCluster
brokerName=broker-a
brokerId=0
namesrvAddr=192.168.1.1:9876;192.168.1.2:9876

# 同步刷盘（金融场景）
flushDiskType=SYNC_FLUSH

# 同步复制（高可靠）
brokerRole=SYNC_MASTER

# 消息保留时间（小时）
fileReservedTime=72

# 删除过期文件时间点
deleteWhen=04
```

### 3. 客户端容错

```java
// 生产者容错
producer.setRetryTimesWhenSendFailed(3);
producer.setRetryAnotherBrokerWhenNotStoreOK(true);

// 消费者容错
consumer.setMaxReconsumeTimes(16);
```

## 常见陷阱与避免

### 1. 消息堆积

**原因：** 消费速度 < 生产速度

**解决：**

```java
// 增加消费者实例（不超过 Queue 数量）
// 增加消费线程
consumer.setConsumeThreadMax(64);

// 批量消费
consumer.setConsumeMessageBatchMaxSize(10);
```

### 2. 消息丢失

**原因：** 异步刷盘 + 异步复制

**解决：**

```properties
# 同步刷盘
flushDiskType=SYNC_FLUSH

# 同步复制
brokerRole=SYNC_MASTER
```

### 3. 消息重复

**原因：** 网络抖动导致重复投递

**解决：** 消费端实现幂等（见上文）

### 4. 顺序消息消费卡住

**原因：** 顺序消费时某条消息持续失败

**解决：**

```java
consumer.registerMessageListener((MessageListenerOrderly) (msgs, ctx) -> {
    for (MessageExt msg : msgs) {
        if (msg.getReconsumeTimes() >= 3) {
            // 记录后跳过，避免卡住队列
            logFailedMessage(msg);
            return ConsumeOrderlyStatus.SUCCESS;
        }
        // 正常处理
    }
    return ConsumeOrderlyStatus.SUCCESS;
});
```

### 5. 事务消息回查失败

**原因：** 回查逻辑异常或本地事务状态丢失

**解决：**

```java
// 使用本地事务表记录状态
CREATE TABLE transaction_log (
    tx_id VARCHAR(64) PRIMARY KEY,
    status VARCHAR(16),
    create_time TIMESTAMP
);

// 回查时查询事务表
@Override
public LocalTransactionState checkLocalTransaction(MessageExt msg) {
    String txId = msg.getTransactionId();
    TransactionLog log = txLogDao.findByTxId(txId);

    if (log == null) {
        return LocalTransactionState.UNKNOW;
    }
    return "COMMITTED".equals(log.getStatus())
        ? LocalTransactionState.COMMIT_MESSAGE
        : LocalTransactionState.ROLLBACK_MESSAGE;
}
```

## 安全配置

### 1. ACL 访问控制

```properties
# broker.conf
aclEnable=true

# plain_acl.yml
accounts:
  - accessKey: admin
    secretKey: admin123
    admin: true
  - accessKey: producer
    secretKey: producer123
    defaultTopicPerm: PUB
  - accessKey: consumer
    secretKey: consumer123
    defaultTopicPerm: SUB
```

### 2. 客户端配置

```java
// 生产者
DefaultMQProducer producer = new DefaultMQProducer("ProducerGroup",
    new AclClientRPCHook(new SessionCredentials("producer", "producer123")));

// 消费者
DefaultMQPushConsumer consumer = new DefaultMQPushConsumer("ConsumerGroup",
    new AclClientRPCHook(new SessionCredentials("consumer", "consumer123")));
```

## 日志与监控

### 1. 关键日志配置

```xml
<!-- logback.xml -->
<logger name="RocketmqClient" level="WARN"/>
<logger name="RocketmqRemoting" level="WARN"/>
<logger name="RocketmqCommon" level="WARN"/>
```

### 2. 业务日志

```java
// 发送时记录
SendResult result = producer.send(msg);
log.info("发送消息: topic={}, msgId={}, status={}",
    msg.getTopic(), result.getMsgId(), result.getSendStatus());

// 消费时记录
log.info("消费消息: topic={}, msgId={}, reconsumeTimes={}",
    msg.getTopic(), msg.getMsgId(), msg.getReconsumeTimes());
```

## 下一步

- 🏗️ [集群管理](/docs/rocketmq/cluster-management) - 深入了解集群部署
- ⚡ [性能优化](/docs/rocketmq/performance-optimization) - 提升系统性能
- 📊 [监控运维](/docs/rocketmq/monitoring) - 建设监控体系

## 参考资料

- [RocketMQ 官方最佳实践](https://rocketmq.apache.org/docs/bestPractice/)
- [阿里云 RocketMQ 实践](https://help.aliyun.com/document_detail/29532.html)
