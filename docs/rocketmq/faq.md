---
sidebar_position: 9
title: "常见问题"
description: "RocketMQ 常见问题解答"
---

# RocketMQ 常见问题

## 基础问题

### Q1: RocketMQ 和 Kafka 有什么区别？

| 特性       | RocketMQ       | Kafka       |
| ---------- | -------------- | ----------- |
| 开发语言   | Java           | Scala/Java  |
| 事务消息   | 原生支持       | 0.11+ 支持  |
| 延迟消息   | 原生支持       | 需自己实现  |
| 消息过滤   | Tag/SQL92      | 不支持      |
| 消息回溯   | 按时间         | 按 Offset   |
| 管理控制台 | 内置 Dashboard | 需第三方    |
| 适用场景   | 电商/金融      | 大数据/日志 |

### Q2: RocketMQ 的消息可靠性如何保证？

**发送端：**

- 同步发送 + 重试机制
- 发送失败自动切换 Broker

**存储端：**

- 同步刷盘保证不丢消息
- 主从同步复制

**消费端：**

- ACK 机制，消费成功才更新 Offset
- 消费失败自动重试
- 死信队列兜底

### Q3: 消息发送失败怎么处理？

```java
// 方法 1：同步发送 + 检查结果
SendResult result = producer.send(msg);
if (result.getSendStatus() != SendStatus.SEND_OK) {
    // 记录日志，人工处理
}

// 方法 2：异步发送 + 回调处理
producer.send(msg, new SendCallback() {
    @Override
    public void onException(Throwable e) {
        // 记录失败消息，后续重发
        saveFailedMessage(msg);
    }
});

// 方法 3：增加重试次数
producer.setRetryTimesWhenSendFailed(5);
```

### Q4: 如何保证消息不重复消费？

消费端需要实现**幂等性**：

```java
// 方法 1：数据库唯一键
try {
    orderDao.insert(order);  // 主键/唯一索引冲突会失败
} catch (DuplicateKeyException e) {
    // 重复消息，直接返回成功
    return ConsumeConcurrentlyStatus.CONSUME_SUCCESS;
}

// 方法 2：Redis 去重
String msgId = msg.getMsgId();
if (!redis.setNx("consumed:" + msgId, "1", 24, TimeUnit.HOURS)) {
    // 已消费过
    return ConsumeConcurrentlyStatus.CONSUME_SUCCESS;
}

// 方法 3：业务状态判断
Order order = orderDao.findById(orderId);
if (order.getStatus().equals("paid")) {
    // 已处理过
    return ConsumeConcurrentlyStatus.CONSUME_SUCCESS;
}
```

## 连接问题

### Q5: 连接 NameServer 失败怎么办？

**排查步骤：**

```bash
# 1. 检查 NameServer 是否启动
jps | grep NamesrvStartup

# 2. 检查端口是否监听
netstat -tlnp | grep 9876

# 3. 检查防火墙
firewall-cmd --list-ports

# 4. 检查网络连通性
telnet localhost 9876
```

**常见原因：**

- NameServer 未启动
- 防火墙阻止了端口
- 配置的地址不正确

### Q6: 发送消息超时怎么解决？

```java
// 1. 增加超时时间
producer.setSendMsgTimeout(10000);

// 2. 检查 Broker 负载
sh bin/mqadmin clusterList -n localhost:9876

// 3. 检查网络延迟
ping broker-host
```

### Q7: 消费者收不到消息？

**排查步骤：**

1. 确认 Topic 和 ConsumerGroup 正确
2. 检查订阅的 Tag 是否匹配
3. 查看消费进度

```bash
# 查看消费进度
sh bin/mqadmin consumerProgress -n localhost:9876 -g ConsumerGroup
```

4. 检查消费者是否正常注册

```bash
# 查看消费者连接
sh bin/mqadmin consumerConnection -n localhost:9876 -g ConsumerGroup
```

## 性能问题

### Q8: 如何提高发送性能？

```java
// 1. 使用异步发送
producer.send(msg, callback);

// 2. 增加发送线程
// 多线程并发调用 send

// 3. 开启批量发送
List<Message> messages = new ArrayList<>();
producer.send(messages);

// 4. 调整批次参数
producer.setCompressNumOfMsgsOver(100);
```

### Q9: 如何提高消费性能？

```java
// 1. 增加消费线程
consumer.setConsumeThreadMin(20);
consumer.setConsumeThreadMax(64);

// 2. 增加每次拉取数量
consumer.setPullBatchSize(32);

// 3. 增加批量消费数量
consumer.setConsumeMessageBatchMaxSize(10);

// 4. 增加消费者实例数量（不超过 Queue 数量）
```

### Q10: 消息堆积怎么处理？

**临时方案：**

```java
// 扩容消费者实例
// 临时增加 Topic 的 Queue 数量
sh bin/mqadmin updateTopic -n localhost:9876 -t TopicTest -r 16 -w 16
```

**长期方案：**

- 优化消费逻辑，减少耗时
- 异步处理耗时操作
- 使用多线程消费

## 功能问题

### Q11: 顺序消息如何保证顺序？

**发送端：**同一业务 Key 的消息发送到同一 Queue

```java
producer.send(msg, (mqs, message, arg) -> {
    int index = Math.abs(arg.hashCode() % mqs.size());
    return mqs.get(index);
}, orderId);
```

**消费端：**使用 `MessageListenerOrderly`

```java
consumer.registerMessageListener((MessageListenerOrderly) (msgs, ctx) -> {
    return ConsumeOrderlyStatus.SUCCESS;
});
```

### Q12: 延迟消息的自定义时间如何实现？

开源版本只支持固定的 18 个级别，自定义时间需要：

1. **修改 Broker 配置**（不推荐）
2. **使用定时轮询**（推荐）

```java
// 方案：使用最近的延迟级别 + 消费时判断
public void sendScheduledMessage(String content, long targetTime) {
    long delay = targetTime - System.currentTimeMillis();

    // 找到最接近的延迟级别
    int level = calculateDelayLevel(delay);

    Message msg = new Message("ScheduleTopic", content.getBytes());
    msg.setDelayTimeLevel(level);
    msg.putUserProperty("targetTime", String.valueOf(targetTime));

    producer.send(msg);
}

// 消费时判断
if (System.currentTimeMillis() < targetTime) {
    // 还没到时间，重新发送
    resend(msg);
    return ConsumeConcurrentlyStatus.CONSUME_SUCCESS;
}
```

### Q13: 事务消息回查失败怎么办？

- 确保事务监听器的 `checkLocalTransaction` 方法正确实现
- 回查最多 15 次，超过后消息会被丢弃
- 建议使用**本地事务表**记录事务状态

```java
@Override
public LocalTransactionState checkLocalTransaction(MessageExt msg) {
    String txId = msg.getTransactionId();
    TransactionLog log = txLogDao.findByTxId(txId);

    if (log == null) {
        return LocalTransactionState.UNKNOW;
    }

    switch (log.getStatus()) {
        case "COMMITTED":
            return LocalTransactionState.COMMIT_MESSAGE;
        case "ROLLED_BACK":
            return LocalTransactionState.ROLLBACK_MESSAGE;
        default:
            return LocalTransactionState.UNKNOW;
    }
}
```

## 运维问题

### Q14: 如何监控 RocketMQ？

1. **Dashboard 控制台**：http://localhost:8080
2. **Prometheus + Grafana**
3. **日志监控**

```bash
# 查看日志
tail -f ~/logs/rocketmqlogs/broker.log
```

### Q15: 消息丢失如何排查？

1. **检查发送端日志**

```java
SendResult result = producer.send(msg);
log.info("发送结果: {}", result);
```

2. **查询消息轨迹**

```bash
sh bin/mqadmin queryMsgById -n localhost:9876 -i <msgId>
```

3. **检查消费进度**

```bash
sh bin/mqadmin consumerProgress -n localhost:9876 -g ConsumerGroup
```

### Q16: 如何平滑扩容？

1. 增加 Broker 节点
2. 增加 Topic 的 Queue 数量
3. 等待负载均衡完成

```bash
# 增加 Queue 数量
sh bin/mqadmin updateTopic -n localhost:9876 -t TopicTest -r 16 -w 16
```

## 参考资料

- [RocketMQ 官方 FAQ](https://rocketmq.apache.org/docs/faq/)
- [RocketMQ 最佳实践](https://rocketmq.apache.org/docs/bestPractice/)
