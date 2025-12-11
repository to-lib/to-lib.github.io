---
sidebar_position: 8
title: "快速参考"
description: "RocketMQ 常用 API 和命令速查"
---

# RocketMQ 快速参考

## 常用依赖

### Maven

```xml
<!-- RocketMQ 客户端 -->
<dependency>
    <groupId>org.apache.rocketmq</groupId>
    <artifactId>rocketmq-client</artifactId>
    <version>5.1.4</version>
</dependency>

<!-- Spring Boot Starter -->
<dependency>
    <groupId>org.apache.rocketmq</groupId>
    <artifactId>rocketmq-spring-boot-starter</artifactId>
    <version>2.2.3</version>
</dependency>
```

## 生产者 API

### 创建生产者

```java
// 普通生产者
DefaultMQProducer producer = new DefaultMQProducer("ProducerGroup");
producer.setNamesrvAddr("localhost:9876");
producer.start();

// 事务生产者
TransactionMQProducer txProducer = new TransactionMQProducer("TxGroup");
txProducer.setTransactionListener(listener);
txProducer.start();
```

### 发送消息

```java
// 同步发送
SendResult result = producer.send(msg);

// 异步发送
producer.send(msg, new SendCallback() {
    public void onSuccess(SendResult result) {}
    public void onException(Throwable e) {}
});

// 单向发送
producer.sendOneway(msg);

// 顺序发送
producer.send(msg, selector, arg);

// 延迟发送
msg.setDelayTimeLevel(3);  // 10秒
producer.send(msg);

// 批量发送
producer.send(List.of(msg1, msg2, msg3));

// 事务发送
txProducer.sendMessageInTransaction(msg, arg);
```

### 常用配置

| 配置       | 方法                                 | 默认值 |
| ---------- | ------------------------------------ | ------ |
| NameServer | `setNamesrvAddr()`                   | -      |
| 发送超时   | `setSendMsgTimeout()`                | 3000ms |
| 同步重试   | `setRetryTimesWhenSendFailed()`      | 2      |
| 异步重试   | `setRetryTimesWhenSendAsyncFailed()` | 2      |
| 最大消息   | `setMaxMessageSize()`                | 4MB    |

## 消费者 API

### Push 消费者

```java
DefaultMQPushConsumer consumer = new DefaultMQPushConsumer("ConsumerGroup");
consumer.setNamesrvAddr("localhost:9876");
consumer.subscribe("TopicTest", "*");

// 并发消费
consumer.registerMessageListener((MessageListenerConcurrently) (msgs, ctx) -> {
    return ConsumeConcurrentlyStatus.CONSUME_SUCCESS;
});

// 顺序消费
consumer.registerMessageListener((MessageListenerOrderly) (msgs, ctx) -> {
    return ConsumeOrderlyStatus.SUCCESS;
});

consumer.start();
```

### Pull 消费者

```java
DefaultLitePullConsumer consumer = new DefaultLitePullConsumer("ConsumerGroup");
consumer.subscribe("TopicTest", "*");
consumer.start();

while (true) {
    List<MessageExt> msgs = consumer.poll(3000);
    // 处理消息
}
```

### 订阅方式

```java
// 订阅所有
consumer.subscribe("Topic", "*");

// 订阅单个 Tag
consumer.subscribe("Topic", "TagA");

// 订阅多个 Tag
consumer.subscribe("Topic", "TagA || TagB");

// SQL92 过滤
consumer.subscribe("Topic", MessageSelector.bySql("age > 18"));
```

### 常用配置

| 配置     | 方法                              | 默认值      |
| -------- | --------------------------------- | ----------- |
| 消费模式 | `setMessageModel()`               | CLUSTERING  |
| 起始位置 | `setConsumeFromWhere()`           | LAST_OFFSET |
| 最小线程 | `setConsumeThreadMin()`           | 20          |
| 最大线程 | `setConsumeThreadMax()`           | 64          |
| 批量数量 | `setConsumeMessageBatchMaxSize()` | 1           |
| 最大重试 | `setMaxReconsumeTimes()`          | 16          |

## 消息对象

### 创建消息

```java
// 基本消息
Message msg = new Message("Topic", "Body".getBytes());

// 带 Tag
Message msg = new Message("Topic", "Tag", "Body".getBytes());

// 带 Key
Message msg = new Message("Topic", "Tag", "Key", "Body".getBytes());

// 设置属性
msg.putUserProperty("key", "value");

// 设置延迟
msg.setDelayTimeLevel(3);
```

### 消息属性

| 方法                  | 说明           |
| --------------------- | -------------- |
| `getTopic()`          | 获取 Topic     |
| `getTags()`           | 获取 Tag       |
| `getKeys()`           | 获取 Key       |
| `getBody()`           | 获取消息体     |
| `getMsgId()`          | 获取消息 ID    |
| `getQueueId()`        | 获取队列 ID    |
| `getReconsumeTimes()` | 获取重试次数   |
| `getBornTimestamp()`  | 获取发送时间   |
| `getUserProperty()`   | 获取自定义属性 |

## 延迟级别

| 级别 | 时间 | 级别 | 时间 | 级别 | 时间 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 1    | 1s   | 7    | 3m   | 13   | 9m   |
| 2    | 5s   | 8    | 4m   | 14   | 10m  |
| 3    | 10s  | 9    | 5m   | 15   | 20m  |
| 4    | 30s  | 10   | 6m   | 16   | 30m  |
| 5    | 1m   | 11   | 7m   | 17   | 1h   |
| 6    | 2m   | 12   | 8m   | 18   | 2h   |

## Spring Boot 配置

### application.yml

```yaml
rocketmq:
  name-server: localhost:9876
  producer:
    group: producer-group
    send-message-timeout: 3000
    retry-times-when-send-failed: 2
    retry-times-when-send-async-failed: 2
  consumer:
    group: consumer-group
```

### 生产者

```java
@Autowired
private RocketMQTemplate rocketMQTemplate;

// 同步发送
rocketMQTemplate.syncSend("topic", message);
rocketMQTemplate.syncSend("topic:tag", message);

// 异步发送
rocketMQTemplate.asyncSend("topic", message, callback);

// 单向发送
rocketMQTemplate.sendOneWay("topic", message);

// 顺序发送
rocketMQTemplate.syncSendOrderly("topic", message, hashKey);

// 延迟发送
rocketMQTemplate.syncSend("topic", message, timeout, delayLevel);
```

### 消费者

```java
@Service
@RocketMQMessageListener(
    topic = "topic",
    consumerGroup = "consumer-group",
    selectorExpression = "TagA || TagB"
)
public class Consumer implements RocketMQListener<String> {
    @Override
    public void onMessage(String message) {
        // 处理消息
    }
}
```

## 管理命令

### 服务管理

```bash
# 启动 NameServer
nohup sh bin/mqnamesrv &

# 启动 Broker
nohup sh bin/mqbroker -n localhost:9876 &

# 关闭服务
sh bin/mqshutdown broker
sh bin/mqshutdown namesrv
```

### Topic 管理

```bash
# 创建 Topic
sh bin/mqadmin updateTopic -n localhost:9876 -b localhost:10911 -t TopicTest

# 查看 Topic 列表
sh bin/mqadmin topicList -n localhost:9876

# 查看 Topic 状态
sh bin/mqadmin topicStatus -n localhost:9876 -t TopicTest

# 删除 Topic
sh bin/mqadmin deleteTopic -n localhost:9876 -c DefaultCluster -t TopicTest
```

### 消费者组管理

```bash
# 查看消费进度
sh bin/mqadmin consumerProgress -n localhost:9876 -g ConsumerGroup

# 重置消费位点
sh bin/mqadmin resetOffsetByTime -n localhost:9876 \
    -g ConsumerGroup -t TopicTest -s now
```

### 消息查询

```bash
# 根据 MsgId 查询
sh bin/mqadmin queryMsgById -n localhost:9876 -i <msgId>

# 根据 Key 查询
sh bin/mqadmin queryMsgByKey -n localhost:9876 -t TopicTest -k <key>
```

## 端口说明

| 组件       | 端口  | 说明       |
| ---------- | ----- | ---------- |
| NameServer | 9876  | 路由服务   |
| Broker     | 10911 | 消息服务   |
| Broker     | 10909 | VIP 通道   |
| Dashboard  | 8080  | 管理控制台 |

## 常见状态码

### 发送状态

| 状态                  | 说明            |
| --------------------- | --------------- |
| `SEND_OK`             | 发送成功        |
| `FLUSH_DISK_TIMEOUT`  | 刷盘超时        |
| `FLUSH_SLAVE_TIMEOUT` | 同步 Slave 超时 |
| `SLAVE_NOT_AVAILABLE` | Slave 不可用    |

### 消费状态

```java
// 并发消费
ConsumeConcurrentlyStatus.CONSUME_SUCCESS    // 成功
ConsumeConcurrentlyStatus.RECONSUME_LATER    // 稍后重试

// 顺序消费
ConsumeOrderlyStatus.SUCCESS                 // 成功
ConsumeOrderlyStatus.SUSPEND_CURRENT_QUEUE_A_MOMENT  // 暂停
```

### 事务状态

```java
LocalTransactionState.COMMIT_MESSAGE    // 提交
LocalTransactionState.ROLLBACK_MESSAGE  // 回滚
LocalTransactionState.UNKNOW            // 未知
```

## 参考资料

- [RocketMQ 官方文档](https://rocketmq.apache.org/docs/)
- [RocketMQ GitHub](https://github.com/apache/rocketmq)
