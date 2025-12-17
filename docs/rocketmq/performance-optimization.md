---
sidebar_position: 13
title: "性能优化"
description: "RocketMQ 性能调优指南"
---

# RocketMQ 性能优化

本文档介绍 RocketMQ 各层面的性能优化策略，帮助你提升消息系统的吞吐量和降低延迟。

## 性能指标

### 关键指标

| 指标   | 说明            | 参考值      |
| ------ | --------------- | ----------- |
| TPS    | 每秒消息吞吐量  | 单机 10 万+ |
| 延迟   | 消息端到端延迟  | 毫秒级      |
| 堆积   | 未消费消息数量  | 越小越好    |
| 成功率 | 发送/消费成功率 | 99.99%+     |

### 性能测试工具

```bash
# RocketMQ 自带的性能测试工具
# 生产者测试
sh bin/tools.sh org.apache.rocketmq.example.benchmark.Producer \
    -t BenchmarkTopic -w 64 -s 1024

# 消费者测试
sh bin/tools.sh org.apache.rocketmq.example.benchmark.Consumer \
    -t BenchmarkTopic -g BenchmarkConsumer
```

## 发送端优化

### 1. 使用异步发送

```java
// ✅ 异步发送提升吞吐量
CountDownLatch latch = new CountDownLatch(messageCount);
for (int i = 0; i < messageCount; i++) {
    producer.send(msg, new SendCallback() {
        @Override
        public void onSuccess(SendResult result) {
            latch.countDown();
        }
        @Override
        public void onException(Throwable e) {
            latch.countDown();
            failedMessages.add(msg);
        }
    });
}
latch.await();
```

### 2. 批量发送

```java
// ✅ 批量发送减少网络往返
List<Message> messages = new ArrayList<>();
for (int i = 0; i < 100; i++) {
    messages.add(new Message("TopicTest", ("Message " + i).getBytes()));
}

// 注意：总大小不超过 4MB
SendResult result = producer.send(messages);
```

### 3. 消息压缩

```java
// 大于 4KB 自动压缩
producer.setCompressMsgBodyOverHowmuch(4096);

// 也可以手动压缩
byte[] compressed = compress(body);
Message msg = new Message("TopicTest", compressed);
msg.putUserProperty("compressed", "true");
```

### 4. 多线程发送

```java
// 使用线程池并发发送
ExecutorService executor = Executors.newFixedThreadPool(
    Runtime.getRuntime().availableProcessors() * 2);

for (Message msg : messages) {
    executor.submit(() -> {
        try {
            producer.send(msg);
        } catch (Exception e) {
            log.error("发送失败", e);
        }
    });
}
```

### 5. 预热连接

```java
// 启动时预热
producer.start();

// 发送预热消息
for (int i = 0; i < 10; i++) {
    Message warmupMsg = new Message("WarmupTopic", "warmup".getBytes());
    try {
        producer.send(warmupMsg, 1000);
    } catch (Exception e) {
        // 忽略预热消息失败
    }
}
```

### 发送端参数调优

```java
// 发送超时（根据网络情况）
producer.setSendMsgTimeout(3000);

// 重试次数（减少重试提升性能，降低可靠性）
producer.setRetryTimesWhenSendFailed(1);
producer.setRetryTimesWhenSendAsyncFailed(1);

// 是否等待存储完成
// SEND_OK: 刷盘+复制完成
// SLAVE_NOT_AVAILABLE: 只要主节点成功
producer.setRetryAnotherBrokerWhenNotStoreOK(false);
```

## 消费端优化

### 1. 增加消费线程

```java
// 根据业务类型调整
// CPU 密集型：CPU 核心数 + 1
// IO 密集型：CPU 核心数 * 2 或更多
consumer.setConsumeThreadMin(20);
consumer.setConsumeThreadMax(64);
```

### 2. 批量消费

```java
// 批量拉取
consumer.setPullBatchSize(32);

// 批量消费
consumer.setConsumeMessageBatchMaxSize(10);

consumer.registerMessageListener((MessageListenerConcurrently) (msgs, ctx) -> {
    // 批量处理
    List<Order> orders = msgs.stream()
        .map(msg -> JSON.parseObject(msg.getBody(), Order.class))
        .collect(Collectors.toList());

    orderService.batchProcess(orders);
    return ConsumeConcurrentlyStatus.CONSUME_SUCCESS;
});
```

### 3. 并行消费多个 Queue

```java
// 增加 Topic 的 Queue 数量
// Queue 数量 >= 消费者实例数 × 消费线程数
sh bin/mqadmin updateTopic -n localhost:9876 -t TopicTest -r 16 -w 16
```

### 4. 异步处理耗时操作

```java
// 消费端异步处理
ExecutorService asyncExecutor = Executors.newFixedThreadPool(10);

consumer.registerMessageListener((MessageListenerConcurrently) (msgs, ctx) -> {
    for (MessageExt msg : msgs) {
        // 异步处理耗时操作
        asyncExecutor.submit(() -> {
            processWithExternalService(msg);
        });
    }
    // 立即返回成功，异步处理
    return ConsumeConcurrentlyStatus.CONSUME_SUCCESS;
});
```

> ⚠️ **注意：** 异步处理需要自行保证可靠性，消息可能丢失

### 5. Pull 模式精细控制

```java
DefaultLitePullConsumer consumer = new DefaultLitePullConsumer("PullGroup");
consumer.subscribe("TopicTest", "*");
consumer.start();

while (true) {
    // 控制拉取频率和数量
    List<MessageExt> msgs = consumer.poll(1000);

    if (msgs.isEmpty()) {
        Thread.sleep(100);  // 无消息时休眠
        continue;
    }

    // 批量处理
    batchProcess(msgs);
    consumer.commitSync();
}
```

### 消费端参数调优

```java
// 拉取间隔
consumer.setPullInterval(0);  // 0 表示立即拉取

// 每次拉取数量
consumer.setPullBatchSize(32);

// 消费超时时间（分钟）
consumer.setConsumeTimeout(15);

// 消费起始位置
consumer.setConsumeFromWhere(ConsumeFromWhere.CONSUME_FROM_LAST_OFFSET);
```

## Broker 优化

### 存储参数

```properties
# broker.conf

# 异步刷盘（性能优先）
flushDiskType=ASYNC_FLUSH

# 刷盘间隔（毫秒）
flushIntervalCommitLog=500

# 异步复制（性能优先）
brokerRole=ASYNC_MASTER

# 消息存储路径（使用 SSD）
storePathRootDir=/ssd/rocketmq/store
storePathCommitLog=/ssd/rocketmq/store/commitlog

# CommitLog 文件大小（默认 1GB）
mapedFileSizeCommitLog=1073741824

# ConsumeQueue 文件大小
mapedFileSizeConsumeQueue=6000000

# 启用 transientStorePool（堆外内存）
transientStorePoolEnable=true
transientStorePoolSize=5

# 预分配 MappedFile
warmMapedFileEnable=true
```

### 线程池参数

```properties
# 发送消息线程数
sendMessageThreadPoolNums=16

# 拉取消息线程数
pullMessageThreadPoolNums=32

# 查询消息线程数
queryMessageThreadPoolNums=8

# 处理 Consumer 管理线程数
consumerManageThreadPoolNums=32

# 处理心跳线程数
heartbeatThreadPoolNums=8
```

### 内存参数

```properties
# 最大可存储消息比例
accessMessageInMemoryMaxRatio=40

# 清理过期文件时间（凌晨 4 点）
deleteWhen=04

# 消息保留时间（小时）
fileReservedTime=72

# 磁盘使用阈值
diskMaxUsedSpaceRatio=75
```

## JVM 优化

### 生产环境 JVM 参数

```bash
# runbroker.sh 修改

# 堆内存（根据物理内存调整）
JAVA_OPT="${JAVA_OPT} -server -Xms16g -Xmx16g"

# 新生代大小（堆内存的 1/3 ~ 1/2）
JAVA_OPT="${JAVA_OPT} -Xmn8g"

# 永久代/元空间
JAVA_OPT="${JAVA_OPT} -XX:MetaspaceSize=256m -XX:MaxMetaspaceSize=512m"

# GC 参数（G1 垃圾收集器）
JAVA_OPT="${JAVA_OPT} -XX:+UseG1GC"
JAVA_OPT="${JAVA_OPT} -XX:MaxGCPauseMillis=100"
JAVA_OPT="${JAVA_OPT} -XX:InitiatingHeapOccupancyPercent=45"

# GC 日志
JAVA_OPT="${JAVA_OPT} -Xlog:gc*:file=/var/log/rocketmq/gc.log:time,uptime:filecount=5,filesize=100M"
```

### 客户端 JVM 参数

```bash
# 生产者/消费者应用

# 适当的堆内存
-Xms4g -Xmx4g

# G1 垃圾收集器
-XX:+UseG1GC
-XX:MaxGCPauseMillis=50

# 减少 GC 日志
-Xlog:gc:file=/var/log/app/gc.log:time
```

## 操作系统优化

### Linux 内核参数

```bash
# /etc/sysctl.conf

# 文件描述符限制
fs.file-max = 1000000

# 网络参数
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 65535

# TCP 参数
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_tw_recycle = 0
net.ipv4.tcp_fin_timeout = 10
net.ipv4.tcp_keepalive_time = 600
net.ipv4.tcp_max_tw_buckets = 500000

# 内存参数
vm.swappiness = 10
vm.max_map_count = 655360
vm.dirty_ratio = 40
vm.dirty_background_ratio = 10

# 应用配置
sysctl -p
```

### 文件描述符限制

```bash
# /etc/security/limits.conf
* soft nofile 655360
* hard nofile 655360
* soft nproc 655360
* hard nproc 655360

# 生效
ulimit -n 655360
```

### 磁盘 I/O 优化

```bash
# 使用 SSD
# 挂载参数
mount -o noatime,nodiratime /dev/sdb1 /data/rocketmq

# I/O 调度器（SSD 使用 noop/none）
echo noop > /sys/block/sda/queue/scheduler

# 预读取大小
blockdev --setra 16384 /dev/sda
```

## 网络优化

### 网络配置

```bash
# 增加缓冲区大小
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.core.rmem_default = 1048576
net.core.wmem_default = 1048576

# TCP 缓冲区
net.ipv4.tcp_rmem = 4096 1048576 16777216
net.ipv4.tcp_wmem = 4096 1048576 16777216
```

### 客户端配置

```java
// 使用 VIP 通道
System.setProperty(MixAll.SEND_MESSAGE_WITH_VIP_CHANNEL_PROPERTY, "true");

// 连接超时
producer.setVipChannelEnabled(true);
```

## 性能对比

### 刷盘模式对比

| 模式     | TPS    | 延迟  | 可靠性 |
| -------- | ------ | ----- | ------ |
| 同步刷盘 | 5 万   | 10ms+ | 高     |
| 异步刷盘 | 10 万+ | 1ms   | 中     |

### 复制模式对比

| 模式     | TPS    | 延迟 | 可靠性 |
| -------- | ------ | ---- | ------ |
| 同步复制 | 8 万   | 5ms+ | 高     |
| 异步复制 | 10 万+ | 1ms  | 中     |

### 发送方式对比

| 方式     | TPS    | 适用场景 |
| -------- | ------ | -------- |
| 同步发送 | 3 万   | 重要消息 |
| 异步发送 | 10 万+ | 高吞吐   |
| 单向发送 | 15 万+ | 日志类   |

## 性能调优清单

### 发送端

- [ ] 使用异步发送
- [ ] 开启批量发送
- [ ] 合理设置重试次数
- [ ] 多线程并发发送
- [ ] 消息大小控制在 1MB 以内

### 消费端

- [ ] 增加消费线程数
- [ ] 批量拉取和消费
- [ ] Queue 数量 >= 消费者数
- [ ] 异步处理耗时操作
- [ ] 实现消费幂等

### Broker

- [ ] 使用 SSD 存储
- [ ] 异步刷盘（非金融场景）
- [ ] 调整线程池参数
- [ ] 开启 transientStorePool
- [ ] 合理设置内存参数

### 系统

- [ ] 调整文件描述符限制
- [ ] 优化网络参数
- [ ] 使用 G1/ZGC 垃圾收集器
- [ ] 关闭 swap

## 下一步

- 📊 [监控运维](/docs/rocketmq/monitoring) - 建设监控体系
- 🏗️ [集群管理](/docs/rocketmq/cluster-management) - 集群部署与运维
- ✅ [最佳实践](/docs/rocketmq/best-practices) - 生产环境实践

## 参考资料

- [RocketMQ 性能调优](https://rocketmq.apache.org/docs/bestPractice/)
- [RocketMQ 官方 Benchmark](https://rocketmq.apache.org/docs/benchmark/)
