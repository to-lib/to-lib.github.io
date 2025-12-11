---
sidebar_position: 11
title: "常见问题"
description: "RabbitMQ 常见问题解答"
---

# RabbitMQ 常见问题

## 连接问题

### Q: 无法连接到 RabbitMQ?

**A:** 检查以下几点:

```bash
# 1. 检查服务是否运行
rabbitmqctl status

# 2. 检查端口是否开放
nc -zv localhost 5672

# 3. 检查防火墙
sudo iptables -L | grep 5672

# 4. 检查用户权限
rabbitmqctl list_permissions
```

### Q: 连接频繁断开?

**A:** 可能原因和解决方案:

1. **心跳超时**: 调整心跳间隔

   ```java
   factory.setRequestedHeartbeat(60);
   ```

2. **网络问题**: 检查网络稳定性

3. **启用自动恢复**:
   ```java
   factory.setAutomaticRecoveryEnabled(true);
   factory.setNetworkRecoveryInterval(5000);
   ```

## 消息问题

### Q: 消息丢失怎么办?

**A:** 确保以下配置:

1. **队列持久化**: `durable = true`
2. **消息持久化**: `deliveryMode = 2`
3. **禁用自动确认**: `autoAck = false`
4. **启用发布确认**: `confirmSelect()`

### Q: 消息重复消费?

**A:** 实现幂等性:

```java
// 使用消息 ID 去重
String messageId = properties.getMessageId();
if (!redis.setIfAbsent("processed:" + messageId, "1")) {
    return; // 已处理过
}
processMessage(body);
```

### Q: 消息顺序无法保证?

**A:** RabbitMQ 只保证队列内有序:

1. 使用单一队列和单一消费者
2. 或使用消息批次编号排序

## 队列问题

### Q: 队列消息积压?

**A:** 解决方案:

1. **增加消费者数量**
2. **提高预取值**
3. **优化消费逻辑**
4. **清理历史消息**

```bash
# 清空队列
rabbitmqctl purge_queue queue_name
```

### Q: 队列无法删除?

**A:** 确保没有消费者:

```bash
# 查看队列消费者
rabbitmqctl list_queues name consumers

# 强制删除
rabbitmqctl delete_queue queue_name
```

## 性能问题

### Q: 吞吐量低?

**A:** 优化建议:

1. 使用批量发送和确认
2. 调整 QoS 预取值
3. 使用异步确认
4. 减少消息大小

### Q: 内存使用过高?

**A:** 解决方案:

1. 使用 Lazy Queue
2. 设置消息 TTL
3. 限制队列长度
4. 调整内存阈值

```ini
# rabbitmq.conf
vm_memory_high_watermark.relative = 0.4
```

## 集群问题

### Q: 节点无法加入集群?

**A:** 检查:

1. **Erlang Cookie 一致**
2. **主机名解析正确**
3. **端口开放**: 4369, 25672

### Q: 网络分区?

**A:** 配置分区处理:

```ini
cluster_partition_handling = autoheal
```

## 常用命令速查

```bash
# 服务管理
rabbitmqctl status
rabbitmqctl stop_app
rabbitmqctl start_app

# 队列管理
rabbitmqctl list_queues
rabbitmqctl purge_queue <queue>
rabbitmqctl delete_queue <queue>

# 用户管理
rabbitmqctl add_user <user> <pass>
rabbitmqctl set_permissions -p / <user> ".*" ".*" ".*"

# 插件管理
rabbitmq-plugins enable rabbitmq_management
rabbitmq-plugins list
```

## 参考资料

- [RabbitMQ 故障排查](https://www.rabbitmq.com/troubleshooting.html)
