---
sidebar_position: 16
title: "排障手册"
description: "RocketMQ 常见故障定位思路、排障命令与应急处理"
---

# RocketMQ 排障手册

本文按“现象 -> 排查路径 -> 常用命令/日志 -> 处理建议”的方式整理 RocketMQ 生产常见问题。

## 排障总原则

- **先确认范围**：是单个应用问题（Producer/Consumer）还是集群问题（NameServer/Broker）
- **先确认链路**：网络可达 -> 路由可见 -> Topic/Group 正确 -> 权限通过 -> 发送/消费线程池健康
- **先看数据**：堆积量、TPS、失败数、磁盘/内存水位
- **避免“盲目重启”**：重启会掩盖根因，优先基于日志和指标定位

## 快速体检（3 分钟）

### 1. NameServer / Broker 是否存活

```bash
# NameServer/Broker 进程
jps | grep -E "NamesrvStartup|BrokerStartup"

# 端口
# NameServer: 9876
# Broker: 10911 (服务端口), 10909 (VIP)
netstat -an | grep -E "9876|10911|10909"
```

### 2. 集群视角确认

```bash
sh bin/mqadmin clusterList -n <namesrv>
sh bin/mqadmin brokerStatus -n <namesrv> -b <brokerAddr>
```

### 3. Topic 路由是否正常

```bash
sh bin/mqadmin topicList -n <namesrv>
sh bin/mqadmin topicRoute -n <namesrv> -t <topic>
sh bin/mqadmin topicStatus -n <namesrv> -t <topic>
```

### 4. 消费进度/堆积

```bash
sh bin/mqadmin consumerProgress -n <namesrv> -g <consumerGroup>
sh bin/mqadmin consumerConnection -n <namesrv> -g <consumerGroup>
```

## 常见现象与定位

## 现象 1：Producer 发送失败 / 超时

### 常见报错

- `Send [xx] times, still failed, cost [xx]ms`
- `No route info of this topic`
- `connect to <broker> failed`
- `the broker does not exist`

### 排查路径

- **网络**：Producer -> NameServer、Producer -> Broker 是否可达
- **路由**：Topic 是否存在、路由是否正确、NameServer 是否有该 Broker 信息
- **Broker 压力**：线程池满、磁盘满、PageCache 压力、GC
- **权限**：开启 ACL 后 AK/SK 是否正确

### 常用命令

```bash
# 路由
sh bin/mqadmin topicRoute -n <namesrv> -t <topic>

# Broker 状态
sh bin/mqadmin brokerStatus -n <namesrv> -b <brokerAddr>

# Topic 是否存在
sh bin/mqadmin topicList -n <namesrv> | grep <topic>
```

### 处理建议

- **临时缓解**
  - 适当增加 `sendMsgTimeout`
  - 发送端开启异步、批量发送（注意 4MB 限制）
- **根因处理**
  - Broker 写入线程池/磁盘/内存瓶颈：参考 [性能优化](/docs/rocketmq/performance-optimization)
  - 路由缺失：确认 Broker 是否成功注册到所有 NameServer

## 现象 2：`No route info of this topic`

### 常见原因

- Topic 未创建
- Broker 未在 NameServer 注册（NameServer 地址配置错误 / 部分 NameServer 不可用）
- Topic 在某个集群/某个 Broker 上创建了，但客户端查的 NameServer 不一致

### 排查命令

```bash
sh bin/mqadmin topicList -n <namesrv>
sh bin/mqadmin topicRoute -n <namesrv> -t <topic>
sh bin/mqadmin clusterList -n <namesrv>
```

### 处理建议

- 用 `mqadmin updateTopic` 在目标集群创建 Topic（注意读写队列数）
- 客户端配置多个 NameServer（至少 2 个）并确保地址一致

## 现象 3：Consumer 收不到消息

### 排查路径

- **订阅是否正确**：Topic/Tag/SQL 过滤表达式
- **消费模式**：集群消费下是否被其他实例消费了
- **Offset**：是否从最新开始消费导致“历史消息看不到”
- **消费位点卡住**：某些队列积压严重
- **ACL**：无订阅权限/Group 权限

### 常用命令

```bash
# 消费者是否在线
sh bin/mqadmin consumerConnection -n <namesrv> -g <consumerGroup>

# 消费进度
sh bin/mqadmin consumerProgress -n <namesrv> -g <consumerGroup>

# 查询消息（按 key 或 msgId）
sh bin/mqadmin queryMsgByKey -n <namesrv> -t <topic> -k <key>
sh bin/mqadmin queryMsgById  -n <namesrv> -i <msgId>
```

### 处理建议

- 新消费组默认通常从最新 offset 开始：
  - 需要消费历史消息时，设置 `CONSUME_FROM_FIRST_OFFSET` 或重置 offset
- 检查过滤条件（Tag/SQL92）：先用 `*` 放开订阅验证链路

## 现象 4：消息堆积（Backlog/Diff 持续增长）

### 排查路径

- **生产 TPS > 消费 TPS**
- 消费逻辑慢（外部依赖、DB、RPC）
- 消费线程池/批量参数不合理
- Queue 数量不足导致并行度不够

### 常用命令

```bash
# 堆积量
sh bin/mqadmin consumerProgress -n <namesrv> -g <consumerGroup>

# Topic 状态
sh bin/mqadmin topicStatus -n <namesrv> -t <topic>
```

### 应急处理（从低风险到高风险）

- **增加消费者实例**（不超过 Queue 数量的有效并行度）
- **提升消费线程数、批量拉取/批量消费参数**
- **临时增加 Queue 数量**（扩容前评估，避免频繁变更）
- **跳过堆积消息/重置 Offset**（高风险，需业务确认）

## 现象 5：Broker 磁盘占用过高 / 写入失败

### 典型症状

- 日志出现磁盘水位告警
- `putMessage` 失败
- broker 不再接收写入

### 排查路径

- `commitlog` 目录是否暴涨
- `fileReservedTime` 是否过大
- `diskMaxUsedSpaceRatio` 是否过低导致过早保护

### 处理建议

- **先扩盘/清理**（确保系统可继续运行）
- 合理设置：
  - `fileReservedTime`
  - `deleteWhen`
  - `diskMaxUsedSpaceRatio`
- 结合监控提前告警：参考 [监控与运维](/docs/rocketmq/monitoring)

## 现象 6：顺序消费卡住

### 典型原因

- 顺序消费某条消息持续失败，导致队列被“锁住”，后续消息无法推进。

### 处理建议

- 针对顺序消费：
  - 对“可跳过”的错误设置最大重试次数后降级处理
  - 将异常消息落库/告警后返回成功，避免阻塞队列

## 现象 7：开启 ACL 后大量 `permission denied`

- 先核对：Topic/Group 授权是否齐全
- 再核对：来源 IP 是否命中白名单
- 最后核对：客户端是否所有实例都配置了 AK/SK（尤其是灰度环境）

详见：

- 🔒 [安全与 ACL](/docs/rocketmq/security)

## 日志定位指南

### 默认日志位置

```bash
~/logs/rocketmqlogs/
```

建议重点关注：

- `namesrv.log`
- `broker.log`
- `store.log`
- `remoting.log`
- `transaction.log`

### 常用 grep

```bash
# Broker 错误
grep -i "error\|exception" ~/logs/rocketmqlogs/broker.log | tail -n 200

# 网络/连接
grep -i "remoting\|connect" ~/logs/rocketmqlogs/remoting.log | tail -n 200
```

## 常用应急操作（谨慎）

### 重置消费位点

```bash
# 重置到 now（跳过堆积）
sh bin/mqadmin resetOffsetByTime -n <namesrv> -g <consumerGroup> -t <topic> -s now

# 重置到指定时间
sh bin/mqadmin resetOffsetByTime -n <namesrv> -g <consumerGroup> -t <topic> -s "2024-01-01#00:00:00"
```

## 下一步

- 📊 [监控与运维](/docs/rocketmq/monitoring)
- ⚡ [性能优化](/docs/rocketmq/performance-optimization)
- 🔒 [安全与 ACL](/docs/rocketmq/security)
