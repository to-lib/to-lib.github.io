---
sidebar_position: 1
title: Redis 数据库学习指南
---

# Redis 数据库

欢迎来到 Redis 数据库完整学习指南！本指南涵盖了 Redis 从基础到高级的核心知识和实践技巧。

## 📚 学习内容

### 基础知识

- **Redis 简介** - Redis 特点、应用场景、安装配置
- **数据类型** - String、List、Set、Hash、Sorted Set 等类型详解
- **基本操作** - 常用命令、键值操作、数据管理

### 核心特性

- **持久化** - RDB、AOF、混合持久化策略
- **主从复制** - 复制原理、配置、故障处理
- **哨兵模式** - 高可用架构、故障转移
- **Redis 集群** - 数据分片、槽位、扩缩容

### 高级主题

- **事务** - MULTI/EXEC、WATCH、Lua 脚本
- **缓存策略** - 缓存穿透、击穿、雪崩、分布式锁
- **性能优化** - 内存优化、慢查询分析、性能监控

## 🚀 快速开始

如果你是 Redis 初学者，建议按以下顺序学习：

1. [Redis 简介](./introduction) - 了解 Redis 基本概念和特点
2. [数据类型](./data-types) - 掌握各种数据类型的使用
3. [持久化](./persistence) - 理解数据持久化机制
4. [主从复制](./replication) - 学习主从复制原理
5. [缓存策略](./cache-strategies) - 掌握常见缓存问题解决方案

## 📖 学习路径

### 初级开发者

- Redis 基本概念和特点
- 五种基本数据类型（String、List、Set、Hash、Sorted Set）
- 基本命令操作
- 键的过期和删除策略
- 简单持久化配置

### 中级开发者

- 高级数据类型（Bitmap、HyperLogLog、Stream）
- RDB 和 AOF 持久化深入
- 主从复制配置和原理
- 哨兵模式和故障转移
- 事务和 Lua 脚本
- 常见缓存问题（穿透、击穿、雪崩）

### 高级开发者

- Redis 集群架构和原理
- 数据分片和槽位管理
- 分布式锁实现
- 性能调优和内存优化
- 慢查询分析
- 高可用架构设计
- Redis 在实际项目中的最佳实践

## 💡 核心概念速览

### 数据类型

Redis 支持多种数据类型，满足不同场景需求：

- **String** - 字符串，最基本的类型，可存储字符串、数字、二进制数据
- **List** - 列表，有序的字符串列表，支持双端操作
- **Set** - 集合，无序不重复的字符串集合
- **Hash** - 哈希表，字段-值对的集合
- **Sorted Set** - 有序集合，每个成员关联一个分数用于排序

### 持久化方式

- **RDB** - 快照持久化，定期保存数据集的时点快照
- **AOF** - 追加文件，记录每个写操作，重启时重放恢复数据
- **混合持久化** - RDB + AOF，兼顾性能和数据安全性

### 高可用方案

- **主从复制** - 数据冗余，读写分离，提高可用性
- **哨兵模式** - 自动故障转移，监控主从状态
- **Redis 集群** - 数据分片，横向扩展，高可用

## 🔧 常用命令速览

### 字符串操作

```bash
# 设置键值
SET key value

# 获取值
GET key

# 设置带过期时间的值
SETEX key 3600 value

# 自增
INCR counter

# 自减
DECR counter
```

### 列表操作

```bash
# 左侧插入
LPUSH mylist "value1"

# 右侧插入
RPUSH mylist "value2"

# 获取列表范围
LRANGE mylist 0 -1

# 弹出元素
LPOP mylist
RPOP mylist
```

### 集合操作

```bash
# 添加成员
SADD myset "member1" "member2"

# 获取所有成员
SMEMBERS myset

# 集合运算
SUNION set1 set2     # 并集
SINTER set1 set2     # 交集
SDIFF set1 set2      # 差集
```

### 哈希操作

```bash
# 设置哈希字段
HSET user:1 name "张三" age 25

# 获取哈希字段
HGET user:1 name

# 获取所有字段和值
HGETALL user:1

# 删除字段
HDEL user:1 age
```

### 有序集合操作

```bash
# 添加成员
ZADD leaderboard 100 "player1" 200 "player2"

# 按分数范围查询
ZRANGE leaderboard 0 -1 WITHSCORES

# 按分数排名
ZRANK leaderboard "player1"
```

## 📚 完整学习资源

| 主题                                   | 描述                                   |
| -------------------------------------- | -------------------------------------- |
| [Redis 简介](./introduction)           | Redis 特点、应用场景、安装和基本操作   |
| [数据类型](./data-types)               | String、List、Set、Hash、Sorted Set 等 |
| [持久化](./persistence)                | RDB、AOF、混合持久化机制               |
| [主从复制](./replication)              | 主从复制原理、配置、故障处理           |
| [哨兵模式](./sentinel)                 | 哨兵架构、故障转移、高可用配置         |
| [Redis 集群](./cluster)                | 集群架构、数据分片、槽位管理           |
| [事务](./transactions)                 | MULTI/EXEC、WATCH、Lua 脚本            |
| [缓存策略](./cache-strategies)         | 缓存穿透、击穿、雪崩、分布式锁         |
| [性能优化](./performance-optimization) | 内存优化、慢查询分析、性能监控         |

## 🔗 相关资源

- [Java 编程](../java)
- [Spring Framework](../spring)
- [Spring Boot](../springboot)
- [MySQL 数据库](../mysql)

## 📖 推荐学习资源

- [Redis 官方文档](https://redis.io/documentation)
- [Redis 命令参考](https://redis.io/commands)
- [Redis 设计与实现](http://redisbook.com/)

---

**最后更新**: 2025 年 12 月  
**版本**: Redis 7.0+

开始你的 Redis 学习之旅吧！
