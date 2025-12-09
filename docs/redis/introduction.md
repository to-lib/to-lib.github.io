---
sidebar_position: 2
title: Redis 简介
---

# Redis 简介

## 什么是 Redis

Redis（Remote Dictionary Server）是一个开源的、基于内存的高性能键值数据库。它支持多种数据结构，如字符串、哈希、列表、集合、有序集合等，被广泛应用于缓存、消息队列、实时分析等场景。

### 核心特点

- **内存存储** - 数据存储在内存中，读写速度极快
- **持久化** - 支持 RDB 和 AOF 两种持久化方式
- **多种数据结构** - 支持 String、List、Set、Hash、Sorted Set 等
- **原子操作** - 所有操作都是原子性的
- **主从复制** - 支持数据复制和读写分离
- **高可用** - 通过哨兵和集群实现高可用
- **Lua 脚本** - 支持 Lua 脚本扩展功能

## 应用场景

### 1. 缓存

Redis 最常见的应用场景是作为缓存层，减轻数据库压力，提高系统性能。

```java
// Spring Boot 中使用 Redis 缓存
@Cacheable(value = "users", key = "#id")
public User getUserById(Long id) {
    return userRepository.findById(id).orElse(null);
}

@CacheEvict(value = "users", key = "#user.id")
public void updateUser(User user) {
    userRepository.save(user);
}
```

**适用场景**：

- 热点数据缓存
- 数据库查询结果缓存
- 会话缓存（Session）
- 页面缓存

### 2. 分布式锁

使用 Redis 实现分布式锁，解决分布式系统中的并发问题。

```bash
# 获取锁（带过期时间，防止死锁）
SET lock:resource_id unique_value NX EX 30

# 释放锁（使用 Lua 脚本保证原子性）
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
```

**适用场景**：

- 防止重复提交
- 定时任务互斥
- 库存扣减
- 秒杀活动

### 3. 计数器

利用 Redis 的原子自增操作实现各种计数功能。

```bash
# 文章浏览量
INCR article:1001:views

# 点赞数
INCR post:2001:likes

# 限流计数
INCR rate_limit:user:1001
EXPIRE rate_limit:user:1001 60
```

**适用场景**：

- 文章浏览量、点赞数
- API 限流
- 实时统计

### 4. 排行榜

使用有序集合（Sorted Set）实现排行榜功能。

```bash
# 添加分数
ZADD leaderboard 1000 "player1"
ZADD leaderboard 1500 "player2"

# 获取排行榜（前 10 名）
ZREVRANGE leaderboard 0 9 WITHSCORES

# 获取用户排名
ZREVRANK leaderboard "player1"
```

**适用场景**：

- 游戏排行榜
- 热门文章排行
- 销售排行

### 5. 消息队列

使用 List 或 Stream 实现简单的消息队列。

```bash
# 生产者（发布消息）
LPUSH message_queue "task1"

# 消费者（消费消息）
BRPOP message_queue 0
```

**适用场景**：

- 异步任务处理
- 日志收集
- 事件通知

### 6. 实时分析

使用 HyperLogLog 进行基数统计，Bitmap 进行用户行为分析。

```bash
# UV 统计（使用 HyperLogLog）
PFADD page:home:uv user1 user2 user3
PFCOUNT page:home:uv

# 用户签到（使用 Bitmap）
SETBIT user:1001:signin 100 1  # 第 100 天签到
BITCOUNT user:1001:signin       # 总签到天数
```

**适用场景**：

- UV/PV 统计
- 用户签到
- 在线用户统计

## 安装和配置

### 使用 Docker 安装

```bash
# 拉取 Redis 镜像
docker pull redis:7.0

# 运行 Redis 容器
docker run -d \
  --name redis \
  -p 6379:6379 \
  redis:7.0 \
  redis-server --appendonly yes
```

### macOS 安装

```bash
# 使用 Homebrew 安装
brew install redis

# 启动 Redis
brew services start redis

# 停止 Redis
brew services stop redis
```

### Linux 安装

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install redis-server

# CentOS/RHEL
sudo yum install redis

# 启动 Redis
sudo systemctl start redis
sudo systemctl enable redis
```

## 基本配置

Redis 的配置文件通常位于 `/etc/redis/redis.conf` 或 `/usr/local/etc/redis.conf`。

### 常用配置项

```conf
# 绑定地址（生产环境建议绑定内网 IP）
bind 127.0.0.1

# 端口
port 6379

# 后台运行
daemonize yes

# 日志级别
loglevel notice

# 日志文件
logfile /var/log/redis/redis.log

# 数据库数量
databases 16

# 最大内存
maxmemory 2gb

# 内存淘汰策略
maxmemory-policy allkeys-lru

# 持久化配置
save 900 1      # 900 秒内至少 1 个键被修改
save 300 10     # 300 秒内至少 10 个键被修改
save 60 10000   # 60 秒内至少 10000 个键被修改

# AOF 持久化
appendonly yes
appendfilename "appendonly.aof"
```

## 基本操作

### 连接 Redis

```bash
# 使用 redis-cli 连接
redis-cli

# 连接远程 Redis
redis-cli -h 192.168.1.100 -p 6379 -a password

# 测试连接
PING  # 返回 PONG 表示连接成功
```

### 键操作

```bash
# 设置键
SET name "张三"

# 获取键
GET name

# 检查键是否存在
EXISTS name

# 删除键
DEL name

# 设置过期时间（秒）
EXPIRE name 3600

# 查看剩余过期时间
TTL name

# 查看所有键（慎用，生产环境数据量大时会阻塞）
KEYS *

# 扫描键（推荐使用，不会阻塞）
SCAN 0 MATCH user:* COUNT 100
```

### 数据库操作

```bash
# 选择数据库（默认 0-15）
SELECT 1

# 查看当前数据库键数量
DBSIZE

# 清空当前数据库
FLUSHDB

# 清空所有数据库
FLUSHALL
```

## 性能特点

### 高性能原因

1. **内存存储** - 数据存储在内存中，避免磁盘 I/O
2. **单线程模型** - 避免线程切换和竞态问题（Redis 6.0+ 网络 I/O 多线程）
3. **高效的数据结构** - 针对不同场景优化的数据结构
4. **非阻塞 I/O** - 使用 epoll/kqueue 等高效 I/O 多路复用
5. **简单的协议** - RESP 协议简单高效

### 性能指标

- **吞吐量** - 单机可达 10 万+ QPS
- **延迟** - 毫秒级响应时间
- **并发** - 支持数万并发连接

## 最佳实践

### 1. 键命名规范

使用有意义的键名，建议使用冒号分隔：

```bash
# 好的命名
user:1001:profile
order:2023:12:09:1001
cache:product:1001

# 避免
u1
o1
p1
```

### 2. 合理设置过期时间

避免内存溢出，为键设置合理的过期时间：

```bash
# 缓存用户信息，30 分钟过期
SETEX user:1001:cache 1800 "{\"name\":\"张三\",\"age\":25}"
```

### 3. 避免大键

大键会影响性能，尽量避免：

- String 类型不超过 10 KB
- List、Set、Hash、Sorted Set 元素数量不超过 1 万

### 4. 使用连接池

避免频繁创建连接，使用连接池提高性能：

```java
// Jedis 连接池配置
JedisPoolConfig config = new JedisPoolConfig();
config.setMaxTotal(100);
config.setMaxIdle(20);
config.setMinIdle(10);

JedisPool pool = new JedisPool(config, "localhost", 6379);
```

## 小结

Redis 是一个功能强大、性能卓越的内存数据库，适用于缓存、分布式锁、计数器、排行榜、消息队列等多种场景。理解 Redis 的核心特点和应用场景，是高效使用 Redis 的基础。

在接下来的章节中，我们将深入学习 Redis 的各种数据类型、持久化机制、高可用架构等核心知识。
