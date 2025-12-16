---
sidebar_position: 2
title: 快速入门
---

# Redis 快速入门

5 分钟快速上手 Redis，从安装到编写第一个程序。

## 安装 Redis

### Docker（推荐）

最快的方式是使用 Docker：

```bash
# 拉取并运行 Redis
docker run -d --name redis -p 6379:6379 redis:7.0

# 验证运行
docker exec -it redis redis-cli ping
# 返回 PONG 表示成功
```

### macOS

```bash
# 使用 Homebrew
brew install redis

# 启动服务
brew services start redis

# 连接测试
redis-cli ping
```

### Linux

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install redis-server

# 启动服务
sudo systemctl start redis-server
sudo systemctl enable redis-server

# 测试连接
redis-cli ping
```

### Windows

推荐使用 WSL2 或 Docker Desktop。

## 连接 Redis

使用 `redis-cli` 命令行工具连接：

```bash
# 本地连接
redis-cli

# 连接远程服务器
redis-cli -h 192.168.1.100 -p 6379

# 带密码连接
redis-cli -h 192.168.1.100 -p 6379 -a yourpassword

# 测试连接
127.0.0.1:6379> PING
PONG
```

## 核心数据类型速览

Redis 支持 5 种基本数据类型，每种都有特定的应用场景。

### 1. String（字符串）

最基本的类型，可存储文本、数字、二进制数据。

```bash
# 设置值
SET name "张三"

# 获取值
GET name
# "张三"

# 设置带过期时间（60秒）
SETEX token 60 "abc123"

# 计数器
SET counter 0
INCR counter       # 1
INCRBY counter 10  # 11
```

**应用场景**：缓存、计数器、Session

### 2. List（列表）

有序的字符串列表，支持两端操作。

```bash
# 添加元素
LPUSH tasks "task1"
LPUSH tasks "task2"
RPUSH tasks "task3"

# 获取列表
LRANGE tasks 0 -1
# ["task2", "task1", "task3"]

# 弹出元素
LPOP tasks   # "task2"
RPOP tasks   # "task3"
```

**应用场景**：消息队列、最新列表、任务队列

### 3. Hash（哈希）

字段-值对的集合，适合存储对象。

```bash
# 设置字段
HSET user:1001 name "张三" age "25" city "北京"

# 获取字段
HGET user:1001 name
# "张三"

# 获取所有字段
HGETALL user:1001
# {"name": "张三", "age": "25", "city": "北京"}

# 自增字段
HINCRBY user:1001 age 1
```

**应用场景**：用户信息、商品详情、配置存储

### 4. Set（集合）

无序不重复的字符串集合。

```bash
# 添加成员
SADD tags "java" "redis" "docker"

# 获取所有成员
SMEMBERS tags
# ["java", "redis", "docker"]

# 判断是否存在
SISMEMBER tags "java"  # 1 (true)

# 集合运算
SADD tags2 "redis" "mysql"
SINTER tags tags2      # ["redis"]
```

**应用场景**：标签、共同好友、去重

### 5. Sorted Set（有序集合）

每个元素关联一个分数，按分数排序。

```bash
# 添加成员和分数
ZADD leaderboard 100 "player1"
ZADD leaderboard 200 "player2"
ZADD leaderboard 150 "player3"

# 按分数排名（从高到低）
ZREVRANGE leaderboard 0 -1 WITHSCORES
# ["player2", 200, "player3", 150, "player1", 100]

# 获取排名
ZREVRANK leaderboard "player1"  # 2 (第3名)
```

**应用场景**：排行榜、优先级队列、延迟队列

## 常用操作

### 键管理

```bash
# 查看所有键（慎用）
KEYS *

# 推荐使用 SCAN
SCAN 0 MATCH user:* COUNT 10

# 删除键
DEL key1 key2

# 检查键是否存在
EXISTS key

# 设置过期时间
EXPIRE key 3600

# 查看剩余时间
TTL key
```

### 数据库操作

```bash
# Redis 默认有 16 个数据库（0-15）
SELECT 1    # 切换到数据库 1

# 查看键数量
DBSIZE

# 清空当前数据库（危险操作）
FLUSHDB

# 清空所有数据库（危险操作）
FLUSHALL
```

## 第一个程序

### Java (Jedis)

添加依赖：

```xml
<dependency>
    <groupId>redis.clients</groupId>
    <artifactId>jedis</artifactId>
    <version>5.0.0</version>
</dependency>
```

代码示例：

```java
import redis.clients.jedis.Jedis;

public class RedisQuickStart {
    public static void main(String[] args) {
        // 连接 Redis
        try (Jedis jedis = new Jedis("localhost", 6379)) {
            // 测试连接
            System.out.println("连接成功: " + jedis.ping());

            // String 操作
            jedis.set("name", "张三");
            System.out.println("name = " + jedis.get("name"));

            // 计数器
            jedis.set("counter", "0");
            jedis.incr("counter");
            System.out.println("counter = " + jedis.get("counter"));

            // Hash 操作
            jedis.hset("user:1001", "name", "张三");
            jedis.hset("user:1001", "age", "25");
            System.out.println("user = " + jedis.hgetAll("user:1001"));

            // 设置过期时间
            jedis.setex("session", 60, "session_data");
            System.out.println("TTL = " + jedis.ttl("session") + "秒");
        }
    }
}
```

### Python

安装依赖：

```bash
pip install redis
```

代码示例：

```python
import redis

# 连接 Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# 测试连接
print("连接成功:", r.ping())

# String 操作
r.set('name', '张三')
print("name =", r.get('name'))

# 计数器
r.set('counter', 0)
r.incr('counter')
print("counter =", r.get('counter'))

# Hash 操作
r.hset('user:1001', mapping={'name': '张三', 'age': '25'})
print("user =", r.hgetall('user:1001'))

# 设置过期时间
r.setex('session', 60, 'session_data')
print("TTL =", r.ttl('session'), "秒")
```

### Spring Boot

添加依赖：

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-redis</artifactId>
</dependency>
```

配置：

```yaml
spring:
  redis:
    host: localhost
    port: 6379
```

使用 RedisTemplate：

```java
@RestController
public class RedisController {

    @Autowired
    private RedisTemplate<String, String> redisTemplate;

    @GetMapping("/redis/test")
    public String test() {
        // 设置值
        redisTemplate.opsForValue().set("name", "张三");

        // 获取值
        String name = redisTemplate.opsForValue().get("name");

        // 设置过期时间
        redisTemplate.opsForValue().set("token", "abc123", 60, TimeUnit.SECONDS);

        return "name = " + name;
    }
}
```

## 常见问题

### 连接失败？

```bash
# 检查 Redis 是否运行
redis-cli ping

# 检查端口是否开放
telnet localhost 6379

# Docker 检查容器状态
docker ps
docker logs redis
```

### 密码错误？

```bash
# 连接时指定密码
redis-cli -a yourpassword

# 或连接后认证
redis-cli
127.0.0.1:6379> AUTH yourpassword
```

### 远程无法连接？

检查 `redis.conf` 配置：

```conf
# 绑定地址（允许远程连接）
bind 0.0.0.0

# 关闭保护模式
protected-mode no

# 设置密码
requirepass yourpassword
```

## 下一步

恭喜你完成了 Redis 快速入门！接下来建议学习：

1. [数据类型详解](/docs/redis/data-types) - 深入了解各种数据类型
2. [持久化](/docs/redis/persistence) - 数据持久化机制
3. [缓存策略](/docs/redis/cache-strategies) - 缓存穿透、击穿、雪崩
4. [实战案例](/docs/redis/practical-examples) - 分布式锁、限流等

## 小结

- ✅ Redis 安装简单，推荐使用 Docker
- ✅ 5 种基本数据类型满足大部分场景
- ✅ 客户端库丰富，Java/Python/Spring Boot 都有良好支持
- ✅ 使用连接池提高性能
- ✅ 设置合理的过期时间，避免内存溢出

开始你的 Redis 之旅吧！
