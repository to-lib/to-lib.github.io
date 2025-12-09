---
sidebar_position: 3
title: Redis 数据类型
---

# Redis 数据类型

Redis 支持多种数据类型，每种类型都有其特定的使用场景和操作命令。理解这些数据类型是高效使用 Redis 的关键。

## String（字符串）

String 是 Redis 最基本的数据类型，一个键最多可以存储 512 MB 的数据。

### 基本操作

```bash
# 设置值
SET name "张三"

# 获取值
GET name

# 设置多个键值对
MSET key1 "value1" key2 "value2"

# 获取多个值
MGET key1 key2

# 设置值并返回旧值
GETSET name "李四"

# 追加字符串
APPEND name " - 程序员"

# 获取字符串长度
STRLEN name
```

### 数值操作

```bash
# 自增
INCR counter          # 增加 1
INCRBY counter 5      # 增加 5

# 自减
DECR counter          # 减少 1
DECRBY counter 3      # 减少 3

# 浮点数增加
INCRBYFLOAT price 0.5
```

### 带过期时间的操作

```bash
# 设置键值和过期时间（秒）
SETEX session:1001 3600 "user_data"

# 设置键值和过期时间（毫秒）
PSETEX cache:1001 60000 "cached_data"

# 仅当键不存在时设置
SETNX lock:resource "locked"
```

### 应用场景

- **缓存** - 缓存用户信息、商品信息等
- **计数器** - 浏览量、点赞数、库存等
- **分布式锁** - 使用 SETNX 实现
- **Session 存储** - 存储用户会话信息

## List（列表）

List 是一个有序的字符串列表，按照插入顺序排序，可以在两端进行插入和删除操作。

### 基本操作

```bash
# 左侧插入
LPUSH mylist "a" "b" "c"  # 结果: c -> b -> a

# 右侧插入
RPUSH mylist "d" "e"      # 结果: c -> b -> a -> d -> e

# 获取列表元素
LRANGE mylist 0 -1        # 获取所有元素
LRANGE mylist 0 2         # 获取前 3 个元素

# 获取列表长度
LLEN mylist

# 获取指定索引的元素
LINDEX mylist 0

# 设置指定索引的元素
LSET mylist 0 "new_value"
```

### 弹出操作

```bash
# 左侧弹出
LPOP mylist

# 右侧弹出
RPOP mylist

# 阻塞式弹出（用于消息队列）
BLPOP mylist 30   # 30 秒超时
BRPOP mylist 30
```

### 修剪和删除

```bash
# 保留指定范围的元素
LTRIM mylist 0 99  # 只保留前 100 个元素

# 删除指定值的元素
LREM mylist 2 "value"  # 删除 2 个 "value"
```

### 应用场景

- **消息队列** - 使用 LPUSH + BRPOP 实现
- **最新列表** - 最新文章、最新评论
- **任务队列** - 异步任务处理

## Set（集合）

Set 是一个无序的字符串集合，集合中的元素是唯一的。

### 基本操作

```bash
# 添加元素
SADD myset "a" "b" "c"

# 获取所有元素
SMEMBERS myset

# 判断元素是否存在
SISMEMBER myset "a"

# 获取集合元素数量
SCARD myset

# 随机获取元素
SRANDMEMBER myset 2  # 随机获取 2 个元素

# 删除元素
SREM myset "a"

# 随机弹出元素
SPOP myset
```

### 集合运算

```bash
# 并集
SUNION set1 set2

# 交集
SINTER set1 set2

# 差集
SDIFF set1 set2

# 并集并存储
SUNIONSTORE result set1 set2

# 交集并存储
SINTERSTORE result set1 set2
```

### 应用场景

- **标签系统** - 文章标签、商品标签
- **共同好友** - 使用交集操作
- **推荐系统** - 基于标签的推荐
- **去重** - 利用集合的唯一性

## Hash（哈希）

Hash 是一个字段-值对的集合，适合存储对象。

### 基本操作

```bash
# 设置单个字段
HSET user:1001 name "张三"
HSET user:1001 age 25

# 设置多个字段
HMSET user:1001 name "张三" age 25 city "北京"

# 获取单个字段
HGET user:1001 name

# 获取多个字段
HMGET user:1001 name age

# 获取所有字段和值
HGETALL user:1001

# 获取所有字段名
HKEYS user:1001

# 获取所有值
HVALS user:1001

# 获取字段数量
HLEN user:1001

# 判断字段是否存在
HEXISTS user:1001 name

# 删除字段
HDEL user:1001 age
```

### 数值操作

```bash
# 字段值自增
HINCRBY user:1001 age 1
HINCRBY user:1001 score 10

# 浮点数自增
HINCRBYFLOAT user:1001 balance 10.5
```

### 应用场景

- **对象存储** - 用户信息、商品信息
- **购物车** - 商品 ID 为字段，数量为值
- **计数器** - 多个相关计数器

## Sorted Set（有序集合）

Sorted Set 是有序的字符串集合，每个元素关联一个分数（score），根据分数排序。

### 基本操作

```bash
# 添加元素
ZADD leaderboard 100 "player1"
ZADD leaderboard 200 "player2" 150 "player3"

# 获取元素数量
ZCARD leaderboard

# 获取分数
ZSCORE leaderboard "player1"

# 增加分数
ZINCRBY leaderboard 10 "player1"

# 删除元素
ZREM leaderboard "player1"
```

### 范围查询

```bash
# 按索引范围查询（从小到大）
ZRANGE leaderboard 0 9

# 按索引范围查询（从大到小）
ZREVRANGE leaderboard 0 9

# 带分数查询
ZRANGE leaderboard 0 9 WITHSCORES
ZREVRANGE leaderboard 0 9 WITHSCORES

# 按分数范围查询
ZRANGEBYSCORE leaderboard 100 200
ZREVRANGEBYSCORE leaderboard 200 100
```

### 排名操作

```bash
# 获取排名（从小到大，从 0 开始）
ZRANK leaderboard "player1"

# 获取排名（从大到小）
ZREVRANK leaderboard "player1"

# 按分数范围统计数量
ZCOUNT leaderboard 100 200
```

### 应用场景

- **排行榜** - 游戏排行、销售排行
- **优先级队列** - 任务优先级
- **延迟队列** - 使用时间戳作为分数
- **范围查询** - 价格区间、时间区间

## Bitmap（位图）

Bitmap 不是独立的数据类型，而是基于 String 类型的位操作。

### 基本操作

```bash
# 设置位
SETBIT user:1001:signin 0 1   # 第 0 天签到
SETBIT user:1001:signin 1 1   # 第 1 天签到

# 获取位
GETBIT user:1001:signin 0

# 统计位为 1 的数量
BITCOUNT user:1001:signin

# 位运算
BITOP AND result bitmap1 bitmap2  # 与运算
BITOP OR result bitmap1 bitmap2   # 或运算
BITOP XOR result bitmap1 bitmap2  # 异或运算
```

### 应用场景

- **用户签到** - 每天一位，365 天只需 46 字节
- **在线状态** - 用户 ID 对应位，1 表示在线
- **权限控制** - 每个权限一位
- **布隆过滤器** - 判断元素是否存在

## HyperLogLog

HyperLogLog 是一种概率数据结构，用于基数统计（去重计数）。

### 基本操作

```bash
# 添加元素
PFADD page:home:uv user1 user2 user3

# 获取基数（去重后的数量）
PFCOUNT page:home:uv

# 合并多个 HyperLogLog
PFMERGE result page:home:uv page:product:uv
```

### 特点

- **内存占用小** - 每个 HyperLogLog 只需 12 KB
- **有误差** - 误差率约 0.81%
- **不可逆** - 无法获取具体元素

### 应用场景

- **UV 统计** - 页面访问用户数
- **独立 IP 统计** - 网站独立访问 IP
- **搜索关键词统计** - 独立搜索词数量

## Stream（流）

Stream 是 Redis 5.0 引入的数据类型，用于消息队列和日志存储。

### 基本操作

```bash
# 添加消息
XADD mystream * field1 value1 field2 value2

# 读取消息
XREAD COUNT 10 STREAMS mystream 0

# 获取流长度
XLEN mystream

# 创建消费者组
XGROUP CREATE mystream mygroup 0

# 消费者组读取
XREADGROUP GROUP mygroup consumer1 COUNT 1 STREAMS mystream >

# 确认消息
XACK mystream mygroup message_id
```

### 应用场景

- **消息队列** - 支持消费者组
- **事件溯源** - 记录所有事件
- **日志收集** - 分布式日志收集

## 数据类型选择指南

| 场景         | 推荐类型       | 原因                         |
| ------------ | -------------- | ---------------------------- |
| 缓存对象     | Hash           | 结构化存储，部分字段更新     |
| 计数器       | String         | 原子自增操作                 |
| 排行榜       | Sorted Set     | 自动排序，范围查询           |
| 消息队列     | List 或 Stream | List 简单，Stream 功能更强大 |
| 标签系统     | Set            | 去重，集合运算               |
| 用户签到     | Bitmap         | 内存占用小                   |
| UV 统计      | HyperLogLog    | 基数统计，内存占用极小       |
| Session 存储 | String 或 Hash | String 简单，Hash 结构化     |
| 分布式锁     | String         | SETNX 原子操作               |
| 延迟队列     | Sorted Set     | 使用时间戳排序               |

## 最佳实践

### 1. 选择合适的数据类型

根据业务场景选择最合适的数据类型，避免滥用：

```bash
# ❌ 不好的做法：使用 String 存储对象
SET user:1001 '{"name":"张三","age":25,"city":"北京"}'

# ✅ 好的做法：使用 Hash 存储对象
HMSET user:1001 name "张三" age 25 city "北京"
```

### 2. 避免大键

控制集合类型的元素数量：

- List、Set、Hash、Sorted Set 元素不超过 1 万
- 如果数据量大，考虑分片存储

### 3. 合理使用过期时间

为键设置过期时间，避免内存溢出：

```bash
# 设置缓存过期时间
SETEX cache:product:1001 3600 "product_data"

# Hash 无法直接设置字段过期，需要设置整个键过期
HMSET user:1001 name "张三"
EXPIRE user:1001 3600
```

### 4. 使用 Pipeline 批量操作

减少网络往返次数，提高性能：

```java
Jedis jedis = new Jedis("localhost", 6379);
Pipeline pipeline = jedis.pipelined();

for (int i = 0; i < 1000; i++) {
    pipeline.set("key" + i, "value" + i);
}

pipeline.sync();
```

## 小结

Redis 提供了丰富的数据类型，每种类型都有其特定的应用场景：

- **String** - 最基本的类型，适用于缓存、计数器、分布式锁
- **List** - 有序列表，适用于消息队列、最新列表
- **Set** - 无序集合，适用于标签、去重
- **Hash** - 哈希表，适用于对象存储
- **Sorted Set** - 有序集合，适用于排行榜、优先级队列
- **Bitmap** - 位图，适用于签到、在线状态
- **HyperLogLog** - 基数统计，适用于 UV 统计
- **Stream** - 流，适用于消息队列、事件溯源

选择合适的数据类型，是高效使用 Redis 的关键。
