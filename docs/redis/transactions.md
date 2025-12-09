---
sidebar_position: 8
title: Redis 事务
---

# Redis 事务

Redis 事务提供了一种将多个命令打包执行的机制，保证这些命令的原子性执行。

## 事务基础

### 基本命令

```bash
# 开启事务
MULTI

# 命令入队
SET key1 "value1"
SET key2 "value2"
INCR counter

# 执行事务
EXEC

# 取消事务
DISCARD
```

### 执行流程

```bash
127.0.0.1:6379> MULTI
OK
127.0.0.1:6379> SET name "张三"
QUEUED
127.0.0.1:6379> SET age 25
QUEUED
127.0.0.1:6379> EXEC
1) OK
2) OK
```

## WATCH 命令

`WATCH` 提供了乐观锁机制，监控键的变化。如果被监控的键在事务执行前被修改，事务将失败。

### 基本用法

```bash
# 监控键
WATCH key1

# 开启事务
MULTI
SET key1 "new_value"
EXEC

# 如果 key1 在 WATCH 后、EXEC 前被修改，EXEC 返回 nil
```

### 实现乐观锁

```bash
# 客户端 1
WATCH balance:user:1001
GET balance:user:1001  # 假设余额为 100
MULTI
DECRBY balance:user:1001 50
EXEC  # 如果期间余额被修改，返回 nil

# 客户端 2（同时修改余额）
SET balance:user:1001 90
# 客户端 1 的事务会失败
```

### 取消监控

```bash
# 取消所有监控
UNWATCH
```

## 事务特性

### Redis 事务 vs 传统 ACID

| 特性       | 传统数据库事务 | Redis 事务    |
| ---------- | -------------- | ------------- |
| 原子性 (A) | ✅ 支持        | ⚠️ 部分支持   |
| 一致性 (C) | ✅ 支持        | ✅ 支持       |
| 隔离性 (I) | ✅ 支持        | ❌ 不支持     |
| 持久性 (D) | ✅ 支持        | ⚠️ 取决于配置 |

### 原子性

Redis 事务保证所有命令作为一个整体执行，但**不支持回滚**。

#### 编译时错误（语法错误）

命令入队时发现错误,整个事务不会执行：

```bash
127.0.0.1:6379> MULTI
OK
127.0.0.1:6379> SET key1 "value1"
QUEUED
127.0.0.1:6379> INVALID_COMMAND
(error) ERR unknown command 'INVALID_COMMAND'
127.0.0.1:6379> EXEC
(error) EXECABORT Transaction discarded
```

#### 运行时错误

命令执行时发现错误，**已执行的命令不会回滚**：

```bash
127.0.0.1:6379> MULTI
OK
127.0.0.1:6379> SET key "value"
QUEUED
127.0.0.1:6379> INCR key  # key 是字符串，无法自增
QUEUED
127.0.0.1:6379> SET key2 "value2"
QUEUED
127.0.0.1:6379> EXEC
1) OK
2) (error) ERR value is not an integer
3) OK  # 即使 INCR 失败，SET key2 仍然执行
```

### 隔离性

Redis 是单线程模型，事务执行期间不会被其他命令打断，具有隔离性。但**不支持事务隔离级别**。

### 持久性

取决于持久化配置：

- **无持久化** - 不保证持久性
- **RDB** - 定期持久化，可能丢失数据
- **AOF (everysec)** - 最多丢失 1 秒数据
- **AOF (always)** - 每个写命令都持久化

## 应用场景

### 1. 批量操作

将多个操作打包执行，提高性能：

```bash
MULTI
SET user:1001:name "张三"
SET user:1001:age 25
SET user:1001:city "北京"
EXEC
```

### 2. 库存扣减

使用 WATCH 实现乐观锁，防止超卖：

```bash
WATCH stock:product:1001
stock = GET stock:product:1001

if stock > 0:
    MULTI
    DECR stock:product:1001
    # 创建订单等其他操作
    EXEC
else:
    UNWATCH
    # 库存不足
```

### 3. 转账操作

```bash
# 转账：user:1001 -> user:1002，金额 100
WATCH balance:user:1001 balance:user:1002

balance1 = GET balance:user:1001
if balance1 >= 100:
    MULTI
    DECRBY balance:user:1001 100
    INCRBY balance:user:1002 100
    EXEC
else:
    UNWATCH
    # 余额不足
```

## Lua 脚本

Lua 脚本提供了比事务更强大的功能，支持条件判断、循环等复杂逻辑。

### 基本用法

```bash
# EVAL script numkeys key [key ...] arg [arg ...]
EVAL "return redis.call('SET', KEYS[1], ARGV[1])" 1 mykey myvalue
```

### 脚本示例

#### 1. 原子自增并返回

```bash
# Lua 脚本
local current = redis.call('GET', KEYS[1])
if current == false then
    current = 0
end
current = current + ARGV[1]
redis.call('SET', KEYS[1], current)
return current

# 执行
EVAL "..." 1 counter 10
```

#### 2. 分布式锁

```lua
-- 获取锁
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
```

```bash
# 释放锁
EVAL "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end" 1 lock:resource unique_id
```

#### 3. 限流（令牌桶）

```lua
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local interval = tonumber(ARGV[2])

local current = tonumber(redis.call('get', key) or "0")

if current + 1 > limit then
    return 0
else
    redis.call('incr', key)
    if current == 0 then
        redis.call('expire', key, interval)
    end
    return 1
end
```

```bash
# 限流：每 60 秒最多 100 次请求
EVAL "..." 1 rate_limit:user:1001 100 60
```

### 加载脚本

使用 `SCRIPT LOAD` 预加载脚本，返回 SHA1 值：

```bash
# 加载脚本
SCRIPT LOAD "return redis.call('SET', KEYS[1], ARGV[1])"
# 返回: "c686f316aaf1eb01d5a4de1b0b63cd233010e63d"

# 使用 SHA1 执行
EVALSHA "c686f316aaf1eb01d5a4de1b0b63cd233010e63d" 1 mykey myvalue
```

### 脚本管理

```bash
# 检查脚本是否存在
SCRIPT EXISTS sha1 [sha1 ...]

# 清空脚本缓存
SCRIPT FLUSH

# 杀死正在执行的脚本（仅限只读脚本）
SCRIPT KILL
```

### Lua 脚本优点

1. **原子性** - 脚本执行期间不会被打断
2. **减少网络往返** - 一次发送所有逻辑
3. **复杂逻辑** - 支持条件、循环等
4. **复用** - 使用 SCRIPT LOAD 预加载

## Pipeline vs 事务 vs Lua

| 特性         | Pipeline       | 事务 (MULTI/EXEC) | Lua 脚本     |
| ------------ | -------------- | ----------------- | ------------ |
| 原子性       | ❌ 不保证      | ✅ 保证           | ✅ 保证      |
| 减少网络往返 | ✅ 是          | ⚠️ 部分           | ✅ 是        |
| 条件判断     | ❌ 不支持      | ❌ 不支持         | ✅ 支持      |
| 回滚         | ❌ 不支持      | ❌ 不支持         | ❌ 不支持    |
| 适用场景     | 批量非原子操作 | 批量原子操作      | 复杂原子逻辑 |

### 使用建议

- **Pipeline** - 批量操作，不需要原子性
- **事务** - 简单的批量原子操作
- **Lua 脚本** - 复杂的原子逻辑，需要条件判断

## 最佳实践

### 1. 使用 WATCH 实现乐观锁

```python
def transfer(from_user, to_user, amount):
    while True:
        # 监控源账户
        redis.watch(f'balance:{from_user}')

        balance = int(redis.get(f'balance:{from_user}') or 0)
        if balance < amount:
            redis.unwatch()
            return False

        # 开启事务
        pipe = redis.pipeline()
        pipe.decrby(f'balance:{from_user}', amount)
        pipe.incrby(f'balance:{to_user}', amount)

        try:
            pipe.execute()
            return True
        except redis.WatchError:
            # 重试
            continue
```

### 2. 避免大事务

事务包含的命令不宜过多，影响性能：

```bash
# ❌ 不好的做法
MULTI
# 1000 个命令...
EXEC

# ✅ 好的做法：分批执行或使用 Pipeline
```

### 3. 优先使用 Lua 脚本

对于复杂逻辑，优先使用 Lua 脚本：

```bash
# ❌ 使用事务（需要客户端判断）
WATCH stock
stock = GET stock
if stock > 0:
    MULTI
    DECR stock
    EXEC

# ✅ 使用 Lua 脚本（一次完成）
EVAL "local stock = redis.call('GET', KEYS[1]); if tonumber(stock) > 0 then return redis.call('DECR', KEYS[1]) else return nil end" 1 stock
```

### 4. 处理 WATCH 失败

WATCH 失败时需要重试：

```java
int maxRetry = 3;
for (int i = 0; i < maxRetry; i++) {
    jedis.watch("key");

    // 读取数据
    String value = jedis.get("key");

    // 处理逻辑
    Transaction tx = jedis.multi();
    tx.set("key", newValue);

    List<Object> result = tx.exec();
    if (result != null) {
        // 成功
        break;
    }
    // 失败，重试
}
```

## 常见问题

### 1. Redis 事务支持回滚吗？

不支持。运行时错误不会导致事务回滚，已执行的命令会生效。

### 2. WATCH 监控多个键会怎样？

任何一个被监控的键被修改，事务都会失败：

```bash
WATCH key1 key2 key3
MULTI
# ...
EXEC  # 如果 key1、key2、key3 任一被修改，返回 nil
```

### 3. Lua 脚本执行时间过长怎么办？

- 配置 `lua-time-limit`（默认 5 秒）
- 使用 `SCRIPT KILL` 杀死只读脚本
- 避免在 Lua 脚本中执行耗时操作

### 4. 事务和 Lua 脚本如何选择？

- **简单批量操作** - 使用事务
- **需要条件判断** - 使用 Lua 脚本
- **复杂业务逻辑** - 使用 Lua 脚本

## 小结

Redis 事务提供了命令的原子性执行：

- **MULTI/EXEC** - 批量原子操作
- **WATCH** - 乐观锁机制
- **Lua 脚本** - 复杂原子逻辑

关键特性：

- 不支持回滚
- 单线程保证隔离性
- WATCH 实现乐观锁
- Lua 脚本功能更强大

适用场景：

- 批量原子操作（事务）
- 库存扣减、转账（WATCH + 乐观锁）
- 复杂原子逻辑（Lua 脚本）
