---
sidebar_position: 12
title: 管道和批量操作
---

# Redis 管道和批量操作

管道（Pipeline）和批量操作是 Redis 性能优化的重要手段。

## Pipeline 管道

### 基本概念

Pipeline 允许客户端一次性发送多个命令，然后一次性读取所有响应，减少网络往返时间（RTT）。

**传统方式：**

```
Client: SET key1 value1  →  Server
Client: ← OK               Server
Client: SET key2 value2  →  Server
Client: ← OK               Server
```

3 次网络往返，耗时 = 3 \* RTT

**Pipeline 方式：**

```
Client: SET key1 value1  →  Server
        SET key2 value2  →
        SET key3 value3  →
Client: ← OK               Server
        ← OK
        ← OK
```

1 次网络往返，耗时 = 1 \* RTT

### Java 实现

#### Jedis Pipeline

```java
import redis.clients.jedis.Jedis;
import redis.clients.jedis.Pipeline;

public class PipelineExample {
    public static void main(String[] args) {
        Jedis jedis = new Jedis("localhost", 6379);

        // 创建 Pipeline
        Pipeline pipeline = jedis.pipelined();

        // 批量添加命令
        for (int i = 0; i < 10000; i++) {
            pipeline.set("key" + i, "value" + i);
        }

        // 执行并获取结果
        List<Object> results = pipeline.syncAndReturnAll();

        System.out.println("执行了 " + results.size() + " 个命令");

        jedis.close();
    }
}
```

#### 性能对比

```java
// 普通方式
long start = System.currentTimeMillis();
for (int i = 0; i < 10000; i++) {
    jedis.set("key" + i, "value" + i);
}
long normalTime = System.currentTimeMillis() - start;
System.out.println("普通方式: " + normalTime + "ms");

// Pipeline 方式
start = System.currentTimeMillis();
Pipeline pipeline = jedis.pipelined();
for (int i = 0; i < 10000; i++) {
    pipeline.set("key" + i, "value" + i);
}
pipeline.sync();
long pipelineTime = System.currentTimeMillis() - start;
System.out.println("Pipeline: " + pipelineTime + "ms");
System.out.println("提升: " + (normalTime / pipelineTime) + "x");
```

### Spring RedisTemplate

```java
@Autowired
private RedisTemplate<String, String> redisTemplate;

public void batchSet(Map<String, String> data) {
    redisTemplate.executePipelined(new SessionCallback<Object>() {
        @Override
        public <K, V> Object execute(RedisOperations<K, V> operations) {
            data.forEach((k, v) -> operations.opsForValue().set(k, v));
            return null;
        }
    });
}
```

## 批量操作命令

### MSET - 批量设置

```bash
# 一次设置多个键值对
MSET key1 "value1" key2 "value2" key3 "value3"

# 返回：OK
```

### MGET - 批量获取

```bash
# 一次获取多个值
MGET key1 key2 key3

# 返回数组
1) "value1"
2) "value2"
3) "value3"
```

### MSETNX - 批量设置（不存在时）

```bash
# 所有键都不存在时才设置
MSETNX key1 "value1" key2 "value2"

# 返回：1（成功）或 0（失败）
```

### DEL - 批量删除

```bash
# 删除多个键
DEL key1 key2 key3

# 返回：删除的键数量
```

### EXISTS - 批量检查

```bash
# 检查多个键是否存在
EXISTS key1 key2 key3

# 返回：存在的键数量
```

## Lua 脚本

### 基本用法

Lua 脚本在 Redis 服务器端原子性执行，适合复杂的批量操作。

```bash
# EVAL 命令
EVAL script numkeys key [key ...] arg [arg ...]

# 示例：批量设置并设置过期时间
EVAL "
  for i, key in ipairs(KEYS) do
    redis.call('SET', key, ARGV[i])
    redis.call('EXPIRE', key, 3600)
  end
  return #KEYS
" 3 key1 key2 key3 value1 value2 value3
```

### Java 调用 Lua

```java
// 加载脚本
String script =
    "for i, key in ipairs(KEYS) do " +
    "  redis.call('SET', key, ARGV[i]) " +
    "  redis.call('EXPIRE', key, ARGV[#ARGV]) " +
    "end " +
    "return #KEYS";

// 执行脚本
List<String> keys = Arrays.asList("user:1", "user:2", "user:3");
List<String> args = Arrays.asList("Alice", "Bob", "Charlie", "3600");

Object result = jedis.eval(
    script,
    keys,
    args
);

System.out.println("设置了 " + result + " 个键");
```

### SCRIPT LOAD / EVALSHA

```java
// 加载脚本，返回 SHA1
String sha = jedis.scriptLoad(script);

// 使用 SHA1 执行（性能更好）
Object result = jedis.evalsha(sha, keys, args);
```

### 实战案例

#### 分布式限流

```lua
-- 限流脚本
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local expire = tonumber(ARGV[2])

local current = redis.call('GET', key)

if current and tonumber(current) >= limit then
    return 0  -- 超过限制
end

redis.call('INCR', key)
redis.call('EXPIRE', key, expire)
return 1  -- 允许访问
```

```java
String script =
    "local key = KEYS[1] " +
    "local limit = tonumber(ARGV[1]) " +
    "local expire = tonumber(ARGV[2]) " +
    "local current = redis.call('GET', key) " +
    "if current and tonumber(current) >= limit then " +
    "  return 0 " +
    "end " +
    "redis.call('INCR', key) " +
    "redis.call('EXPIRE', key, expire) " +
    "return 1";

Long allowed = (Long) jedis.eval(
    script,
    Collections.singletonList("rate:limit:user:1001"),
    Arrays.asList("10", "60")  // 60秒内最多10次
);

if (allowed == 1) {
    System.out.println("允许访问");
} else {
    System.out.println("访问过于频繁");
}
```

#### 库存扣减

```lua
-- 扣减库存脚本
local key = KEYS[1]
local quantity = tonumber(ARGV[1])

local stock = redis.call('GET', key)
if not stock or tonumber(stock) < quantity then
    return 0  -- 库存不足
end

redis.call('DECRBY', key, quantity)
return 1  -- 扣减成功
```

## Transaction vs Pipeline vs Lua

### 特性对比

| 特性     | Transaction | Pipeline  | Lua Script   |
| -------- | ----------- | --------- | ------------ |
| 原子性   | ✅ 保证     | ❌ 不保证 | ✅ 保证      |
| 性能提升 | ⭐⭐        | ⭐⭐⭐    | ⭐⭐⭐       |
| 复杂逻辑 | ❌ 不支持   | ❌ 不支持 | ✅ 支持      |
| 适用场景 | 简单事务    | 批量操作  | 复杂原子操作 |

### Transaction（MULTI/EXEC）

```bash
MULTI
SET key1 value1
SET key2 value2
INCR counter
EXEC
```

特点：

- 保证原子性
- 不支持条件判断
- 命令在 EXEC 时一次性执行

### Pipeline

```bash
# Pipeline 只是批量发送，不保证原子性
SET key1 value1
SET key2 value2
INCR counter
```

特点：

- 减少网络往返
- 不保证原子性
- 性能最好

### Lua Script

```lua
local result = redis.call('GET', KEYS[1])
if tonumber(result) > 100 then
    redis.call('SET', KEYS[1], 0)
    return 1
end
return 0
```

特点：

- 原子性执行
- 支持复杂逻辑
- 服务器端执行

## 性能优化

### 1. 合理批量大小

```java
// 不要一次发送过多命令
int BATCH_SIZE = 1000;

for (int i = 0; i < totalCount; i += BATCH_SIZE) {
    Pipeline pipeline = jedis.pipelined();

    for (int j = i; j < Math.min(i + BATCH_SIZE, totalCount); j++) {
        pipeline.set("key" + j, "value" + j);
    }

    pipeline.sync();
}
```

### 2. Pipeline + Transaction

```java
// 结合 Pipeline 和 Transaction
Pipeline pipeline = jedis.pipelined();
pipeline.multi();  // 开始事务

for (int i = 0; i < 1000; i++) {
    pipeline.set("key" + i, "value" + i);
}

pipeline.exec();  // 执行事务
pipeline.sync();  // 同步Pipeline
```

### 3. Lua 脚本优化

```lua
-- 避免在循环中调用 redis.call
-- 不好
for i = 1, 10000 do
    redis.call('SET', 'key' .. i, 'value')
end

-- 更好：批量操作
local keys = {}
local values = {}
for i = 1, 10000 do
    table.insert(keys, 'key' .. i)
    table.insert(values, 'value')
end
redis.call('MSET', unpack(keys), unpack(values))
```

## 注意事项

### 1. Pipeline 不保证原子性

```java
// 可能部分成功、部分失败
Pipeline pipeline = jedis.pipelined();
pipeline.set("key1", "value1");  // 可能成功
// 如果这里网络断开...
pipeline.set("key2", "value2");  // 可能失败
pipeline.sync();
```

### 2. Lua 脚本阻塞

```lua
-- 避免长时间运行的脚本
-- 不好：死循环
while true do
    redis.call('INCR', 'counter')
end

-- 好：有明确的结束条件
for i = 1, 1000 do
    redis.call('INCR', 'counter')
end
```

### 3. 内存占用

```java
// Pipeline 在客户端缓冲所有命令
// 避免一次发送过多
Pipeline pipeline = jedis.pipelined();
for (int i = 0; i < 1000000; i++) {  // 100万条可能导致内存问题
    pipeline.set("key" + i, "value" + i);
}
```

## 最佳实践

### 1. 批量导入数据

```java
public void importData(List<Map<String, String>> dataList) {
    int BATCH_SIZE = 1000;

    for (int i = 0; i < dataList.size(); i += BATCH_SIZE) {
        Pipeline pipeline = jedis.pipelined();

        int end = Math.min(i + BATCH_SIZE, dataList.size());
        for (int j = i; j < end; j++) {
            Map<String, String> data = dataList.get(j);
            pipeline.hmset("user:" + data.get("id"), data);
        }

        pipeline.sync();

        System.out.println("已导入: " + end + "/" + dataList.size());
    }
}
```

### 2. 批量删除

```java
public void deleteByPattern(String pattern) {
    Set<String> keys = jedis.keys(pattern);

    if (!keys.isEmpty()) {
        Pipeline pipeline = jedis.pipelined();
        keys.forEach(pipeline::del);
        pipeline.sync();
    }
}
```

### 3. 原子性计数

```java
// 使用 Lua 保证原子性
String script =
    "local current = redis.call('INCRBY', KEYS[1], ARGV[1]) " +
    "if current > tonumber(ARGV[2]) then " +
    "  redis.call('SET', KEYS[1], ARGV[2]) " +
    "  return ARGV[2] " +
    "end " +
    "return current";

Long count = (Long) jedis.eval(
    script,
    Collections.singletonList("counter"),
    Arrays.asList("10", "1000")  // 增加10，最大值1000
);
```

## 总结

- ✅ Pipeline 适合批量操作，性能提升显著
- ✅ Lua 脚本保证原子性，支持复杂逻辑
- ✅ 合理选择批量大小，避免内存问题
- ⚠️ Pipeline 不保证原子性
- ⚠️ Lua 脚本会阻塞 Redis
- 💡 根据场景选择合适的批量操作方式

掌握这些技巧，能大幅提升 Redis 性能！
