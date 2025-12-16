---
sidebar_position: 11
title: Lua 脚本
---

# Redis Lua 脚本

Lua 脚本是 Redis 的强大功能，允许在服务器端原子性执行复杂逻辑。

## 为什么使用 Lua 脚本

### 优势

1. **原子性** - 脚本执行期间不会被其他命令打断
2. **减少网络往返** - 一次发送，服务器端执行
3. **复杂逻辑** - 支持条件判断、循环、函数
4. **可复用** - 脚本可缓存，使用 SHA1 调用

### 适用场景

- 分布式锁
- 限流器
- 库存扣减
- 原子性计数器
- 条件更新

## 基本语法

### EVAL 命令

```bash
# 语法
EVAL script numkeys key [key ...] arg [arg ...]

# 示例：简单脚本
EVAL "return 'Hello, Redis!'" 0

# 带参数的脚本
EVAL "return {KEYS[1], KEYS[2], ARGV[1], ARGV[2]}" 2 key1 key2 arg1 arg2
```

参数说明：

- `script` - Lua 脚本内容
- `numkeys` - 键的数量
- `key [key ...]` - 键列表，在脚本中通过 `KEYS[n]` 访问
- `arg [arg ...]` - 参数列表，在脚本中通过 `ARGV[n]` 访问

### EVALSHA 命令

使用脚本的 SHA1 哈希值调用（性能更好）：

```bash
# 1. 加载脚本，获取 SHA1
SCRIPT LOAD "return 'Hello'"
# "6b1bf486c81ceb7edf3c093f4a7fc83b6cd2"

# 2. 使用 SHA1 执行脚本
EVALSHA "6b1bf486c81ceb7edf3c093f4a7fc83b6c" 0
```

## Redis Lua API

### redis.call()

执行 Redis 命令，错误时抛出异常：

```lua
local value = redis.call('GET', KEYS[1])
redis.call('SET', KEYS[1], 'new_value')
redis.call('INCR', KEYS[2])
```

### redis.pcall()

执行 Redis 命令，错误时返回错误对象：

```lua
local result = redis.pcall('GET', KEYS[1])
if result.err then
    -- 处理错误
    return result.err
end
```

### 返回值

```lua
-- 返回字符串
return "Hello"

-- 返回数字
return 42

-- 返回数组（table）
return {1, 2, 3}

-- 返回多个值
return {KEYS[1], ARGV[1]}

-- 返回 nil
return nil
-- 或
return false
```

## Lua 基础语法

### 变量和数据类型

```lua
-- 变量
local name = "Redis"
local count = 10
local flag = true

-- 表（数组/字典）
local arr = {1, 2, 3}
local dict = {name = "Redis", version = "7.0"}

-- 访问表元素
print(arr[1])       -- 1（Lua 数组从 1 开始）
print(dict.name)    -- Redis
print(dict["name"]) -- Redis
```

### 条件判断

```lua
local value = redis.call('GET', KEYS[1])

if value == false then
    -- 键不存在
    return nil
elseif tonumber(value) > 100 then
    -- 值大于 100
    return "large"
else
    -- 其他情况
    return "small"
end
```

### 循环

```lua
-- for 循环
for i = 1, 10 do
    redis.call('SET', 'key' .. i, i)
end

-- 遍历数组
for i, key in ipairs(KEYS) do
    redis.call('DEL', key)
end

-- 遍历字典
local data = {a = 1, b = 2, c = 3}
for k, v in pairs(data) do
    print(k, v)
end

-- while 循环
local i = 1
while i <= 10 do
    redis.call('SET', 'key' .. i, i)
    i = i + 1
end
```

### 函数

```lua
-- 定义函数
local function add(a, b)
    return a + b
end

-- 调用函数
local result = add(10, 20)
```

### 字符串操作

```lua
-- 拼接
local str = "Hello" .. " " .. "World"

-- 长度
local len = #str

-- 转换
local num = tonumber("42")
local str = tostring(42)
```

## 实战案例

### 1. 分布式锁

**获取锁**：

```lua
-- lock.lua
-- KEYS[1]: 锁的键名
-- ARGV[1]: 锁的唯一标识（UUID）
-- ARGV[2]: 过期时间（秒）

if redis.call('SETNX', KEYS[1], ARGV[1]) == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[2])
    return 1
else
    return 0
end
```

**释放锁**（原子性判断）：

```lua
-- unlock.lua
-- KEYS[1]: 锁的键名
-- ARGV[1]: 锁的唯一标识

if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
```

**Java 调用**：

```java
// 获取锁
String lockScript =
    "if redis.call('SETNX', KEYS[1], ARGV[1]) == 1 then " +
    "    redis.call('EXPIRE', KEYS[1], ARGV[2]) " +
    "    return 1 " +
    "else " +
    "    return 0 " +
    "end";

String lockKey = "lock:resource:1001";
String lockValue = UUID.randomUUID().toString();
int expireSeconds = 30;

Long result = (Long) jedis.eval(
    lockScript,
    Collections.singletonList(lockKey),
    Arrays.asList(lockValue, String.valueOf(expireSeconds))
);

if (result == 1) {
    try {
        // 获取锁成功，执行业务逻辑
        doSomething();
    } finally {
        // 释放锁
        String unlockScript =
            "if redis.call('GET', KEYS[1]) == ARGV[1] then " +
            "    return redis.call('DEL', KEYS[1]) " +
            "else " +
            "    return 0 " +
            "end";

        jedis.eval(unlockScript,
            Collections.singletonList(lockKey),
            Collections.singletonList(lockValue));
    }
}
```

### 2. 限流器（滑动窗口）

```lua
-- rate_limit.lua
-- KEYS[1]: 限流键
-- ARGV[1]: 窗口大小（秒）
-- ARGV[2]: 最大请求数
-- ARGV[3]: 当前时间戳（毫秒）

local key = KEYS[1]
local window = tonumber(ARGV[1]) * 1000  -- 转为毫秒
local limit = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

-- 删除窗口外的记录
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

-- 统计当前窗口内的请求数
local count = redis.call('ZCARD', key)

if count < limit then
    -- 未超过限制，添加当前请求
    redis.call('ZADD', key, now, now)
    redis.call('EXPIRE', key, ARGV[1])
    return 1  -- 允许
else
    return 0  -- 拒绝
end
```

### 3. 库存扣减

```lua
-- deduct_stock.lua
-- KEYS[1]: 库存键
-- ARGV[1]: 扣减数量

local stock = redis.call('GET', KEYS[1])

if stock == false then
    return -1  -- 库存不存在
end

stock = tonumber(stock)
local quantity = tonumber(ARGV[1])

if stock < quantity then
    return 0  -- 库存不足
end

redis.call('DECRBY', KEYS[1], quantity)
return 1  -- 扣减成功
```

### 4. 原子性计数器（带上限）

```lua
-- counter_with_limit.lua
-- KEYS[1]: 计数器键
-- ARGV[1]: 增加值
-- ARGV[2]: 上限值

local current = redis.call('GET', KEYS[1])

if current == false then
    current = 0
else
    current = tonumber(current)
end

local increment = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])

local new_value = current + increment

if new_value > limit then
    new_value = limit
end

redis.call('SET', KEYS[1], new_value)
return new_value
```

### 5. 延迟队列（取出到期任务）

```lua
-- fetch_delayed_tasks.lua
-- KEYS[1]: 延迟队列键
-- ARGV[1]: 当前时间戳
-- ARGV[2]: 批量大小

local key = KEYS[1]
local now = tonumber(ARGV[1])
local batch_size = tonumber(ARGV[2])

-- 获取到期的任务
local tasks = redis.call('ZRANGEBYSCORE', key, 0, now, 'LIMIT', 0, batch_size)

if #tasks > 0 then
    -- 删除这些任务
    redis.call('ZREM', key, unpack(tasks))
end

return tasks
```

## 脚本管理

### 脚本加载和缓存

```bash
# 加载脚本到缓存
SCRIPT LOAD "return 'Hello'"
# 返回 SHA1: "6b1bf486c81ceb7edf3c093f4a7fc83b6cd2"

# 使用 SHA1 执行
EVALSHA "6b1bf486c81ceb7edf3c093f4a7fc83b6cd2" 0

# 检查脚本是否存在
SCRIPT EXISTS "6b1bf486c81ceb7edf3c093f4a7fc83b6cd2"

# 清空脚本缓存
SCRIPT FLUSH

# 杀死正在执行的脚本（仅限只读脚本）
SCRIPT KILL
```

### Java 封装

```java
public class RedisScript {
    private final Jedis jedis;
    private final String sha;

    public RedisScript(Jedis jedis, String script) {
        this.jedis = jedis;
        this.sha = jedis.scriptLoad(script);
    }

    public Object execute(List<String> keys, List<String> args) {
        try {
            return jedis.evalsha(sha, keys, args);
        } catch (JedisNoScriptException e) {
            // 脚本不存在，重新加载
            return jedis.eval(script, keys, args);
        }
    }
}

// 使用
RedisScript lockScript = new RedisScript(jedis,
    "if redis.call('SETNX', KEYS[1], ARGV[1]) == 1 then " +
    "    redis.call('EXPIRE', KEYS[1], ARGV[2]) " +
    "    return 1 " +
    "else return 0 end"
);

Object result = lockScript.execute(
    Collections.singletonList("lock:key"),
    Arrays.asList("uuid", "30")
);
```

## 调试和优化

### 调试脚本

```bash
# 使用 redis.log() 输出日志
EVAL "redis.log(redis.LOG_WARNING, 'Debug: ' .. KEYS[1])" 1 mykey

# 查看 Redis 日志
tail -f /var/log/redis/redis.log
```

### 性能优化

**1. 避免大循环**：

```lua
-- ❌ 不好：循环 100 万次
for i = 1, 1000000 do
    redis.call('SET', 'key' .. i, 'value')
end

-- ✅ 好：分批处理
for i = 1, 1000 do
    redis.call('SET', 'key' .. i, 'value')
end
```

**2. 使用批量命令**：

```lua
-- ❌ 不好：多次调用
for i, key in ipairs(KEYS) do
    redis.call('DEL', key)
end

-- ✅ 好：一次调用
redis.call('DEL', unpack(KEYS))
```

**3. 预加载脚本**：

```java
// 应用启动时加载
String sha = jedis.scriptLoad(script);

// 运行时使用 EVALSHA
jedis.evalsha(sha, keys, args);
```

### 超时处理

```conf
# redis.conf
# Lua 脚本超时时间（毫秒）
lua-time-limit 5000
```

脚本超时后：

- 只读脚本可被 `SCRIPT KILL` 终止
- 写脚本只能通过 `SHUTDOWN NOSAVE` 终止

## 最佳实践

### 1. 使用 KEYS 和 ARGV

```lua
-- ✅ 好：使用 KEYS 和 ARGV
local key = KEYS[1]
local value = ARGV[1]
redis.call('SET', key, value)

-- ❌ 不好：硬编码键名（影响集群模式）
redis.call('SET', 'fixed_key', 'value')
```

### 2. 处理 nil 值

```lua
local value = redis.call('GET', KEYS[1])

-- 注意：GET 返回 nil 时，Lua 中是 false
if value == false then
    return nil
end
```

### 3. 类型转换

```lua
-- 字符串转数字
local num = tonumber(ARGV[1])

-- 数字转字符串
local str = tostring(42)
```

### 4. 错误处理

```lua
-- 使用 pcall 安全调用
local ok, result = pcall(function()
    return redis.call('INCR', KEYS[1])
end)

if not ok then
    return {err = result}
end

return result
```

### 5. 集群模式注意事项

在 Redis Cluster 中，所有键必须在同一个槽位：

```lua
-- ✅ 使用哈希标签确保同一槽位
-- KEYS: {user}:1001, {user}:1002
redis.call('MGET', KEYS[1], KEYS[2])

-- ❌ 不同槽位会报错
-- KEYS: user:1001, order:1001
redis.call('MGET', KEYS[1], KEYS[2])  -- 可能失败
```

## 常见问题

### Q: 脚本执行超时怎么办？

配置 `lua-time-limit`，使用 `SCRIPT KILL` 终止。

### Q: 脚本和事务有什么区别？

| 特性     | 脚本   | 事务 |
| -------- | ------ | ---- |
| 原子性   | ✅     | ✅   |
| 条件判断 | ✅     | ❌   |
| 回滚     | ❌     | ❌   |
| 性能     | ⭐⭐⭐ | ⭐⭐ |

### Q: 如何在集群模式使用脚本？

确保所有 KEYS 使用相同的哈希标签：`{tag}:key1`、`{tag}:key2`。

## 小结

- ✅ Lua 脚本保证原子性，适合复杂逻辑
- ✅ 使用 EVALSHA 提高性能
- ✅ 通过 KEYS 和 ARGV 传递参数
- ✅ 注意脚本超时和集群模式限制
- ✅ 常见场景：分布式锁、限流、库存扣减

掌握 Lua 脚本，解锁 Redis 的全部能力！
