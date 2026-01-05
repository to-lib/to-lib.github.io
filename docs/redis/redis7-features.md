---
sidebar_position: 21
title: Redis 7.0+ 新特性
---

# Redis 7.0+ 新特性

Redis 7.0 是一个重大版本更新，引入了多项新功能和改进。本文详细介绍 Redis 7.0+ 的核心新特性。

## 版本概览

| 版本 | 发布时间 | 主要特性 |
|------|----------|----------|
| 7.0 | 2022-04 | Functions、Multi-part AOF、ACL v2 |
| 7.2 | 2023-08 | 更多功能增强、性能优化 |

## Redis Functions

### 概述

Redis Functions 是 Lua 脚本的升级版，提供更好的管理、持久化和集群支持。

**与 Lua 脚本的对比**：

| 特性 | EVAL (Lua) | Functions |
|------|------------|-----------|
| 持久化 | ❌ 不持久化 | ✅ 可以持久化 |
| 集群复制 | ❌ 需手动同步 | ✅ 自动复制 |
| 管理 | 基于 SHA1 | 命名函数库 |
| 重启保留 | ❌ 丢失 | ✅ 保留 |

### 创建函数

```lua
#!lua name=mylib

-- 定义函数
local function my_set(keys, args)
    local key = keys[1]
    local value = args[1]
    local ttl = args[2] or 0
    
    redis.call('SET', key, value)
    if tonumber(ttl) > 0 then
        redis.call('EXPIRE', key, ttl)
    end
    return 'OK'
end

-- 定义另一个函数
local function my_get_with_default(keys, args)
    local key = keys[1]
    local default = args[1] or ''
    
    local value = redis.call('GET', key)
    if value == false then
        return default
    end
    return value
end

-- 注册函数
redis.register_function('my_set', my_set)
redis.register_function('my_get_with_default', my_get_with_default)
```

### 加载和使用函数

```bash
# 加载函数库
cat mylib.lua | redis-cli -x FUNCTION LOAD REPLACE

# 调用函数
FCALL my_set 1 mykey myvalue 3600
FCALL my_get_with_default 1 mykey "default_value"

# 只读函数调用（可在从节点执行）
FCALL_RO my_get_with_default 1 mykey "default_value"
```

### 函数管理

```bash
# 列出所有函数库
FUNCTION LIST

# 列出函数库详情（包含代码）
FUNCTION LIST WITHCODE

# 删除函数库
FUNCTION DELETE mylib

# 导出函数库（用于备份/迁移）
FUNCTION DUMP

# 导入函数库
FUNCTION RESTORE <serialized-data>

# 清空所有函数
FUNCTION FLUSH
```

### 实用函数示例

**限流函数**：

```lua
#!lua name=ratelimit

local function rate_limit(keys, args)
    local key = keys[1]
    local limit = tonumber(args[1])
    local window = tonumber(args[2])
    
    local current = redis.call('INCR', key)
    if current == 1 then
        redis.call('EXPIRE', key, window)
    end
    
    if current > limit then
        return 0  -- 被限流
    end
    return 1  -- 允许
end

redis.register_function('rate_limit', rate_limit)
```

```bash
# 使用：每分钟最多 100 次请求
FCALL rate_limit 1 "ratelimit:user:1001" 100 60
```

**分布式 ID 生成**：

```lua
#!lua name=idgen

local function next_id(keys, args)
    local key = keys[1]
    local id = redis.call('INCR', key)
    local timestamp = redis.call('TIME')
    -- 返回格式: timestamp_id
    return timestamp[1] .. '_' .. id
end

redis.register_function('next_id', next_id)
```

## Multi-part AOF

### 概述

Redis 7.0 引入多部分 AOF，将 AOF 文件拆分成多个文件，解决了大 AOF 文件的管理问题。

### 文件结构

```
/var/lib/redis/appendonlydir/
├── appendonly.aof.1.base.rdb      # 基础 RDB 文件
├── appendonly.aof.2.incr.aof      # 增量 AOF 文件
├── appendonly.aof.3.incr.aof      # 增量 AOF 文件
└── appendonly.aof.manifest        # 清单文件
```

### 清单文件格式

```
file appendonly.aof.1.base.rdb seq 1 type b
file appendonly.aof.2.incr.aof seq 2 type i
file appendonly.aof.3.incr.aof seq 3 type i
```

### 配置

```conf
# AOF 目录名
appenddirname "appendonlydir"

# 开启 AOF
appendonly yes

# 使用混合持久化
aof-use-rdb-preamble yes
```

### 优势

1. **更好的重写性能**：重写时不会阻塞主进程
2. **更快的加载速度**：基础 RDB + 增量 AOF
3. **更简单的备份**：整个目录复制即可
4. **更少的磁盘占用**：自动清理过期文件

### 备份和恢复

```bash
# 备份整个 AOF 目录
tar -czf aof_backup.tar.gz /var/lib/redis/appendonlydir

# 恢复
tar -xzf aof_backup.tar.gz -C /var/lib/redis/
```

## ACL v2 增强

### Selector（选择器）

ACL v2 允许为用户定义多个权限选择器：

```bash
# 创建用户，使用选择器
ACL SETUSER alice on >password \
    (~app:* +@all -@dangerous) \
    (~admin:* +@all)
```

用户 `alice` 可以：

- 对 `app:*` 键执行所有非危险命令
- 对 `admin:*` 键执行所有命令

### 新增权限

```bash
# 允许特定频道的 Pub/Sub
ACL SETUSER bob on >password &channel:* +subscribe +publish

# 限制特定的哈希槽（集群）
ACL SETUSER charlie on >password %R~key:* +get
# %R 表示只读，~key:* 表示键模式
```

### 权限继承

```bash
# 使用 @all 然后排除
ACL SETUSER user1 on >pass +@all -@admin -@dangerous

# 使用 resetkeys 重置键权限
ACL SETUSER user1 resetkeys ~newpattern:*
```

### ACL 日志

```bash
# 查看 ACL 违规日志
ACL LOG 10

# 清空日志
ACL LOG RESET
```

## Sharded Pub/Sub

### 概述

传统 Pub/Sub 的消息会广播到集群所有节点，Sharded Pub/Sub 只在负责该频道的节点上处理。

### 使用

```bash
# 订阅 Sharded 频道
SSUBSCRIBE channel:orders

# 发布到 Sharded 频道
SPUBLISH channel:orders "new order: #12345"

# 取消订阅
SUNSUBSCRIBE channel:orders
```

### 与传统 Pub/Sub 对比

| 特性 | 传统 Pub/Sub | Sharded Pub/Sub |
|------|-------------|-----------------|
| 消息路由 | 广播到所有节点 | 只路由到对应槽位节点 |
| 集群支持 | 差 | 好 |
| 性能 | 低（集群） | 高 |
| 命令 | SUBSCRIBE/PUBLISH | SSUBSCRIBE/SPUBLISH |

### 配置

```conf
# 允许在主节点下线时继续 Sharded Pub/Sub
cluster-allow-pubsubshard-when-down yes
```

## 命令增强

### GETEX

获取值并更新过期时间：

```bash
# 获取值并设置过期时间
GETEX key EX 60
GETEX key PX 60000
GETEX key EXAT 1704067200
GETEX key PXAT 1704067200000

# 获取值并删除过期时间
GETEX key PERSIST
```

### GETDEL

获取并删除：

```bash
GETDEL key
# 原子地获取值并删除键
```

### COPY

复制键：

```bash
# 复制到同数据库
COPY source destination

# 复制到其他数据库
COPY source destination DB 1

# 覆盖已存在的键
COPY source destination REPLACE
```

### LMPOP / ZMPOP

从多个 List/ZSet 弹出：

```bash
# 从多个列表弹出
LMPOP 2 list1 list2 LEFT COUNT 3

# 从多个有序集合弹出
ZMPOP 2 zset1 zset2 MIN COUNT 2
```

### CLIENT NO-EVICT

保护客户端连接不被内存淘汰断开：

```bash
CLIENT NO-EVICT ON
# 执行关键操作
CLIENT NO-EVICT OFF
```

### LATENCY HISTOGRAM

查看延迟直方图：

```bash
LATENCY HISTOGRAM
LATENCY HISTOGRAM get set
```

## 性能改进

### 更快的 RDB 加载

- 优化了 RDB 加载的内存分配
- 并行加载某些数据结构
- 加载速度提升约 20-30%

### 更好的内存效率

```conf
# 优化 List 的内存使用
list-max-listpack-size -2

# 优化 Hash 的内存使用
hash-max-listpack-entries 512
hash-max-listpack-value 64

# 优化 ZSet 的内存使用
zset-max-listpack-entries 128
zset-max-listpack-value 64
```

### I/O 线程改进

```conf
# 更好的多线程 I/O 支持
io-threads 4
io-threads-do-reads yes
```

## 升级注意事项

### 兼容性变更

1. **RDB 版本**：Redis 7.0 使用 RDB 版本 10
2. **AOF 格式**：使用多部分 AOF
3. **复制协议**：部分变更

### 升级步骤

```bash
# 1. 备份数据
redis-cli BGSAVE

# 2. 停止 Redis 6.x
systemctl stop redis

# 3. 安装 Redis 7.0
# 根据你的安装方式进行

# 4. 更新配置文件（如需要）
# 添加新配置项

# 5. 启动 Redis 7.0
systemctl start redis

# 6. 验证
redis-cli INFO server | grep redis_version
```

### 回滚

```bash
# Redis 7.0 的 RDB 文件无法被 6.x 读取
# 如需回滚，需使用 6.x 版本的备份
```

## 新配置项

```conf
# === Functions ===
# 函数可以写入持久化文件
lua-replicate-commands yes

# === AOF ===
# 多部分 AOF 目录
appenddirname "appendonlydir"

# AOF 时间戳（用于 PITR）
aof-timestamp-enabled no

# === 集群 ===
# Sharded Pub/Sub 在主节点下线时是否可用
cluster-allow-pubsubshard-when-down yes

# === 性能 ===
# 启用惰性删除
lazyfree-lazy-user-del yes

# 活跃碎片整理配置
active-defrag-cycle-min 1
active-defrag-cycle-max 25
```

## 小结

| 特性 | 描述 | 适用场景 |
|------|------|----------|
| **Functions** | 可持久化的函数库 | 复杂业务逻辑、跨集群同步 |
| **Multi-part AOF** | 多文件 AOF | 大数据量、更快恢复 |
| **ACL v2** | 更细粒度权限 | 多租户、安全要求高 |
| **Sharded Pub/Sub** | 分片消息 | 集群环境的消息队列 |
| **新命令** | GETEX、COPY 等 | 简化操作 |

**升级建议**：

- ✅ 新项目直接使用 Redis 7.0+
- ✅ 旧项目评估兼容性后升级
- ✅ 充分测试后再上生产
- ✅ 准备好回滚方案
