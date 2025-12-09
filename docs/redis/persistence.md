---
sidebar_position: 4
title: Redis 持久化
---

# Redis 持久化

Redis 是内存数据库，数据存储在内存中。为了保证数据不丢失，Redis 提供了两种持久化机制：RDB（快照）和 AOF（追加文件），以及 Redis 4.0+ 的混合持久化。

## RDB 持久化

RDB（Redis Database）是 Redis 的默认持久化方式，它将某个时间点的内存数据快照保存到磁盘文件中。

### 工作原理

1. Redis 调用 `fork()` 创建子进程
2. 子进程将数据集写入临时 RDB 文件
3. 写入完成后，用新文件替换旧文件

### 触发方式

#### 1. 自动触发

通过配置文件设置自动保存规则：

```conf
# redis.conf
save 900 1      # 900 秒内至少 1 个键被修改
save 300 10     # 300 秒内至少 10 个键被修改
save 60 10000   # 60 秒内至少 10000 个键被修改
```

#### 2. 手动触发

```bash
# SAVE 命令（阻塞）
SAVE

# BGSAVE 命令（后台保存，不阻塞）
BGSAVE

# 查看最后一次保存时间
LASTSAVE
```

#### 3. 其他触发

- 执行 `SHUTDOWN` 命令
- 执行 `FLUSHALL` 命令
- 主从复制时，主节点自动触发

### 配置参数

```conf
# RDB 文件名
dbfilename dump.rdb

# 保存目录
dir /var/lib/redis

# 保存失败时停止写入
stop-writes-on-bgsave-error yes

# 压缩 RDB 文件
rdbcompression yes

# 校验和检查
rdbchecksum yes
```

### 优点

- **性能好** - 只需定期持久化，对性能影响小
- **恢复快** - 恢复大数据集速度快
- **文件紧凑** - 适合备份和灾难恢复

### 缺点

- **数据安全性低** - 可能丢失最后一次快照之后的数据
- **fork 开销** - 数据集大时，fork 可能耗时较长
- **不适合实时持久化** - 无法做到秒级持久化

## AOF 持久化

AOF（Append Only File）记录每个写操作，通过重放这些操作来恢复数据。

### 工作原理

1. 客户端执行写命令
2. Redis 将命令追加到 AOF 缓冲区
3. 根据策略将缓冲区内容写入 AOF 文件
4. 定期重写 AOF 文件，压缩文件大小

### 开启 AOF

```conf
# 开启 AOF
appendonly yes

# AOF 文件名
appendfilename "appendonly.aof"

# 保存目录
dir /var/lib/redis
```

### 同步策略

AOF 提供三种同步策略，控制数据写入磁盘的时机：

```conf
# always - 每个写命令都同步（最安全，性能最差）
appendfsync always

# everysec - 每秒同步一次（推荐，性能和安全性平衡）
appendfsync everysec

# no - 由操作系统决定何时同步（性能最好，安全性最差）
appendfsync no
```

**推荐使用 `everysec`**，它在性能和数据安全性之间取得了很好的平衡。

### AOF 重写

随着时间推移，AOF 文件会变得越来越大。Redis 支持 AOF 重写来压缩文件。

#### 工作原理

AOF 重写不是读取旧文件，而是直接根据当前内存数据生成新的 AOF 文件。

#### 触发方式

**自动触发**：

```conf
# 当前 AOF 文件大小超过上次重写后的 100%
auto-aof-rewrite-percentage 100

# 最小重写文件大小
auto-aof-rewrite-min-size 64mb
```

**手动触发**：

```bash
BGREWRITEAOF
```

### AOF 配置参数

```conf
# 开启 AOF
appendonly yes

# AOF 文件名
appendfilename "appendonly.aof"

# 同步策略
appendfsync everysec

# AOF 重写期间是否同步
no-appendfsync-on-rewrite no

# 自动重写配置
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# AOF 加载时是否忽略错误
aof-load-truncated yes

# 使用 RDB-AOF 混合持久化
aof-use-rdb-preamble yes
```

### 优点

- **数据安全性高** - 最多丢失 1 秒数据（everysec 模式）
- **可读性好** - AOF 文件是文本格式，可读可修改
- **自动重写** - 避免文件无限增长

### 缺点

- **文件大** - AOF 文件通常比 RDB 文件大
- **恢复慢** - 恢复大数据集比 RDB 慢
- **性能开销** - 写操作性能比 RDB 差

## 混合持久化

Redis 4.0+ 引入混合持久化，结合 RDB 和 AOF 的优点。

### 工作原理

AOF 重写时，将重写时刻的内存数据以 RDB 格式写入 AOF 文件开头，后续的增量数据以 AOF 格式追加。

### 开启混合持久化

```conf
# 开启混合持久化
aof-use-rdb-preamble yes
```

### 优点

- **恢复快** - 使用 RDB 格式存储主要数据
- **数据安全** - 使用 AOF 格式存储增量数据
- **文件小** - RDB 部分比纯 AOF 小

## 持久化策略选择

### 不同场景的推荐

| 场景                 | 推荐策略       | 原因                         |
| -------------------- | -------------- | ---------------------------- |
| 缓存服务（可丢失）   | 关闭持久化     | 性能优先，数据可从数据库恢复 |
| 缓存服务（不可丢失） | RDB            | 定期备份，性能好             |
| 消息队列             | AOF (everysec) | 数据重要，不能丢失太多       |
| 会话存储             | AOF (everysec) | 丢失会话影响用户体验         |
| 分布式锁             | AOF (everysec) | 保证锁的可靠性               |
| 高并发写入           | RDB            | 减少 AOF 写入开销            |
| 数据安全性要求高     | 混合持久化     | 兼顾性能和数据安全           |

### 推荐配置

#### 1. 高性能场景（缓存）

```conf
# 关闭持久化或仅使用 RDB
appendonly no
save 900 1
save 300 10
save 60 10000
```

#### 2. 高可靠性场景（数据不能丢失）

```conf
# 使用混合持久化
appendonly yes
aof-use-rdb-preamble yes
appendfsync everysec

# RDB 作为备份
save 900 1
save 300 10
save 60 10000
```

#### 3. 平衡场景（推荐）

```conf
# AOF + RDB
appendonly yes
appendfsync everysec
aof-use-rdb-preamble yes

save 900 1
save 300 10
save 60 10000
```

## 数据恢复

### 恢复优先级

Redis 启动时按以下优先级加载数据：

1. 如果开启了 AOF，优先加载 AOF 文件
2. 如果没有 AOF，加载 RDB 文件

### 恢复步骤

1. **备份当前文件**（如果有）：

```bash
cp /var/lib/redis/appendonly.aof /backup/
cp /var/lib/redis/dump.rdb /backup/
```

2. **放置恢复文件**：

```bash
# 将备份文件复制到 Redis 数据目录
cp /backup/appendonly.aof /var/lib/redis/
cp /backup/dump.rdb /var/lib/redis/
```

3. **重启 Redis**：

```bash
sudo systemctl restart redis
```

### AOF 文件损坏修复

如果 AOF 文件损坏，可以使用 `redis-check-aof` 工具修复：

```bash
# 检查并修复 AOF 文件
redis-check-aof --fix appendonly.aof
```

### RDB 文件损坏检查

使用 `redis-check-rdb` 检查 RDB 文件：

```bash
redis-check-rdb dump.rdb
```

## 最佳实践

### 1. 同时使用 RDB 和 AOF

```conf
# 使用混合持久化
appendonly yes
aof-use-rdb-preamble yes
appendfsync everysec

# RDB 作为定期备份
save 900 1
save 300 10
save 60 10000
```

### 2. 定期备份

设置定时任务备份 RDB 文件：

```bash
#!/bin/bash
# backup-redis.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/redis"

# 创建备份目录
mkdir -p $BACKUP_DIR

# 触发 RDB 保存
redis-cli BGSAVE

# 等待保存完成
sleep 10

# 复制文件
cp /var/lib/redis/dump.rdb $BACKUP_DIR/dump_$DATE.rdb

# 删除 7 天前的备份
find $BACKUP_DIR -name "dump_*.rdb" -mtime +7 -delete
```

### 3. 监控持久化状态

```bash
# 查看 RDB 状态
INFO persistence

# 关键指标
# rdb_last_save_time: 最后一次保存时间
# rdb_changes_since_last_save: 上次保存后的修改数
# aof_enabled: AOF 是否开启
# aof_last_rewrite_time_sec: 最后一次重写耗时
```

### 4. 优化配置

```conf
# 避免 fork 阻塞
save 900 1
save 300 10
save 60 10000

# AOF 重写时不同步（提高性能）
no-appendfsync-on-rewrite yes

# 合理设置重写阈值
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
```

### 5. 主从架构下的持久化

- **主节点** - 开启 AOF，保证数据不丢失
- **从节点** - 可以关闭持久化，减少开销（主节点已有持久化）

```conf
# 主节点
appendonly yes
appendfsync everysec

# 从节点（可选）
appendonly no
```

## 常见问题

### 1. RDB 保存时 Redis 会阻塞吗？

`BGSAVE` 使用 fork 子进程，不会阻塞主进程。但 fork 本身可能有短暂阻塞。

### 2. AOF 文件越来越大怎么办？

配置自动 AOF 重写：

```conf
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
```

### 3. RDB 和 AOF 能同时使用吗？

可以。Redis 会优先使用 AOF 恢复数据，RDB 作为备份。

### 4. 持久化对性能的影响有多大？

- RDB：定期保存，影响小
- AOF (everysec)：约 5-10% 性能损耗
- AOF (always)：性能损耗较大，不推荐

## 小结

Redis 持久化机制对比：

| 特性       | RDB            | AOF               | 混合持久化 |
| ---------- | -------------- | ----------------- | ---------- |
| 数据完整性 | 低（可能丢失） | 高（最多丢 1 秒） | 高         |
| 文件大小   | 小             | 大                | 中         |
| 恢复速度   | 快             | 慢                | 快         |
| 性能开销   | 低             | 中                | 中         |
| 推荐场景   | 缓存、定期备份 | 重要数据          | 生产环境   |

**推荐配置**：在生产环境中使用混合持久化（AOF + RDB），兼顾性能和数据安全性。
