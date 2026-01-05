---
sidebar_position: 25
title: 监控与排障
---

# Redis 监控与排障

本章详细介绍 Redis 的监控指标、诊断工具和常见故障排查方法，帮助你快速定位和解决生产环境中的问题。

## 核心监控指标

### INFO 命令详解

`INFO` 命令是 Redis 监控的基础，返回服务器的各类统计信息。

```bash
# 获取所有信息
redis-cli INFO

# 获取特定模块
redis-cli INFO server      # 服务器信息
redis-cli INFO clients     # 客户端连接
redis-cli INFO memory      # 内存使用
redis-cli INFO persistence # 持久化状态
redis-cli INFO stats       # 统计数据
redis-cli INFO replication # 复制信息
redis-cli INFO cpu         # CPU 使用
redis-cli INFO commandstats # 命令统计
redis-cli INFO cluster     # 集群信息
redis-cli INFO keyspace    # 键空间统计
```

### 服务器信息 (server)

```bash
redis-cli INFO server

# 关键指标
redis_version:7.0.0          # Redis 版本
redis_mode:standalone        # 运行模式
os:Linux 5.4.0               # 操作系统
uptime_in_seconds:86400      # 运行时间（秒）
uptime_in_days:1             # 运行时间（天）
hz:10                        # serverCron 频率
executable:/usr/bin/redis-server
config_file:/etc/redis/redis.conf
```

### 客户端信息 (clients)

```bash
redis-cli INFO clients

# 关键指标
connected_clients:100        # 当前连接数
blocked_clients:0            # 阻塞连接数（执行 BLPOP 等）
tracking_clients:0           # 客户端追踪数
clients_in_timeout_table:0   # 超时表中的客户端
```

**告警阈值**：

| 指标 | 警告 | 严重 |
|------|------|------|
| connected_clients | > 5000 | > 8000 |
| blocked_clients | > 100 | > 500 |

### 内存信息 (memory)

```bash
redis-cli INFO memory

# 关键指标
used_memory:1073741824       # 已使用内存（字节）
used_memory_human:1.00G      # 已使用内存（可读）
used_memory_rss:1181116416   # 操作系统分配内存
used_memory_rss_human:1.10G
used_memory_peak:1200000000  # 内存使用峰值
used_memory_peak_human:1.12G
used_memory_lua:37888        # Lua 引擎内存
maxmemory:2147483648         # 配置的最大内存
maxmemory_policy:allkeys-lru # 淘汰策略
mem_fragmentation_ratio:1.10 # 内存碎片率
mem_allocator:jemalloc-5.2.1 # 内存分配器
```

**内存碎片率分析**：

```
碎片率 = used_memory_rss / used_memory
```

| 碎片率 | 含义 | 操作建议 |
|--------|------|----------|
| < 1 | 使用了 swap | 紧急！增加内存或减少数据 |
| 1.0 - 1.5 | 正常 | 无需处理 |
| 1.5 - 2.0 | 碎片较多 | 观察或开启碎片整理 |
| > 2.0 | 碎片严重 | 重启 Redis 或开启整理 |

### 持久化信息 (persistence)

```bash
redis-cli INFO persistence

# RDB 相关
rdb_changes_since_last_save:100  # 上次保存后的修改数
rdb_bgsave_in_progress:0         # 是否正在保存
rdb_last_save_time:1704067200    # 最后保存时间戳
rdb_last_bgsave_status:ok        # 最后保存状态
rdb_last_bgsave_time_sec:2       # 最后保存耗时

# AOF 相关
aof_enabled:1                    # AOF 是否开启
aof_rewrite_in_progress:0        # 是否正在重写
aof_last_rewrite_time_sec:5      # 最后重写耗时
aof_current_size:1073741824      # 当前 AOF 大小
aof_base_size:536870912          # 重写后基础大小
```

**告警条件**：

- `rdb_last_bgsave_status` 不是 `ok`
- `rdb_changes_since_last_save` 持续增长且无保存
- `aof_last_write_status` 不是 `ok`

### 统计信息 (stats)

```bash
redis-cli INFO stats

# 关键指标
total_connections_received:10000  # 累计连接数
total_commands_processed:1000000  # 累计命令数
instantaneous_ops_per_sec:15000   # 当前 OPS
rejected_connections:0            # 拒绝的连接数
expired_keys:5000                 # 过期删除的键数
evicted_keys:100                  # 淘汰的键数
keyspace_hits:900000              # 命中次数
keyspace_misses:100000            # 未命中次数
```

**缓存命中率计算**：

```
命中率 = keyspace_hits / (keyspace_hits + keyspace_misses)
```

| 命中率 | 评价 | 操作建议 |
|--------|------|----------|
| > 95% | 优秀 | 无需处理 |
| 90% - 95% | 正常 | 观察 |
| 80% - 90% | 偏低 | 检查缓存策略 |
| < 80% | 较差 | 优化缓存设计 |

### 复制信息 (replication)

```bash
redis-cli INFO replication

# 主节点
role:master
connected_slaves:2
slave0:ip=192.168.1.101,port=6379,state=online,offset=123456,lag=0
slave1:ip=192.168.1.102,port=6379,state=online,offset=123456,lag=1

# 从节点
role:slave
master_host:192.168.1.100
master_port:6379
master_link_status:up
master_last_io_seconds_ago:1
master_sync_in_progress:0
slave_repl_offset:123456
```

**告警条件**：

- `master_link_status` 不是 `up`
- `master_last_io_seconds_ago` > 10
- 主从 offset 差距过大

## 慢查询分析

### 配置慢查询

```conf
# 慢查询阈值（微秒，10000 = 10ms）
slowlog-log-slower-than 10000

# 慢查询日志最大条数
slowlog-max-len 128
```

### 查看慢查询

```bash
# 获取最近 10 条慢查询
redis-cli SLOWLOG GET 10

# 慢查询日志条数
redis-cli SLOWLOG LEN

# 清空慢查询日志
redis-cli SLOWLOG RESET
```

### 慢查询日志解读

```bash
1) 1) (integer) 14               # 日志 ID
   2) (integer) 1704067200       # 时间戳（Unix）
   3) (integer) 15000            # 执行时间（微秒）
   4) 1) "KEYS"                  # 命令
      2) "*"
   5) "127.0.0.1:12345"          # 客户端地址
   6) ""                         # 客户端名称
```

### 常见慢命令

| 命令 | 时间复杂度 | 原因 | 替代方案 |
|------|-----------|------|----------|
| `KEYS *` | O(N) | 遍历所有键 | SCAN |
| `HGETALL` | O(N) | 获取大 Hash 所有字段 | HSCAN/部分获取 |
| `SMEMBERS` | O(N) | 获取大 Set 所有成员 | SSCAN |
| `LRANGE 0 -1` | O(N) | 获取大 List 所有元素 | 分页获取 |
| `DEL bigkey` | O(N) | 删除大键 | UNLINK |
| `FLUSHDB` | O(N) | 清空数据库 | 避免使用 |

## 大键分析

### 发现大键

```bash
# 使用 --bigkeys 扫描
redis-cli --bigkeys

# 输出示例
# -------- summary -------
# Biggest string found 'user:1001' has 50000 bytes
# Biggest list found 'orders' has 100000 items
# Biggest hash found 'product:info' has 50000 fields
```

### 分析大键内存

```bash
# 查看键的内存占用
redis-cli MEMORY USAGE key

# 查看内存使用详情
redis-cli MEMORY DOCTOR
```

### 大键处理脚本

```bash
#!/bin/bash
# find_big_keys.sh - 扫描大键

THRESHOLD=10240  # 10KB
OUTPUT_FILE="big_keys.txt"

echo "开始扫描大键（阈值: $THRESHOLD 字节）..."
echo "" > $OUTPUT_FILE

cursor=0
count=0

while true; do
    result=$(redis-cli SCAN $cursor COUNT 100)
    cursor=$(echo "$result" | head -1)
    keys=$(echo "$result" | tail -n +2)
    
    for key in $keys; do
        size=$(redis-cli MEMORY USAGE "$key" 2>/dev/null)
        if [ -n "$size" ] && [ "$size" -gt "$THRESHOLD" ]; then
            type=$(redis-cli TYPE "$key")
            echo "$key | $type | $size 字节" >> $OUTPUT_FILE
            ((count++))
        fi
    done
    
    if [ "$cursor" = "0" ]; then
        break
    fi
done

echo "扫描完成，发现 $count 个大键"
cat $OUTPUT_FILE
```

## Prometheus 监控

### 安装 redis_exporter

```bash
# 下载
wget https://github.com/oliver006/redis_exporter/releases/download/v1.55.0/redis_exporter-v1.55.0.linux-amd64.tar.gz
tar -xzf redis_exporter-v1.55.0.linux-amd64.tar.gz

# 运行
./redis_exporter --redis.addr=redis://localhost:6379 \
                 --redis.password=your_password \
                 --web.listen-address=:9121
```

### Systemd 服务

```ini
# /etc/systemd/system/redis_exporter.service
[Unit]
Description=Redis Exporter
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/redis_exporter \
    --redis.addr=redis://localhost:6379 \
    --redis.password=${REDIS_PASSWORD}
Restart=always

[Install]
WantedBy=multi-user.target
```

### Prometheus 配置

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']
```

### 常用 PromQL 查询

```promql
# 内存使用率
redis_memory_used_bytes / redis_memory_max_bytes * 100

# OPS（每秒操作数）
rate(redis_commands_processed_total[1m])

# 命中率
rate(redis_keyspace_hits_total[5m]) / 
(rate(redis_keyspace_hits_total[5m]) + rate(redis_keyspace_misses_total[5m])) * 100

# 连接数
redis_connected_clients

# 内存碎片率
redis_memory_used_rss_bytes / redis_memory_used_bytes

# 淘汰键速率
rate(redis_evicted_keys_total[5m])
```

### Grafana 仪表盘

推荐使用 [Redis Dashboard](https://grafana.com/grafana/dashboards/11835) (ID: 11835)

```bash
# 导入仪表盘
# 1. Grafana -> Dashboards -> Import
# 2. 输入 Dashboard ID: 11835
# 3. 选择 Prometheus 数据源
```

## 告警规则

### Prometheus AlertManager 规则

```yaml
# redis_alerts.yml
groups:
  - name: redis
    rules:
      # 内存使用率过高
      - alert: RedisMemoryHigh
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Redis 内存使用率超过 90%"
          description: "实例 {{ $labels.instance }} 内存使用率 {{ $value | humanizePercentage }}"
      
      # 连接数过高
      - alert: RedisConnectionsHigh
        expr: redis_connected_clients > 8000
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Redis 连接数过高"
      
      # 命中率过低
      - alert: RedisHitRateLow
        expr: |
          rate(redis_keyspace_hits_total[5m]) / 
          (rate(redis_keyspace_hits_total[5m]) + rate(redis_keyspace_misses_total[5m])) < 0.8
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Redis 缓存命中率低于 80%"
      
      # 主从延迟
      - alert: RedisReplicationLag
        expr: redis_connected_slaves < 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis 主从复制断开"
      
      # 持久化失败
      - alert: RedisPersistenceError
        expr: redis_rdb_last_bgsave_status != 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis RDB 持久化失败"
```

## 故障排查指南

### 1. 连接问题

**症状**：无法连接 Redis

**排查步骤**：

```bash
# 1. 检查 Redis 是否运行
ps aux | grep redis
systemctl status redis

# 2. 检查端口监听
netstat -tlnp | grep 6379
ss -tlnp | grep 6379

# 3. 检查绑定地址
redis-cli CONFIG GET bind

# 4. 检查防火墙
iptables -L -n | grep 6379
firewall-cmd --list-all

# 5. 测试连接
redis-cli -h 127.0.0.1 -p 6379 ping
telnet 127.0.0.1 6379
```

**常见原因**：

- Redis 未启动
- bind 配置错误
- protected-mode 开启但无密码
- 防火墙阻止

### 2. 内存问题

**症状**：OOM 错误、写入失败、性能下降

**排查步骤**：

```bash
# 1. 检查内存使用
redis-cli INFO memory

# 2. 检查淘汰情况
redis-cli INFO stats | grep evicted

# 3. 检查大键
redis-cli --bigkeys

# 4. 检查碎片率
redis-cli INFO memory | grep fragmentation

# 5. 检查淘汰策略
redis-cli CONFIG GET maxmemory-policy
```

**解决方案**：

- 增加 maxmemory
- 调整淘汰策略为 allkeys-lru
- 处理大键
- 开启碎片整理

### 3. CPU 问题

**症状**：CPU 使用率高、响应变慢

**排查步骤**：

```bash
# 1. 检查命令统计
redis-cli INFO commandstats

# 2. 检查慢查询
redis-cli SLOWLOG GET 20

# 3. 实时监控（慎用，有性能开销）
redis-cli MONITOR  # Ctrl+C 停止

# 4. 检查客户端列表
redis-cli CLIENT LIST
```

**常见原因**：

- 大量 KEYS 命令
- 大 Key 操作
- Lua 脚本执行过久
- 热点 Key

### 4. 主从复制问题

**症状**：主从延迟、数据不一致

**排查步骤**：

```bash
# 主节点
redis-cli INFO replication

# 关注指标
# connected_slaves: 连接的从节点数
# slave0: offset, lag 等

# 从节点
redis-cli INFO replication

# 关注指标
# master_link_status: 连接状态
# master_last_io_seconds_ago: 最后同步时间
# master_sync_in_progress: 是否正在同步
```

**常见原因**：

- 网络不稳定
- 主节点写入量大
- 复制缓冲区不足
- 大 Key 导致同步慢

### 5. 集群问题

**症状**：MOVED/ASK 错误、槽位不可用

**排查步骤**：

```bash
# 检查集群状态
redis-cli -c CLUSTER INFO

# 检查节点状态
redis-cli -c CLUSTER NODES

# 检查槽位分配
redis-cli -c CLUSTER SLOTS

# 修复集群
redis-cli --cluster check host:port
redis-cli --cluster fix host:port
```

**常见原因**：

- 节点下线
- 槽位迁移中
- 配置不一致

## 诊断命令速查

### 基础诊断

```bash
# 连通性测试
redis-cli PING

# 服务器时间
redis-cli TIME

# 调试信息
redis-cli DEBUG SLEEP 0.1

# 客户端信息
redis-cli CLIENT INFO
redis-cli CLIENT LIST

# 配置检查
redis-cli CONFIG GET "*"
```

### 性能诊断

```bash
# 实时统计
redis-cli --stat

# 延迟测试
redis-cli --latency
redis-cli --latency-history
redis-cli --latency-dist

# 内存分析
redis-cli --memkeys

# 大键扫描
redis-cli --bigkeys
```

### 数据诊断

```bash
# 键空间统计
redis-cli INFO keyspace

# 键数量
redis-cli DBSIZE

# 随机键
redis-cli RANDOMKEY

# 键扫描
redis-cli SCAN 0 COUNT 100

# 键类型和编码
redis-cli TYPE key
redis-cli OBJECT ENCODING key
```

## 运维 SOP

### 日常巡检清单

```bash
#!/bin/bash
# daily_check.sh

echo "=== Redis 日常巡检 $(date) ==="

# 1. 存活检查
echo "1. 存活检查"
redis-cli ping

# 2. 内存使用
echo "2. 内存使用"
redis-cli INFO memory | grep -E "used_memory_human|maxmemory_human|mem_fragmentation"

# 3. 连接数
echo "3. 连接数"
redis-cli INFO clients | grep connected_clients

# 4. 命中率
echo "4. 命中率"
redis-cli INFO stats | grep -E "keyspace_hits|keyspace_misses"

# 5. 慢查询
echo "5. 慢查询数量"
redis-cli SLOWLOG LEN

# 6. 持久化状态
echo "6. 持久化状态"
redis-cli INFO persistence | grep -E "rdb_last_bgsave_status|aof_last_write_status"

# 7. 复制状态（如有从节点）
echo "7. 复制状态"
redis-cli INFO replication | grep -E "role|connected_slaves|master_link_status"

echo "=== 巡检完成 ==="
```

### 紧急故障处理

```bash
# 紧急限流 - 降低 maxclients
redis-cli CONFIG SET maxclients 1000

# 紧急内存释放 - 触发淘汰
redis-cli CONFIG SET maxmemory 1gb

# 杀掉阻塞客户端
redis-cli CLIENT KILL ID <client-id>

# 杀掉慢查询
redis-cli CLIENT KILL ADDR <ip>:<port>

# 紧急持久化
redis-cli BGSAVE
```

## 小结

| 监控类别 | 核心指标 | 告警阈值 |
|----------|----------|----------|
| 可用性 | ping 响应 | 无响应 |
| 内存 | used_memory / maxmemory | > 90% |
| 连接 | connected_clients | > 8000 |
| 性能 | instantaneous_ops_per_sec | 异常下降 |
| 命中率 | hits / (hits + misses) | < 80% |
| 碎片率 | mem_fragmentation_ratio | > 1.5 |
| 持久化 | rdb_last_bgsave_status | != ok |
| 复制 | master_link_status | != up |

**最佳实践**：

- ✅ 配置 Prometheus + Grafana 监控
- ✅ 设置合理的告警规则
- ✅ 定期巡检和慢查询分析
- ✅ 建立故障排查 SOP
- ✅ 定期进行故障演练
