---
sidebar_position: 24
title: 配置与部署
---

# Redis 配置与部署

本章全面介绍 Redis 在生产环境中的配置项、部署方式和最佳实践，帮助你构建稳定可靠的 Redis 服务。

## 配置文件位置

不同安装方式的默认配置文件位置：

| 安装方式 | 配置文件路径 |
|----------|-------------|
| 源码编译 | `/usr/local/redis/redis.conf` |
| APT/YUM | `/etc/redis/redis.conf` |
| Homebrew (macOS) | `/usr/local/etc/redis.conf` 或 `/opt/homebrew/etc/redis.conf` |
| Docker | 需要挂载配置文件 |

## 启动与配置加载

### 启动方式

```bash
# 指定配置文件启动
redis-server /etc/redis/redis.conf

# 命令行覆盖配置
redis-server /etc/redis/redis.conf --port 6380 --daemonize yes

# 无配置文件启动（使用默认配置）
redis-server
```

### 动态修改配置

```bash
# 查看配置
redis-cli CONFIG GET maxmemory
redis-cli CONFIG GET "*"

# 动态修改（立即生效，不持久化）
redis-cli CONFIG SET maxmemory 2gb

# 将当前配置写入配置文件
redis-cli CONFIG REWRITE
```

## 网络配置

### 绑定地址与端口

```conf
# 绑定 IP 地址
# 只允许本地访问
bind 127.0.0.1

# 允许本地和内网访问
bind 127.0.0.1 192.168.1.100

# 允许所有 IP（不推荐，需配合密码和防火墙）
bind 0.0.0.0

# 端口（默认 6379）
port 6379

# 禁用 TCP（仅 Unix Socket）
port 0
unixsocket /var/run/redis/redis.sock
unixsocketperm 700
```

### 保护模式

```conf
# 保护模式（推荐开启）
# 当 bind 0.0.0.0 且无密码时，拒绝外部连接
protected-mode yes
```

### 连接管理

```conf
# 最大客户端连接数（默认 10000）
maxclients 10000

# 客户端空闲超时（秒，0 表示不超时）
timeout 300

# TCP keepalive（秒，推荐 300）
tcp-keepalive 300

# TCP 连接积压队列
tcp-backlog 511
```

### TLS/SSL 配置

```conf
# TLS 端口
tls-port 6380

# 证书配置
tls-cert-file /etc/redis/redis.crt
tls-key-file /etc/redis/redis.key
tls-ca-cert-file /etc/redis/ca.crt

# 客户端证书验证
tls-auth-clients optional

# 禁用非 TLS 端口
port 0

# 主从复制使用 TLS
tls-replication yes

# 集群使用 TLS
tls-cluster yes
```

## 认证与权限

### 密码认证

```conf
# 设置密码（推荐 32 位以上强密码）
requirepass your_strong_password_here

# 主从复制认证
masterauth your_master_password
```

### ACL 权限控制 (Redis 6.0+)

```conf
# 使用 ACL 文件
aclfile /etc/redis/users.acl

# 或直接在配置文件定义用户
user default on nopass ~* +@all
user readonly on >password ~* +@read -@write
user admin on >adminpass ~* +@all
```

**ACL 文件示例** (`/etc/redis/users.acl`)：

```
# 默认用户（关闭或设置密码）
user default off

# 应用用户（限制键前缀和命令）
user app on >app_password ~app:* +@all -@dangerous

# 只读用户
user reader on >reader_password ~* +@read

# 管理员用户
user admin on >admin_password ~* +@all
```

```bash
# ACL 管理命令
ACL LIST                       # 查看所有用户
ACL WHOAMI                     # 查看当前用户
ACL SETUSER username ...       # 创建/修改用户
ACL DELUSER username           # 删除用户
ACL LOAD                       # 重新加载 ACL 文件
ACL SAVE                       # 保存 ACL 到文件
```

## 内存配置

### 内存限制

```conf
# 最大内存（支持 kb, mb, gb）
maxmemory 2gb

# 内存淘汰策略
maxmemory-policy allkeys-lru

# LRU/LFU 采样数量（精度，默认 5）
maxmemory-samples 5

# LFU 配置
lfu-log-factor 10
lfu-decay-time 1
```

### 内存淘汰策略详解

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `noeviction` | 不淘汰，写满报错 | 数据不能丢失 |
| `allkeys-lru` | 所有键 LRU 淘汰 | 缓存（推荐） |
| `allkeys-lfu` | 所有键 LFU 淘汰 | 热点数据场景 |
| `allkeys-random` | 随机淘汰 | 均匀访问场景 |
| `volatile-lru` | 有过期时间的键 LRU | 混合存储 |
| `volatile-lfu` | 有过期时间的键 LFU | 混合存储 |
| `volatile-ttl` | 优先淘汰即将过期的 | 利用 TTL 控制 |
| `volatile-random` | 有过期时间的随机淘汰 | 临时数据 |

### 惰性删除 (Lazy Freeing)

```conf
# 淘汰时异步删除
lazyfree-lazy-eviction yes

# 过期键异步删除
lazyfree-lazy-expire yes

# DEL 命令异步删除
lazyfree-lazy-server-del yes

# 从节点全量同步前清空数据异步
replica-lazy-flush yes

# UNLINK 替代 DEL
lazyfree-lazy-user-del yes
```

### 内存碎片整理

```conf
# 开启活跃碎片整理
activedefrag yes

# 碎片率阈值
active-defrag-ignore-bytes 100mb
active-defrag-threshold-lower 10
active-defrag-threshold-upper 100

# CPU 使用限制
active-defrag-cycle-min 1
active-defrag-cycle-max 25
```

## 持久化配置

### RDB 配置

```conf
# 自动保存规则
save 900 1      # 900秒(15分钟)内至少1个键改变
save 300 10     # 300秒(5分钟)内至少10个键改变
save 60 10000   # 60秒内至少10000个键改变

# 禁用 RDB
save ""

# RDB 文件名
dbfilename dump.rdb

# 数据目录
dir /var/lib/redis

# 保存失败时停止写入
stop-writes-on-bgsave-error yes

# 压缩 RDB 文件
rdbcompression yes

# RDB 校验和
rdbchecksum yes
```

### AOF 配置

```conf
# 开启 AOF
appendonly yes

# AOF 文件名
appendfilename "appendonly.aof"

# 同步策略
# always    - 每个写命令都同步（最安全，最慢）
# everysec  - 每秒同步（推荐，性能和安全平衡）
# no        - 由操作系统决定（最快，最不安全）
appendfsync everysec

# AOF 重写期间不执行 fsync（提高性能）
no-appendfsync-on-rewrite yes

# 自动重写触发
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# AOF 加载时忽略错误
aof-load-truncated yes

# 混合持久化（RDB + AOF，Redis 4.0+）
aof-use-rdb-preamble yes
```

### Redis 7.0+ 多部分 AOF

```conf
# AOF 目录（Redis 7.0+）
appenddirname "appendonlydir"

# AOF 时间戳（用于 PITR）
aof-timestamp-enabled no
```

## 复制配置

### 从节点配置

```conf
# 指定主节点
replicaof 192.168.1.100 6379

# 主节点密码
masterauth your_master_password

# 从节点只读
replica-read-only yes

# 主节点下线时是否提供服务
replica-serve-stale-data yes

# 复制优先级（越小优先级越高，用于哨兵选举）
replica-priority 100
```

### 复制缓冲区

```conf
# 复制积压缓冲区大小
repl-backlog-size 1mb

# 缓冲区释放时间（主节点无从节点后）
repl-backlog-ttl 3600

# 复制超时
repl-timeout 60

# 禁用 TCP_NODELAY
repl-disable-tcp-nodelay no
```

### 无盘复制

```conf
# 无盘复制（直接发送 RDB 流，不写磁盘）
repl-diskless-sync yes

# 等待更多从节点的延迟（秒）
repl-diskless-sync-delay 5

# 无盘加载（从节点直接加载 RDB 流）
repl-diskless-load disabled
```

### 写入要求

```conf
# 要求至少 N 个从节点才允许写入
min-replicas-to-write 1

# 从节点延迟不超过 N 秒
min-replicas-max-lag 10
```

## 慢查询配置

```conf
# 慢查询阈值（微秒，10000 = 10ms）
slowlog-log-slower-than 10000

# 慢查询日志最大条数
slowlog-max-len 128
```

```bash
# 查看慢查询
redis-cli SLOWLOG GET 10
redis-cli SLOWLOG LEN
redis-cli SLOWLOG RESET
```

## 日志配置

```conf
# 日志级别：debug, verbose, notice, warning
loglevel notice

# 日志文件（空字符串输出到 stdout）
logfile /var/log/redis/redis.log

# 是否输出到 syslog
syslog-enabled no
syslog-ident redis
syslog-facility local0
```

## 线程配置 (Redis 6.0+)

```conf
# I/O 线程数（建议 CPU 核数的一半，最大 128）
io-threads 4

# I/O 线程处理读操作
io-threads-do-reads yes
```

> **注意**：I/O 多线程主要用于网络 I/O，命令执行仍然是单线程。

## 集群配置

```conf
# 开启集群模式
cluster-enabled yes

# 集群配置文件（自动生成）
cluster-config-file nodes.conf

# 节点超时时间（毫秒）
cluster-node-timeout 15000

# 故障转移要求的从节点数据新鲜度
cluster-replica-validity-factor 10

# 需要完整槽位覆盖才提供服务
cluster-require-full-coverage yes

# 从节点迁移阈值
cluster-migration-barrier 1

# 主节点下线时从节点是否提供读服务
cluster-allow-reads-when-down no

# 槽位迁移时是否允许节点提供服务（Redis 7.0+）
cluster-allow-pubsubshard-when-down yes
```

## 哨兵配置

哨兵使用单独的配置文件 `sentinel.conf`：

```conf
# 哨兵端口
port 26379

# 监控主节点
sentinel monitor mymaster 192.168.1.100 6379 2

# 主节点密码
sentinel auth-pass mymaster your_password

# 主节点下线判定时间（毫秒）
sentinel down-after-milliseconds mymaster 30000

# 故障转移超时（毫秒）
sentinel failover-timeout mymaster 180000

# 并行复制的从节点数
sentinel parallel-syncs mymaster 1
```

## 容器化部署

### Docker 部署

```bash
# 基础运行
docker run -d --name redis \
    -p 6379:6379 \
    redis:7.0

# 挂载配置文件和数据
docker run -d --name redis \
    -p 6379:6379 \
    -v /opt/redis/redis.conf:/usr/local/etc/redis/redis.conf \
    -v /opt/redis/data:/data \
    redis:7.0 redis-server /usr/local/etc/redis/redis.conf
```

**Docker Compose 示例**：

```yaml
version: '3.8'
services:
  redis:
    image: redis:7.0
    container_name: redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - ./redis.conf:/usr/local/etc/redis/redis.conf
      - redis-data:/data
    command: redis-server /usr/local/etc/redis/redis.conf
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  redis-data:
```

### Kubernetes 部署

**ConfigMap**：

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-config
data:
  redis.conf: |
    bind 0.0.0.0
    port 6379
    maxmemory 1gb
    maxmemory-policy allkeys-lru
    appendonly yes
    appendfsync everysec
```

**Deployment**：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
        - name: redis
          image: redis:7.0
          ports:
            - containerPort: 6379
          volumeMounts:
            - name: config
              mountPath: /usr/local/etc/redis
            - name: data
              mountPath: /data
          command: ["redis-server", "/usr/local/etc/redis/redis.conf"]
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "1Gi"
              cpu: "500m"
      volumes:
        - name: config
          configMap:
            name: redis-config
        - name: data
          persistentVolumeClaim:
            claimName: redis-pvc
```

## 生产环境推荐配置

```conf
# === 网络 ===
bind 127.0.0.1 192.168.1.100
port 6379
protected-mode yes
tcp-backlog 511
timeout 300
tcp-keepalive 300

# === 认证 ===
requirepass your_strong_32char_password_here

# === 内存 ===
maxmemory 2gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# === 惰性删除 ===
lazyfree-lazy-eviction yes
lazyfree-lazy-expire yes
lazyfree-lazy-server-del yes

# === 持久化 ===
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec
no-appendfsync-on-rewrite yes
aof-use-rdb-preamble yes
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# === 慢查询 ===
slowlog-log-slower-than 10000
slowlog-max-len 128

# === 日志 ===
loglevel notice
logfile /var/log/redis/redis.log

# === 线程 (Redis 6.0+) ===
io-threads 4
io-threads-do-reads yes

# === 碎片整理 ===
activedefrag yes
```

## 配置故障排查

### 常见配置问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 无法远程连接 | bind 127.0.0.1 | 添加内网 IP 到 bind |
| 连接被拒绝 | protected-mode | 设置密码或关闭保护模式 |
| 内存超限报错 | maxmemory 设置过小 | 增加 maxmemory 或调整淘汰策略 |
| 持久化失败 | 磁盘空间不足 | 清理磁盘或调整持久化策略 |
| 启动失败 | 配置文件语法错误 | 检查配置文件格式 |

### 验证配置

```bash
# 检查配置文件语法
redis-server /etc/redis/redis.conf --test-memory 1024

# 查看运行时配置
redis-cli CONFIG GET "*"

# 查看配置与默认值的差异
redis-cli CONFIG GET "*" | diff - default.conf
```

## 小结

| 配置类别 | 核心配置项 | 推荐值 |
|----------|-----------|--------|
| 网络 | bind, port, protected-mode | 内网 IP, 6379, yes |
| 认证 | requirepass | 32+ 字符强密码 |
| 内存 | maxmemory, maxmemory-policy | 物理内存 70%, allkeys-lru |
| 持久化 | appendonly, appendfsync | yes, everysec |
| 性能 | io-threads | CPU 核数 / 2 |
| 安全 | ACL | 最小权限原则 |

**最佳实践**：

- ✅ 使用配置文件而非命令行参数
- ✅ 使用 `CONFIG REWRITE` 持久化动态修改
- ✅ 定期审查和更新配置
- ✅ 在测试环境验证配置变更
- ✅ 保留配置文件的版本历史
