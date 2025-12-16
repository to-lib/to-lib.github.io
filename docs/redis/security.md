---
sidebar_position: 12
title: 安全配置
---

# Redis 安全配置

Redis 安全性至关重要，本文介绍如何保护 Redis 免受未授权访问和攻击。

## 默认安全风险

Redis 默认配置存在安全风险：

- **无密码认证** - 任何人都可以连接
- **绑定所有接口** - 可从外网访问
- **危险命令可用** - FLUSHALL、CONFIG 等
- **无 TLS 加密** - 数据明文传输

**生产环境必须进行安全配置！**

## 认证配置

### 密码认证

设置访问密码：

```conf
# redis.conf
requirepass your_strong_password_here
```

客户端连接时认证：

```bash
# 命令行
redis-cli -a your_password

# 或连接后认证
redis-cli
127.0.0.1:6379> AUTH your_password
OK
```

Java 连接：

```java
Jedis jedis = new Jedis("localhost", 6379);
jedis.auth("your_password");

// 或使用 JedisPool
JedisPool pool = new JedisPool(
    new JedisPoolConfig(),
    "localhost",
    6379,
    2000,           // 超时
    "your_password" // 密码
);
```

### 密码强度建议

- 至少 32 个字符
- 包含大小写字母、数字、特殊符号
- 定期更换密码

```bash
# 生成强密码
openssl rand -base64 32
```

## ACL 访问控制（Redis 6.0+）

ACL（Access Control List）提供细粒度的权限控制。

### 创建用户

```bash
# 创建用户并授权
ACL SETUSER myuser on >mypassword ~cache:* +get +set

# 参数说明：
# on - 启用用户
# >mypassword - 设置密码
# ~cache:* - 允许访问 cache: 开头的键
# +get +set - 允许 GET 和 SET 命令
```

### 查看用户

```bash
# 查看所有用户
ACL LIST

# 查看当前用户
ACL WHOAMI

# 查看用户详情
ACL GETUSER myuser
```

### 删除用户

```bash
ACL DELUSER myuser
```

### 常用权限配置

```bash
# 只读用户
ACL SETUSER reader on >password ~* +@read

# 读写用户（排除危险命令）
ACL SETUSER writer on >password ~* +@all -@dangerous

# 管理员用户
ACL SETUSER admin on >password ~* +@all

# 只能操作特定键的用户
ACL SETUSER app on >password ~user:* ~session:* +@all
```

### ACL 规则语法

| 规则         | 说明           |
| ------------ | -------------- |
| `on` / `off` | 启用/禁用用户  |
| `>password`  | 添加密码       |
| `<password`  | 删除密码       |
| `~pattern`   | 允许的键模式   |
| `+command`   | 允许的命令     |
| `-command`   | 禁止的命令     |
| `+@category` | 允许的命令类别 |
| `-@category` | 禁止的命令类别 |

### 命令类别

```bash
# 查看所有类别
ACL CAT

# 常用类别
# @read - 读命令
# @write - 写命令
# @dangerous - 危险命令
# @admin - 管理命令
# @fast - 快速命令
# @slow - 慢命令
```

### 持久化 ACL

```conf
# redis.conf
# 使用 ACL 文件
aclfile /etc/redis/users.acl
```

ACL 文件格式：

```
user default on nopass ~* +@all
user myuser on >mypassword ~cache:* +get +set
user reader on >password ~* +@read
```

加载 ACL 文件：

```bash
ACL LOAD
```

## 网络安全

### 绑定 IP

只允许指定 IP 访问：

```conf
# redis.conf
# 只允许本地访问
bind 127.0.0.1

# 允许特定内网 IP
bind 127.0.0.1 192.168.1.100

# 允许所有（不推荐）
bind 0.0.0.0
```

### 保护模式

```conf
# redis.conf
# 开启保护模式（推荐）
protected-mode yes

# 保护模式下，如果没有设置密码且绑定了 0.0.0.0，
# Redis 会拒绝外部连接
```

### 修改端口

```conf
# redis.conf
port 16379  # 使用非默认端口
```

### 防火墙配置

**Linux iptables**：

```bash
# 只允许特定 IP 访问 Redis
iptables -A INPUT -p tcp --dport 6379 -s 192.168.1.0/24 -j ACCEPT
iptables -A INPUT -p tcp --dport 6379 -j DROP
```

**Linux firewalld**：

```bash
# 允许特定 IP
firewall-cmd --permanent --add-rich-rule='rule family=ipv4 source address=192.168.1.0/24 port port=6379 protocol=tcp accept'
firewall-cmd --reload
```

**nftables**：

```bash
nft add rule inet filter input tcp dport 6379 ip saddr 192.168.1.0/24 accept
nft add rule inet filter input tcp dport 6379 drop
```

## 命令安全

### 禁用危险命令

```conf
# redis.conf
# 禁用命令（设为空字符串）
rename-command FLUSHALL ""
rename-command FLUSHDB ""
rename-command CONFIG ""
rename-command SHUTDOWN ""
rename-command KEYS ""
rename-command DEBUG ""
rename-command SLAVEOF ""
rename-command REPLICAOF ""
```

### 重命名命令

```conf
# redis.conf
# 重命名为复杂名称
rename-command CONFIG "CONFIG_a8f3b2c1d4e5"
rename-command FLUSHALL "FLUSHALL_x9y8z7w6"
```

### 使用 ACL 限制命令

```bash
# 禁止用户使用危险命令
ACL SETUSER myuser on >password ~* +@all -@dangerous -config -flushall -flushdb
```

## TLS/SSL 加密

### 生成证书

```bash
# 生成 CA 证书
openssl genrsa -out ca.key 4096
openssl req -x509 -new -nodes -key ca.key -sha256 -days 3650 -out ca.crt

# 生成服务器证书
openssl genrsa -out redis.key 2048
openssl req -new -key redis.key -out redis.csr
openssl x509 -req -in redis.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out redis.crt -days 365 -sha256

# 生成客户端证书
openssl genrsa -out client.key 2048
openssl req -new -key client.key -out client.csr
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client.crt -days 365 -sha256
```

### 配置 Redis TLS

```conf
# redis.conf
# TLS 端口
tls-port 6380

# 证书配置
tls-cert-file /etc/redis/redis.crt
tls-key-file /etc/redis/redis.key
tls-ca-cert-file /etc/redis/ca.crt

# 要求客户端证书认证
tls-auth-clients yes

# 禁用非 TLS 端口（可选）
port 0
```

### 客户端连接 TLS

命令行：

```bash
redis-cli --tls \
    --cert /etc/redis/client.crt \
    --key /etc/redis/client.key \
    --cacert /etc/redis/ca.crt \
    -p 6380
```

Java（Jedis）：

```java
// 配置 SSL
SSLSocketFactory sslSocketFactory = ...; // 使用你的证书

JedisPool pool = new JedisPool(
    new JedisPoolConfig(),
    "localhost",
    6380,
    2000,
    "password",
    0,
    null,
    sslSocketFactory,
    null,
    null
);
```

### 主从复制 TLS

```conf
# 主节点
tls-replication yes

# 从节点
replicaof master_host 6380
tls-replication yes
tls-cert-file /etc/redis/redis.crt
tls-key-file /etc/redis/redis.key
tls-ca-cert-file /etc/redis/ca.crt
```

## 审计和监控

### 慢查询日志

```conf
# redis.conf
slowlog-log-slower-than 10000  # 10ms
slowlog-max-len 128
```

查看慢查询：

```bash
SLOWLOG GET 10
```

### 监控连接

```bash
# 查看客户端列表
CLIENT LIST

# 查看客户端连接数
INFO clients

# 踢出客户端
CLIENT KILL ADDR 192.168.1.100:12345
```

### 安全日志

```conf
# redis.conf
loglevel notice
logfile /var/log/redis/redis.log
```

定期检查日志：

```bash
tail -f /var/log/redis/redis.log | grep -E "(AUTH|ACL|CONNECT)"
```

## 安全检查清单

### 基础安全

- [ ] 设置强密码（requirepass）
- [ ] 绑定内网 IP（bind）
- [ ] 开启保护模式（protected-mode yes）
- [ ] 修改默认端口

### 高级安全

- [ ] 配置 ACL 权限控制
- [ ] 禁用/重命名危险命令
- [ ] 配置 TLS 加密
- [ ] 配置防火墙规则

### 运维安全

- [ ] 定期更换密码
- [ ] 监控异常连接
- [ ] 检查慢查询日志
- [ ] 定期备份数据
- [ ] 及时更新 Redis 版本

## 安全事件应对

### 发现被入侵

1. **立即断开网络**：

```bash
# 关闭外网访问
iptables -A INPUT -p tcp --dport 6379 -j DROP
```

2. **修改密码**：

```bash
CONFIG SET requirepass new_strong_password
```

3. **检查数据**：

```bash
# 查看可疑键
KEYS *
SCAN 0 COUNT 1000
```

4. **检查配置**：

```bash
# 是否被修改
CONFIG GET *
INFO
```

5. **恢复数据**：

```bash
# 从备份恢复
cp /backup/dump.rdb /var/lib/redis/
systemctl restart redis
```

### 常见攻击

**1. 未授权访问**：

攻击者可能：

- 读取/修改数据
- 写入 SSH 公钥
- 执行任意命令

**防护**：设置密码，绑定内网 IP

**2. 主从复制攻击**：

攻击者伪装成主节点进行数据注入。

**防护**：

```conf
# 禁用 SLAVEOF 命令
rename-command SLAVEOF ""
rename-command REPLICAOF ""
```

## 最佳实践

### 1. 最小权限原则

```bash
# 应用用户只能操作特定键
ACL SETUSER app on >password ~app:* +@all

# 监控用户只读
ACL SETUSER monitor on >password ~* +@read
```

### 2. 网络隔离

```
Internet  →  防火墙  →  应用服务器  →  Redis（内网）
                              ↑
                        Redis 只在内网可访问
```

### 3. 定期安全审计

```bash
# 每周检查
# 1. 异常连接
CLIENT LIST

# 2. 慢查询
SLOWLOG GET 100

# 3. 内存使用
INFO memory

# 4. ACL 配置
ACL LIST
```

### 4. 密钥管理

- 使用密钥管理服务存储密码
- 不要在代码中硬编码密码
- 使用环境变量或配置中心

```java
// 从环境变量获取密码
String password = System.getenv("REDIS_PASSWORD");
```

## 小结

Redis 安全配置要点：

| 安全措施     | 重要性 | 配置项            |
| ------------ | ------ | ----------------- |
| 密码认证     | ⭐⭐⭐ | requirepass       |
| 绑定 IP      | ⭐⭐⭐ | bind              |
| 保护模式     | ⭐⭐⭐ | protected-mode    |
| ACL 权限     | ⭐⭐   | ACL SETUSER       |
| 禁用危险命令 | ⭐⭐   | rename-command    |
| TLS 加密     | ⭐⭐   | tls-port          |
| 防火墙       | ⭐⭐   | iptables/nftables |

**生产环境必须配置安全措施，避免数据泄露和服务中断！**
