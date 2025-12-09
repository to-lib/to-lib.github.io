---
sidebar_position: 6
title: Redis 哨兵模式
---

# Redis 哨兵模式

Redis Sentinel（哨兵）是 Redis 官方提供的高可用解决方案，用于监控主从架构，并在主节点故障时自动进行故障转移。

## 哨兵的作用

### 核心功能

1. **监控（Monitoring）** - 监控主从节点是否正常运行
2. **通知（Notification）** - 发送故障通知
3. **自动故障转移（Automatic Failover）** - 主节点故障时，自动提升从节点为主节点
4. **配置中心（Configuration Provider）** - 客户端通过 Sentinel 获取主节点地址

### 架构

```
        +----------+  +----------+  +----------+
        | Sentinel1|  | Sentinel2|  | Sentinel3|
        +----------+  +----------+  +----------+
             |             |             |
             +-------------+-------------+
                           |
        +------------------+------------------+
        |                  |                  |
    +--------+        +--------+        +--------+
    | Master |------->| Slave1 |        | Slave2 |
    +--------+        +--------+        +--------+
```

## 配置哨兵

### 1. 准备 Redis 主从

首先配置一个主从架构（参考[主从复制](./replication)章节）。

### 2. 配置哨兵

创建 `sentinel.conf` 配置文件：

```conf
# 哨兵端口
port 26379

# 后台运行
daemonize yes

# 日志文件
logfile "/var/log/redis/sentinel.log"

# 工作目录
dir /var/lib/redis

# 监控主节点
# sentinel monitor <master-name> <ip> <port> <quorum>
# quorum: 判定主节点下线需要的哨兵数量
sentinel monitor mymaster 192.168.1.100 6379 2

# 主节点密码
sentinel auth-pass mymaster mypassword

# 判定主节点下线时间（毫秒）
sentinel down-after-milliseconds mymaster 30000

# 故障转移超时时间（毫秒）
sentinel failover-timeout mymaster 180000

# 同时进行复制的从节点数量
sentinel parallel-syncs mymaster 1
```

### 3. 启动哨兵

```bash
# 启动哨兵
redis-sentinel /etc/redis/sentinel.conf

# 或者
redis-server /etc/redis/sentinel.conf --sentinel
```

### 4. 查看哨兵状态

```bash
# 连接哨兵
redis-cli -p 26379

# 查看主节点信息
SENTINEL master mymaster

# 查看从节点信息
SENTINEL slaves mymaster

# 查看其他哨兵信息
SENTINEL sentinels mymaster

# 查看监控的主节点列表
SENTINEL masters
```

## 故障转移流程

### 主节点下线判定

1. **主观下线（SDOWN）** - 单个哨兵认为主节点下线
2. **客观下线（ODOWN）** - 达到 quorum 数量的哨兵认为主节点下线

### 选举 Leader

多个哨兵通过 Raft 算法选举出一个 Leader 哨兵负责故障转移。

### 选择新主节点

Leader 哨兵按以下规则选择新主节点：

1. 过滤掉下线的从节点
2. 过滤掉 5 秒内没有回复哨兵 INFO 命令的从节点
3. 过滤掉与主节点断开连接时间过长的从节点
4. 选择优先级最高的从节点（`replica-priority`）
5. 如果优先级相同，选择复制偏移量最大的（数据最新）
6. 如果偏移量相同，选择 runid 最小的

### 故障转移步骤

1. 在选中的从节点上执行 `REPLICAOF NO ONE`
2. 向其他从节点发送 `REPLICAOF <new-master-ip> <new-master-port>`
3. 更新哨兵配置
4. 等待旧主节点恢复后，将其设置为从节点

## 客户端连接

### Java（Jedis）

```java
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisSentinelPool;

import java.util.HashSet;
import java.util.Set;

public class SentinelExample {
    public static void main(String[] args) {
        // 哨兵地址
        Set<String> sentinels = new HashSet<>();
        sentinels.add("192.168.1.201:26379");
        sentinels.add("192.168.1.202:26379");
        sentinels.add("192.168.1.203:26379");

        // 创建哨兵连接池
        JedisSentinelPool pool = new JedisSentinelPool(
            "mymaster",    // master name
            sentinels,     // sentinel addresses
            "mypassword"   // password
        );

        // 获取连接
        try (Jedis jedis = pool.getResource()) {
            jedis.set("key", "value");
            String value = jedis.get("key");
            System.out.println(value);
        }

        pool.close();
    }
}
```

### Spring Boot

```yaml
spring:
  redis:
    sentinel:
      master: mymaster
      nodes:
        - 192.168.1.201:26379
        - 192.168.1.202:26379
        - 192.168.1.203:26379
    password: mypassword
```

```java
@Configuration
public class RedisConfig {
    @Bean
    public RedisConnectionFactory redisConnectionFactory() {
        RedisSentinelConfiguration config = new RedisSentinelConfiguration()
            .master("mymaster")
            .sentinel("192.168.1.201", 26379)
            .sentinel("192.168.1.202", 26379)
            .sentinel("192.168.1.203", 26379);

        return new LettuceConnectionFactory(config);
    }
}
```

## 哨兵命令

### 监控命令

```bash
# 查看主节点信息
SENTINEL master <master-name>

# 查看所有主节点
SENTINEL masters

# 查看从节点
SENTINEL slaves <master-name>

# 查看其他哨兵
SENTINEL sentinels <master-name>

# 获取主节点地址
SENTINEL get-master-addr-by-name <master-name>
```

### 管理命令

```bash
# 手动故障转移
SENTINEL failover <master-name>

# 重置主节点
SENTINEL reset <pattern>

# 移除监控
SENTINEL remove <master-name>

# 添加监控
SENTINEL monitor <master-name> <ip> <port> <quorum>

# 修改配置
SENTINEL set <master-name> <option> <value>
```

## 配置参数详解

### 监控配置

```conf
# 监控主节点
sentinel monitor mymaster 192.168.1.100 6379 2

# 参数说明：
# mymaster: 主节点名称（可自定义）
# 192.168.1.100: 主节点 IP
# 6379: 主节点端口
# 2: quorum（判定主节点下线需要的哨兵数量）
```

### 认证配置

```conf
# 主节点密码
sentinel auth-pass mymaster mypassword

# 如果从节点密码不同（一般相同）
sentinel auth-user mymaster <username>
```

### 超时配置

```conf
# 主节点下线判定时间（毫秒）
# 哨兵在这个时间内没有收到主节点的有效回复，则判定为主观下线
sentinel down-after-milliseconds mymaster 30000

# 故障转移超时时间（毫秒）
sentinel failover-timeout mymaster 180000
```

### 复制配置

```conf
# 故障转移期间，同时进行复制的从节点数量
# 设置为 1 可以降低主节点压力，但复制时间更长
sentinel parallel-syncs mymaster 1
```

### 通知配置

```conf
# 故障转移时执行的脚本
sentinel notification-script mymaster /path/to/notify.sh

# 故障转移完成后执行的脚本
sentinel client-reconfig-script mymaster /path/to/reconfig.sh
```

## 哨兵集群

### 部署建议

- **奇数个哨兵** - 3 个或 5 个，避免脑裂
- **分布式部署** - 部署在不同的物理机或可用区
- **quorum 设置** - 通常设置为 `n/2 + 1`

### 示例：3 哨兵集群

**Sentinel 1**（192.168.1.201）：

```conf
port 26379
sentinel monitor mymaster 192.168.1.100 6379 2
sentinel down-after-milliseconds mymaster 30000
sentinel parallel-syncs mymaster 1
sentinel failover-timeout mymaster 180000
```

**Sentinel 2**（192.168.1.202）：

```conf
port 26379
sentinel monitor mymaster 192.168.1.100 6379 2
sentinel down-after-milliseconds mymaster 30000
sentinel parallel-syncs mymaster 1
sentinel failover-timeout mymaster 180000
```

**Sentinel 3**（192.168.1.203）：

```conf
port 26379
sentinel monitor mymaster 192.168.1.100 6379 2
sentinel down-after-milliseconds mymaster 30000
sentinel parallel-syncs mymaster 1
sentinel failover-timeout mymaster 180000
```

## 常见问题

### 1. 脑裂问题

网络分区导致出现多个主节点。

**解决方案**：

```conf
# 要求至少 1 个从节点在线
min-replicas-to-write 1

# 要求从节点延迟不超过 10 秒
min-replicas-max-lag 10
```

### 2. 哨兵无法连接主节点

**原因**：

- 防火墙阻止
- 主节点宕机
- 网络问题

**解决方案**：

```bash
# 检查网络连通性
telnet 192.168.1.100 6379

# 检查防火墙
sudo firewall-cmd --list-all
```

### 3. 频繁故障转移

**原因**：

- `down-after-milliseconds` 设置过小
- 网络不稳定

**解决方案**：

```conf
# 增加判定时间
sentinel down-after-milliseconds mymaster 60000
```

### 4. 故障转移失败

**原因**：

- 没有可用的从节点
- 从节点数据太旧

**解决方案**：

- 确保至少有一个健康的从节点
- 检查从节点的复制状态

## 最佳实践

### 1. 哨兵数量

- **小型集群** - 3 个哨兵
- **中型集群** - 5 个哨兵
- **大型集群** - 7 个哨兵

### 2. quorum 设置

```
quorum = 哨兵数量 / 2 + 1
```

- 3 个哨兵 - quorum = 2
- 5 个哨兵 - quorum = 3
- 7 个哨兵 - quorum = 4

### 3. 超时配置

```conf
# 主节点下线判定时间：30-60 秒
sentinel down-after-milliseconds mymaster 30000

# 故障转移超时：3 分钟
sentinel failover-timeout mymaster 180000
```

### 4. 监控哨兵

定期检查哨兵状态：

```bash
# 查看哨兵日志
tail -f /var/log/redis/sentinel.log

# 查看主节点状态
redis-cli -p 26379 SENTINEL master mymaster
```

### 5. 测试故障转移

定期进行故障转移演练：

```bash
# 手动触发故障转移
redis-cli -p 26379 SENTINEL failover mymaster
```

## 小结

Redis Sentinel 提供了自动化的高可用解决方案：

- **自动故障转移** - 主节点故障时自动切换
- **监控** - 持续监控主从节点状态
- **配置中心** - 客户端自动发现主节点

部署要点：

- 奇数个哨兵（3 个或 5 个）
- 合理设置 quorum 和超时参数
- 分布式部署，避免单点故障
- 定期监控和演练故障转移
