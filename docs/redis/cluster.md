---
sidebar_position: 7
title: Redis 集群
---

# Redis 集群

Redis Cluster 是 Redis 官方提供的分布式解决方案，实现了数据自动分片、高可用和水平扩展。

## 集群架构

### 数据分片

Redis Cluster 将所有数据分成 16384 个槽位（slot），每个节点负责部分槽位。

```
Slot 0-5460:  Node1
Slot 5461-10922: Node2
Slot 10923-16383: Node3
```

### 主从架构

每个主节点可以有多个从节点，实现数据冗余和高可用。

```
Master1 (0-5460)    Master2 (5461-10922)    Master3 (10923-16383)
    |                      |                        |
 Slave1                 Slave2                   Slave3
```

## 搭建集群

### 1. 准备节点

创建 6 个 Redis 实例（3 主 3 从）。

**节点 7001 配置** (`/etc/redis/redis-7001.conf`)：

```conf
# 端口
port 7001

# 开启集群模式
cluster-enabled yes

# 集群配置文件（自动生成）
cluster-config-file nodes-7001.conf

# 节点超时时间
cluster-node-timeout 15000

# 持久化
appendonly yes

# 工作目录
dir /var/lib/redis/7001

# 后台运行
daemonize yes

# 日志文件
logfile /var/log/redis/redis-7001.log
```

类似地配置 7002、7003、7004、7005、7006 六个节点。

### 2. 启动节点

```bash
redis-server /etc/redis/redis-7001.conf
redis-server /etc/redis/redis-7002.conf
redis-server /etc/redis/redis-7003.conf
redis-server /etc/redis/redis-7004.conf
redis-server /etc/redis/redis-7005.conf
redis-server /etc/redis/redis-7006.conf
```

### 3. 创建集群

使用 `redis-cli` 创建集群：

```bash
redis-cli --cluster create \
  127.0.0.1:7001 \
  127.0.0.1:7002 \
  127.0.0.1:7003 \
  127.0.0.1:7004 \
  127.0.0.1:7005 \
  127.0.0.1:7006 \
  --cluster-replicas 1
```

参数说明：

- `--cluster-replicas 1` - 每个主节点有 1 个从节点
- 前 3 个节点为主节点，后 3 个为从节点

### 4. 验证集群

```bash
# 连接集群
redis-cli -c -p 7001

# 查看集群信息
CLUSTER INFO

# 查看节点信息
CLUSTER NODES
```

## 槽位分配

### 哈希槽计算

Redis 使用 CRC16 算法计算键的哈希槽：

```
slot = CRC16(key) % 16384
```

### 哈希标签

使用 `{}` 可以指定计算哈希的部分：

```bash
# user:1001 和 user:1002 会分配到不同槽位
SET user:1001 "data1"
SET user:1002 "data2"

# 使用哈希标签，确保分配到同一槽位
SET {user}:1001 "data1"
SET {user}:1002 "data2"
```

### 查看槽位分配

```bash
# 查看键所在的槽位
CLUSTER KEYSLOT mykey

# 查看槽位包含的键数量
CLUSTER COUNTKEYSINSLOT 5460

# 获取槽位中的键
CLUSTER GETKEYSINSLOT 5460 10
```

## 集群操作

### 数据操作

```bash
# 连接集群（-c 参数自动重定向）
redis-cli -c -p 7001

# 设置值
SET key1 "value1"

# 如果键不在当前节点，Redis 会自动重定向
-> Redirected to slot [9189] located at 127.0.0.1:7002
```

### 集群命令

```bash
# 查看集群信息
CLUSTER INFO

# 查看节点信息
CLUSTER NODES

# 查看当前节点的槽位
CLUSTER SLOTS

# 查看槽位状态
CLUSTER KEYSLOT key

# 手动重定向
ASKING  # 临时允许客户端在错误的节点执行命令
```

## 扩容与缩容

### 添加主节点

**1. 启动新节点**：

```bash
redis-server /etc/redis/redis-7007.conf
```

**2. 添加到集群**：

```bash
redis-cli --cluster add-node 127.0.0.1:7007 127.0.0.1:7001
```

**3. 分配槽位**：

```bash
# 重新分片
redis-cli --cluster reshard 127.0.0.1:7001

# 按提示输入：
# - 要移动多少槽位
# - 目标节点 ID
# - 源节点 ID（all 表示从所有节点均匀移动）
```

### 添加从节点

```bash
redis-cli --cluster add-node 127.0.0.1:7008 127.0.0.1:7001 \
  --cluster-slave \
  --cluster-master-id <master-node-id>
```

### 删除节点

**1. 移除从节点**：

```bash
redis-cli --cluster del-node 127.0.0.1:7001 <node-id>
```

**2. 移除主节点**：

```bash
# 先将槽位迁移到其他节点
redis-cli --cluster reshard 127.0.0.1:7001

# 然后删除节点
redis-cli --cluster del-node 127.0.0.1:7001 <node-id>
```

## 故障转移

### 自动故障转移

当主节点下线时，集群自动进行故障转移：

1. 检测主节点下线（超过 `cluster-node-timeout`）
2. 从节点发起选举
3. 获得多数投票的从节点晋升为主节点
4. 新主节点接管槽位

### 手动故障转移

```bash
# 在从节点执行
CLUSTER FAILOVER
```

## 客户端连接

### Java（Jedis）

```java
import redis.clients.jedis.HostAndPort;
import redis.clients.jedis.JedisCluster;

import java.util.HashSet;
import java.util.Set;

public class ClusterExample {
    public static void main(String[] args) {
        // 集群节点
        Set<HostAndPort> nodes = new HashSet<>();
        nodes.add(new HostAndPort("127.0.0.1", 7001));
        nodes.add(new HostAndPort("127.0.0.1", 7002));
        nodes.add(new HostAndPort("127.0.0.1", 7003));

        // 创建集群连接
        JedisCluster cluster = new JedisCluster(nodes);

        // 操作数据
        cluster.set("key", "value");
        String value = cluster.get("key");
        System.out.println(value);

        cluster.close();
    }
}
```

### Spring Boot

```yaml
spring:
  redis:
    cluster:
      nodes:
        - 127.0.0.1:7001
        - 127.0.0.1:7002
        - 127.0.0.1:7003
      max-redirects: 3
```

```java
@Configuration
public class RedisConfig {
    @Bean
    public RedisConnectionFactory redisConnectionFactory() {
        RedisClusterConfiguration config = new RedisClusterConfiguration();
        config.addClusterNode(new RedisNode("127.0.0.1", 7001));
        config.addClusterNode(new RedisNode("127.0.0.1", 7002));
        config.addClusterNode(new RedisNode("127.0.0.1", 7003));

        return new LettuceConnectionFactory(config);
    }
}
```

## 集群管理命令

### 集群维护

```bash
# 检查集群
redis-cli --cluster check 127.0.0.1:7001

# 修复集群
redis-cli --cluster fix 127.0.0.1:7001

# 重新平衡槽位
redis-cli --cluster rebalance 127.0.0.1:7001

# 查看集群信息
redis-cli --cluster info 127.0.0.1:7001
```

### 槽位管理

```bash
# 手动分配槽位
CLUSTER ADDSLOTS 0 1 2 3 4 5

# 删除槽位
CLUSTER DELSLOTS 0 1 2

# 设置槽位状态为导入中
CLUSTER SETSLOT 100 IMPORTING <node-id>

# 设置槽位状态为迁移中
CLUSTER SETSLOT 100 MIGRATING <node-id>

# 设置槽位的负责节点
CLUSTER SETSLOT 100 NODE <node-id>
```

## 集群配置参数

```conf
# 开启集群模式
cluster-enabled yes

# 集群配置文件（自动生成和更新）
cluster-config-file nodes.conf

# 节点超时时间（毫秒）
cluster-node-timeout 15000

# 故障转移需要的从节点数据新鲜度
cluster-replica-validity-factor 10

# 要求至少有一个从节点才允许故障转移
cluster-require-full-coverage yes

# 从节点迁移的最小间隔时间
cluster-migration-barrier 1

# 是否允许从节点在主节点下线时提供读服务
cluster-allow-reads-when-down no
```

## 集群限制

### 不支持的功能

1. **多键操作** - 只支持在同一槽位的多键操作
2. **事务** - 只支持同一槽位的事务
3. **数据库** - 只支持 0 号数据库
4. **复制** - 使用集群内置的复制机制

### 解决方案

**多键操作**：

使用哈希标签确保键在同一槽位：

```bash
# 不同槽位（可能失败）
MGET user:1001 user:1002

# 使用哈希标签（同一槽位）
MGET {user}:1001 {user}:1002
```

## 最佳实践

### 1. 节点规划

- **最少 3 主 3 从** - 保证高可用
- **奇数个主节点** - 避免投票平局
- **分布式部署** - 不同物理机或可用区

### 2. 槽位分配

- 均匀分配槽位到各个主节点
- 使用哈希标签控制键的分布

### 3. 监控

定期检查集群状态：

```bash
# 检查集群健康
redis-cli --cluster check 127.0.0.1:7001

# 查看集群信息
redis-cli -p 7001 CLUSTER INFO
```

### 4. 备份

使用 `BGSAVE` 定期备份各节点：

```bash
redis-cli -p 7001 BGSAVE
redis-cli -p 7002 BGSAVE
redis-cli -p 7003 BGSAVE
```

### 5. 扩容策略

- **预留容量** - 提前扩容，避免临时扩容
- **均匀迁移** - 从所有节点均匀迁移槽位
- **低峰期操作** - 在业务低峰期扩缩容

## 常见问题

### 1. 集群无法启动

**原因**：

- 配置文件错误
- 端口被占用
- 防火墙阻止

**解决方案**：

```bash
# 检查端口
netstat -tunlp | grep 7001

# 检查配置
redis-server /etc/redis/redis-7001.conf --test-config

# 查看日志
tail -f /var/log/redis/redis-7001.log
```

### 2. 槽位未完全覆盖

**原因**：

- 节点下线
- 槽位分配不完整

**解决方案**：

```bash
# 检查集群
redis-cli --cluster check 127.0.0.1:7001

# 修复集群
redis-cli --cluster fix 127.0.0.1:7001
```

### 3. 数据倾斜

**原因**：

- 槽位分配不均
- 热点数据集中在某些槽位

**解决方案**：

```bash
# 重新平衡槽位
redis-cli --cluster rebalance 127.0.0.1:7001
```

### 4. 故障转移失败

**原因**：

- 没有可用的从节点
- 过半节点下线

**解决方案**：

- 确保每个主节点至少有一个健康的从节点
- 修复下线的节点

## 小结

Redis Cluster 提供了完整的分布式解决方案：

- **数据分片** - 支持大数据量和高并发
- **高可用** - 自动故障转移
- **水平扩展** - 在线扩缩容

关键特性：

- 16384 个槽位自动分片
- 主从架构保证高可用
- 支持在线扩缩容
- 客户端智能路由

适用场景：

- 数据量超过单机内存
- 需要高可用保证
- 需要水平扩展能力
