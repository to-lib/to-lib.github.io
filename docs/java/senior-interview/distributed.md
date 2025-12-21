---
sidebar_position: 7
title: 分布式与中间件
---

# 🎯 分布式与中间件（专家级）

## 26. Redis 的持久化机制有哪些？如何选择？

**答案要点：**

**两种持久化方式对比：**

| 特性         | RDB                          | AOF               |
| ------------ | ---------------------------- | ----------------- |
| **原理**     | 快照，保存某时刻数据         | 追加写命令日志    |
| **文件大小** | 小（二进制压缩）             | 大（文本命令）    |
| **恢复速度** | 快                           | 慢（需重放命令）  |
| **数据安全** | 可能丢失最后一次快照后的数据 | 最多丢失 1 秒数据 |
| **性能影响** | fork 子进程时可能阻塞        | 每秒 fsync 影响小 |

**RDB 配置：**

```bash
# redis.conf
save 900 1      # 900秒内至少1个key变化则保存
save 300 10     # 300秒内至少10个key变化则保存
save 60 10000   # 60秒内至少10000个key变化则保存

# 手动触发
BGSAVE          # 后台异步保存
SAVE            # 同步保存（阻塞）
```

**AOF 配置：**

```bash
# redis.conf
appendonly yes
appendfilename "appendonly.aof"

# 同步策略
appendfsync always    # 每次写入都同步（最安全，最慢）
appendfsync everysec  # 每秒同步（推荐）
appendfsync no        # 由操作系统决定（最快，不安全）

# AOF 重写
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
```

**混合持久化（Redis 4.0+）：**

```bash
aof-use-rdb-preamble yes
# AOF 文件 = RDB 快照 + 增量 AOF 命令
# 兼顾恢复速度和数据安全
```

**选择建议：**

- 纯缓存场景：可以不开启持久化
- 数据安全要求高：AOF + everysec
- 快速恢复：RDB
- 最佳实践：混合持久化

**延伸：** 参考 [Redis 持久化](/docs/redis/persistence)

---

## 27. Kafka 如何保证消息不丢失？

**答案要点：**

**消息丢失的三个环节：**

```
Producer → Broker → Consumer
   ↓          ↓         ↓
 发送丢失   存储丢失   消费丢失
```

**Producer 端保证：**

```java
Properties props = new Properties();
// 1. acks 配置
props.put("acks", "all");  // 等待所有副本确认

// 2. 重试配置
props.put("retries", 3);
props.put("retry.backoff.ms", 1000);

// 3. 幂等性（防止重复）
props.put("enable.idempotence", true);

// 4. 同步发送或回调确认
producer.send(record, (metadata, exception) -> {
    if (exception != null) {
        // 发送失败，记录日志或重试
        log.error("Send failed", exception);
    }
});
```

**Broker 端保证：**

```bash
# server.properties
# 1. 副本数量
default.replication.factor=3

# 2. 最小同步副本数
min.insync.replicas=2

# 3. 禁止不完全选举
unclean.leader.election.enable=false
```

**Consumer 端保证：**

```java
Properties props = new Properties();
// 1. 手动提交 offset
props.put("enable.auto.commit", false);

// 2. 消费逻辑
while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        try {
            // 处理消息
            processMessage(record);
            // 处理成功后手动提交
            consumer.commitSync();
        } catch (Exception e) {
            // 处理失败，不提交，下次重新消费
            log.error("Process failed", e);
        }
    }
}
```

**延伸：** 参考 [Kafka 最佳实践](/docs/kafka/best-practices)

---

## 28. 如何设计分布式 ID 生成方案？

**答案要点：**

**常见方案对比：**

| 方案       | 优点           | 缺点           | 适用场景     |
| ---------- | -------------- | -------------- | ------------ |
| UUID       | 简单，无依赖   | 无序，存储大   | 非主键场景   |
| 数据库自增 | 简单，有序     | 性能瓶颈，单点 | 小规模系统   |
| Redis INCR | 性能高         | 依赖 Redis     | 中等规模     |
| 雪花算法   | 有序，高性能   | 时钟回拨问题   | 大规模分布式 |
| Leaf       | 高可用，高性能 | 复杂度高       | 大规模分布式 |

**雪花算法（Snowflake）：**

```
64位 ID 结构：
┌─────────────────────────────────────────────────────────────────┐
│ 0 │ 41位时间戳 │ 10位机器ID │ 12位序列号 │
└─────────────────────────────────────────────────────────────────┘
  ↓       ↓            ↓            ↓
符号位  毫秒级时间   机器标识    同毫秒序列
```

**Java 实现：**

```java
public class SnowflakeIdGenerator {
    private final long epoch = 1609459200000L;  // 起始时间戳
    private final long workerIdBits = 10L;
    private final long sequenceBits = 12L;

    private final long maxWorkerId = ~(-1L << workerIdBits);
    private final long sequenceMask = ~(-1L << sequenceBits);

    private final long workerIdShift = sequenceBits;
    private final long timestampShift = sequenceBits + workerIdBits;

    private long workerId;
    private long sequence = 0L;
    private long lastTimestamp = -1L;

    public SnowflakeIdGenerator(long workerId) {
        if (workerId > maxWorkerId || workerId < 0) {
            throw new IllegalArgumentException("Worker ID out of range");
        }
        this.workerId = workerId;
    }

    public synchronized long nextId() {
        long timestamp = System.currentTimeMillis();

        if (timestamp < lastTimestamp) {
            throw new RuntimeException("Clock moved backwards");
        }

        if (timestamp == lastTimestamp) {
            sequence = (sequence + 1) & sequenceMask;
            if (sequence == 0) {
                timestamp = waitNextMillis(lastTimestamp);
            }
        } else {
            sequence = 0L;
        }

        lastTimestamp = timestamp;

        return ((timestamp - epoch) << timestampShift)
                | (workerId << workerIdShift)
                | sequence;
    }

    private long waitNextMillis(long lastTimestamp) {
        long timestamp = System.currentTimeMillis();
        while (timestamp <= lastTimestamp) {
            timestamp = System.currentTimeMillis();
        }
        return timestamp;
    }
}
```

**延伸：** 参考 [分布式系统设计](/docs/microservices/design-patterns)

---

## 29. 如何实现分布式锁？有哪些方案？

**答案要点：**

**分布式锁方案对比：**

| 方案      | 优点             | 缺点             |
| --------- | ---------------- | ---------------- |
| MySQL     | 简单             | 性能差，单点     |
| Redis     | 性能高           | 主从切换可能丢锁 |
| ZooKeeper | 可靠性高         | 性能一般         |
| Etcd      | 可靠性高，性能好 | 复杂度高         |

**Redis 分布式锁实现：**

```java
public class RedisDistributedLock {
    private StringRedisTemplate redisTemplate;
    private String lockKey;
    private String lockValue;
    private long expireTime;

    public boolean tryLock() {
        lockValue = UUID.randomUUID().toString();
        Boolean success = redisTemplate.opsForValue()
            .setIfAbsent(lockKey, lockValue, expireTime, TimeUnit.MILLISECONDS);
        return Boolean.TRUE.equals(success);
    }

    public void unlock() {
        // Lua 脚本保证原子性
        String script =
            "if redis.call('get', KEYS[1]) == ARGV[1] then " +
            "   return redis.call('del', KEYS[1]) " +
            "else " +
            "   return 0 " +
            "end";
        redisTemplate.execute(
            new DefaultRedisScript<>(script, Long.class),
            Collections.singletonList(lockKey),
            lockValue
        );
    }
}
```

**Redisson 分布式锁（推荐）：**

```java
@Service
public class OrderService {
    @Autowired
    private RedissonClient redissonClient;

    public void createOrder(String orderId) {
        RLock lock = redissonClient.getLock("order:" + orderId);
        try {
            // 尝试获取锁，等待10秒，锁定30秒
            if (lock.tryLock(10, 30, TimeUnit.SECONDS)) {
                try {
                    // 业务逻辑
                    doCreateOrder(orderId);
                } finally {
                    lock.unlock();
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
```

**ZooKeeper 分布式锁原理：**

```
/locks/order
    ├── lock-0000000001  ← 客户端A（获得锁）
    ├── lock-0000000002  ← 客户端B（监听上一个节点）
    └── lock-0000000003  ← 客户端C（监听上一个节点）

1. 创建临时顺序节点
2. 获取所有子节点，判断自己是否最小
3. 如果是最小，获得锁；否则监听前一个节点
4. 前一个节点删除时，收到通知，重新判断
```

**延伸：** 参考 [Redis 分布式锁](/docs/redis/cache-strategies)

---

## 30. RPC 框架的核心原理是什么？

**答案要点：**

**RPC 调用流程：**

```
客户端                                    服务端
   │                                        │
   │  1. 调用本地代理                        │
   ▼                                        │
┌──────────┐                                │
│ Proxy    │                                │
└────┬─────┘                                │
     │ 2. 序列化                             │
     ▼                                      │
┌──────────┐                                │
│ Codec    │                                │
└────┬─────┘                                │
     │ 3. 网络传输                           │
     ▼                                      ▼
┌──────────┐    ─────────────────>    ┌──────────┐
│ Transport│                          │ Transport│
└──────────┘                          └────┬─────┘
                                           │ 4. 反序列化
                                           ▼
                                     ┌──────────┐
                                     │ Codec    │
                                     └────┬─────┘
                                           │ 5. 调用实际方法
                                           ▼
                                     ┌──────────┐
                                     │ Service  │
                                     └──────────┘
```

**核心组件：**

| 组件             | 作用               |
| ---------------- | ------------------ |
| **动态代理**     | 生成客户端代理对象 |
| **序列化**       | 对象与字节流转换   |
| **网络通信**     | 数据传输（Netty）  |
| **服务注册发现** | 服务地址管理       |
| **负载均衡**     | 请求分发策略       |

**简易 RPC 框架实现：**

```java
// 1. 服务接口
public interface UserService {
    User getUser(Long id);
}

// 2. 客户端代理
public class RpcProxy {
    @SuppressWarnings("unchecked")
    public static <T> T create(Class<T> interfaceClass) {
        return (T) Proxy.newProxyInstance(
            interfaceClass.getClassLoader(),
            new Class[]{interfaceClass},
            (proxy, method, args) -> {
                // 构建请求
                RpcRequest request = new RpcRequest();
                request.setClassName(interfaceClass.getName());
                request.setMethodName(method.getName());
                request.setParameterTypes(method.getParameterTypes());
                request.setParameters(args);

                // 发送请求
                RpcResponse response = sendRequest(request);

                return response.getResult();
            }
        );
    }
}

// 3. 服务端处理
public class RpcServer {
    private Map<String, Object> serviceMap = new HashMap<>();

    public void register(String serviceName, Object service) {
        serviceMap.put(serviceName, service);
    }

    public Object handle(RpcRequest request) throws Exception {
        Object service = serviceMap.get(request.getClassName());
        Method method = service.getClass().getMethod(
            request.getMethodName(),
            request.getParameterTypes()
        );
        return method.invoke(service, request.getParameters());
    }
}
```

## 31. 分布式共识算法（Paxos/Raft）是如何工作的？

**答案要点：**

**共识问题：** 在分布式系统中，如何让多个节点对某个值（或日志）达成一致。

**Raft 算法核心（易于理解）：**

Raft 将一致性问题分解为三个子问题：

1.  **Leader 选举（Leader Election）**
2.  **日志复制（Log Replication）**
3.  **安全性（Safety）**

**节点状态：**

- **Follower**：随从，被动接收请求。
- **Candidate**：候选人，竞选 Leader。
- **Leader**：领导者，处理所有客户端请求，同步日志给 Follower。

**选举过程：**

1.  节点启动时默认为 Follower。
2.  若超时未收到 Leader 心跳，转为 Candidate，发起投票。
3.  获得大多数（N/2 + 1）选票则成为 Leader。
4.  Leader 周期性发送心跳维持统治。

**日志复制过程：**

```
Client -> Leader -> (AppendEntries) -> Followers
         (Receive Command)
            |
            v
     (Write to Local Log)
            |
            v
     (Replicate to Followers)
            |
            v
     (Majority Acknowledge?) -> Yes -> Commit & Apply -> Response to Client
                                    -> Notify Followers to Commit
```

**Paxos vs Raft：**

| 特性         | Paxos                          | Raft               |
| ------------ | ------------------------------ | ------------------ |
| **理解难度** | 极难（理论性强）               | 较易（工程导向）   |
| **实现难度** | 极难                           | 有详细参考实现     |
| **应用**     | Zookeeper (ZAB), Google Chubby | Etcd, Consul, TIKV |

**延伸：** 参考 [分布式系统 - Raft 详解](/docs/distributed/raft)

---

**延伸：** 参考 [Netty 实战](/docs/netty/practical-examples)
