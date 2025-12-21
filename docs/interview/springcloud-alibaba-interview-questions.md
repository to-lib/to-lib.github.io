---
title: Spring Cloud Alibaba 面试题
sidebar_position: 15
---

# Spring Cloud Alibaba 面试题精选

> [!NOTE] > **面试必备**: 精选 Spring Cloud Alibaba 各组件的高频面试题,附带详细解答和扩展知识点。

## Nacos 面试题

### 1. Nacos 和 Eureka 的区别是什么?

**答案**:

| 特性     | Nacos                  | Eureka       |
| -------- | ---------------------- | ------------ |
| 功能     | 服务注册 + 配置管理    | 仅服务注册   |
| CAP 理论 | 支持 CP 和 AP 模式     | AP 模式      |
| 健康检查 | 支持 TCP/HTTP/MySQL 等 | 仅 HTTP      |
| 负载均衡 | 支持权重               | 基础负载均衡 |
| 服务管理 | 提供控制台             | 需自己搭建   |
| 维护状态 | 持续维护               | 已停止维护   |

**扩展**: Nacos 支持临时实例(AP)和持久化实例(CP),可以根据需求选择。

### 2. Nacos 如何保证服务注册的高可用?

**答案**:

1. **集群部署**: Nacos 支持集群部署,多节点之间数据同步
2. **数据持久化**: 使用 MySQL 存储服务注册信息
3. **健康检查**: 实时检测服务健康状态
4. **故障转移**: 某个节点故障时,客户端自动切换到其他节点

**集群架构**:

```
Client → Nacos Node1 (Leader)
      → Nacos Node2 (Follower) → MySQL
      → Nacos Node3 (Follower)
```

### 3. Nacos 配置中心如何实现动态刷新?

**答案**:

1. **长轮询机制**: 客户端通过长轮询监听配置变化
2. **监听器**: Nacos Client 注册监听器
3. **推送通知**: 配置变更后,Server 推送通知给 Client
4. **@RefreshScope**: Spring Cloud 通过该注解实现配置刷新

**代码示例**:

```java
@RestController
@RefreshScope  // 关键注解
public class ConfigController {
    @Value("${config.value}")
    private String configValue;
}
```

### 4. Nacos 的命名空间、分组、DataID 有什么作用?

**答案**:

**三层隔离结构**:

```
Namespace (环境隔离)
  └── Group (业务隔离)
      └── DataID (具体配置)
```

- **Namespace**: 用于不同环境隔离 (dev/test/prod)
- **Group**: 用于不同业务模块隔离 (order/user/payment)
- **DataID**: 具体的配置文件标识

**示例**:

```
Namespace: prod
  Group: ORDER_GROUP
    DataID: order-service.yaml
    DataID: order-service-prod.yaml
```

### 5. Nacos 如何实现服务发现的负载均衡?

**答案**:

Nacos 支持多种负载均衡策略:

1. **权重负载均衡**: 设置服务实例权重
2. **保护阈值**: 健康实例比例低于阈值时返回所有实例
3. **元数据路由**: 根据元数据进行路由选择

**配置权重**:

```yaml
spring:
  cloud:
    nacos:
      discovery:
        weight: 0.5 # 权重 0-1
```

## Sentinel 面试题

### 6. Sentinel 和 Hystrix 的区别是什么?

**答案**:

| 特性         | Sentinel               | Hystrix           |
| ------------ | ---------------------- | ----------------- |
| 隔离策略     | 信号量隔离             | 线程池隔离/信号量 |
| 熔断降级策略 | 慢调用/异常比例/异常数 | 异常比例          |
| 实时指标实现 | 滑动窗口               | 滑动窗口          |
| 规则配置     | 支持多种数据源         | 基于代码配置      |
| 扩展性       | 多种扩展点             | 插件式            |
| 控制台       | 开箱即用               | 需单独搭建        |
| 维护状态     | 持续维护               | 已停止维护        |

### 7. Sentinel 的限流算法有哪些?

**答案**:

**1. 滑动窗口算法**:

- Sentinel 默认使用
- 将时间分成多个小窗口
- 统计每个窗口的请求数

**2. 漏桶算法**:

- 请求以固定速率流出
- 超过容量的请求被丢弃

**3. 令牌桶算法**:

- 固定速率生成令牌
- 请求消耗令牌才能通过

**Sentinel 实现**:

```java
// 使用排队等待实现类似漏桶
@SentinelResource(value = "api")
// 控制台配置: 流控效果选择 "排队等待"
```

### 8. Sentinel 如何实现熔断降级?

**答案**:

**三种熔断策略**:

1. **慢调用比例**: 响应时间超过阈值的比例
2. **异常比例**: 异常占总请求的比例
3. **异常数**: 异常数量超过阈值

**熔断状态机**:

```
Closed (关闭) → Open (熔断) → Half-Open (半开) → Closed
```

**示例配置**:

```java
// 慢调用比例熔断
// - 最大RT: 1000ms
// - 比例阈值: 0.5 (50%)
// - 最小请求数: 5
// - 熔断时长: 10s
```

### 9. Sentinel 如何持久化规则?

**答案**:

**持久化方式**:

1. **文件数据源**: 规则保存到本地文件
2. **Nacos 数据源**: 规则保存到 Nacos (推荐)
3. **Apollo 数据源**: 规则保存到 Apollo
4. **ZooKeeper 数据源**: 规则保存到 ZooKeeper

**Nacos 持久化配置**:

```yaml
spring:
  cloud:
    sentinel:
      datasource:
        flow:
          nacos:
            server-addr: localhost:8848
            dataId: ${spring.application.name}-flow-rules
            groupId: SENTINEL_GROUP
            rule-type: flow
```

## Seata 面试题

### 10. Seata 的四种事务模式有什么区别?

**答案**:

| 模式 | 一致性   | 性能 | 复杂度 | 使用场景               |
| ---- | -------- | ---- | ------ | ---------------------- |
| AT   | 最终一致 | 中   | 低     | 大部分业务场景         |
| TCC  | 最终一致 | 高   | 高     | 对性能要求高的场景     |
| SAGA | 最终一致 | 高   | 中     | 长流程业务             |
| XA   | 强一致   | 低   | 低     | 对一致性要求极高的场景 |

**选择建议**: 优先使用 AT 模式,性能要求高时使用 TCC。

### 11. Seata AT 模式的工作原理是什么?

**答案**:

**两阶段提交**:

**一阶段**:

1. 解析 SQL,获取前后镜像
2. 执行业务 SQL
3. 记录 undo log (回滚日志)
4. 提交本地事务
5. 向 TC 注册分支事务

**二阶段提交**:

- 删除 undo log

**二阶段回滚**:

- 根据 undo log 反向补偿

**示例**:

```sql
-- 原始 SQL
UPDATE account SET balance = balance - 100 WHERE id = 1;

-- Undo Log (简化)
{
  "beforeImage": {"balance": 1000},
  "afterImage": {"balance": 900}
}

-- 回滚时
UPDATE account SET balance = 1000 WHERE id = 1;
```

### 12. Seata 如何保证分布式事务的一致性?

**答案**:

**三个角色**:

- **TC** (Transaction Coordinator): 事务协调器
- **TM** (Transaction Manager): 事务管理器
- **RM** (Resource Manager): 资源管理器

**工作流程**:

```
1. TM 向 TC 申请开启全局事务,TC 返回 XID
2. RM 向 TC 注册分支事务,关联 XID
3. TM 根据业务执行结果决定提交或回滚
4. TC 通知所有 RM 提交或回滚分支事务
```

**一致性保证**:

- 通过全局事务 ID (XID) 关联所有分支事务
- TC 协调所有分支事务的提交或回滚
- 失败时通过 undo log 回滚

### 13. Seata 和本地事务的区别是什么?

**答案**:

| 特性       | 本地事务   | Seata 分布式事务 |
| ---------- | ---------- | ---------------- |
| 范围       | 单数据库   | 跨数据库/跨服务  |
| ACID       | 严格 ACID  | 最终一致性       |
| 性能       | 高         | 相对较低         |
| 实现复杂度 | 简单       | 复杂             |
| 回滚机制   | 数据库自动 | 通过 undo log    |

## RocketMQ 面试题

### 14. RocketMQ 如何保证消息不丢失?

**答案**:

**三个阶段保证**:

**1. 生产者阶段**:

- 使用同步发送
- 发送确认机制
- 失败重试

```java
SendResult result = producer.send(msg);
if (result.getSendStatus() != SendStatus.SEND_OK) {
    // 重试
}
```

**2. Broker 阶段**:

- 消息持久化到磁盘
- 主从同步

**3. 消费者阶段**:

- 先消费后确认
- 消费失败自动重试

### 15. RocketMQ 如何保证消息的顺序性?

**答案**:

**两种顺序**:

**1. 全局顺序**:

- 单 Topic 单队列
- 吞吐量低

**2. 分区顺序** (推荐):

- 同一 orderId 发送到同一队列
- 顺序消费

**实现**:

```java
// 发送
rocketMQTemplate.syncSendOrderly(
    "topic",
    message,
    orderId  // 相同 orderId 进入同一队列
);

// 消费
@RocketMQMessageListener(
    topic = "topic",
    consumeMode = ConsumeMode.ORDERLY
)
```

### 16. RocketMQ 事务消息的工作原理是什么?

**答案**:

**三个步骤**:

1. **发送半消息**: 消息对消费者不可见
2. **执行本地事务**
3. **提交或回滚半消息**

**回查机制**:

- 如果长时间未收到确认,Broker 会回查本地事务状态

**流程图**:

```
1. Producer → Broker: 发送半消息
2. Producer: 执行本地事务
3. Producer → Broker: COMMIT/ROLLBACK
4. 如果超时未确认:
   Broker → Producer: 回查事务状态
   Producer → Broker: 返回状态
```

### 17. RocketMQ 如何处理消息积压?

**答案**:

**原因分析**:

- 消费速度 < 生产速度
- 消费者处理慢
- 消费者宕机

**解决方案**:

1. **增加消费者**: 提高并发
2. **批量消费**: 减少网络开销
3. **异步处理**: 消费逻辑异步
4. **临时队列**: 紧急情况转储
5. **优化消费逻辑**: 减少耗时操作

**监控消息积压**:

```java
// RocketMQ 控制台查看
// Consumer 消费进度
```

## Dubbo 面试题

### 18. Dubbo 的架构是怎样的?

**答案**:

**五个角色**:

```
Provider (服务提供者)
Consumer (服务消费者)
Registry (注册中心 - Nacos)
Monitor (监控中心)
Container (服务容器)
```

**调用流程**:

```
1. Provider 启动,向 Registry 注册服务
2. Consumer 启动,向 Registry 订阅服务
3. Registry 返回服务地址给 Consumer
4. Consumer 根据负载均衡算法调用 Provider
5. Consumer 和 Provider 定时向 Monitor 发送统计数据
```

### 19. Dubbo 支持哪些负载均衡策略?

**答案**:

| 策略           | 说明           | 使用场景             |
| -------------- | -------------- | -------------------- |
| Random         | 随机 (默认)    | 通用                 |
| RoundRobin     | 轮询           | 各服务器性能相近     |
| LeastActive    | 最少活跃调用数 | 处理能力差异大       |
| ConsistentHash | 一致性 Hash    | 同一参数调用同一服务 |

**配置**:

```java
@DubboReference(loadbalance = "roundrobin")
private UserService userService;
```

### 20. Dubbo 如何实现服务降级?

**答案**:

**降级策略**:

1. **Mock**: 返回 Mock 数据
2. **Failsafe**: 失败安全,忽略异常
3. **Failfast**: 快速失败

**Mock 降级**:

```java
// 1. 创建 Mock 类
public class UserServiceMock implements UserService {
    @Override
    public User getUser(Long id) {
        return new User(id, "降级用户", "mock@example.com");
    }
}

// 2. 配置
@DubboReference(
    mock = "com.example.UserServiceMock",
    timeout = 3000
)
private UserService userService;
```

### 21. Dubbo 和 Feign 的区别是什么?

**答案**:

| 特性     | Dubbo             | Feign            |
| -------- | ----------------- | ---------------- |
| 协议     | Dubbo/gRPC/HTTP   | HTTP             |
| 性能     | 高 (二进制序列化) | 中 (JSON)        |
| 负载均衡 | 丰富的策略        | Ribbon           |
| 容错     | 多种集群容错策略  | Hystrix/Sentinel |
| 使用     | 需要接口 JAR 包   | 只需服务名       |
| 适用场景 | 高性能内部调用    | 通用 HTTP 调用   |

## 综合面试题

### 22. 微服务架构中如何保证数据一致性?

**答案**:

**方案对比**:

| 方案       | 一致性   | 性能 | 复杂度 |
| ---------- | -------- | ---- | ------ |
| 分布式事务 | 强一致   | 低   | 高     |
| 最终一致性 | 最终一致 | 高   | 中     |
| 补偿机制   | 最终一致 | 高   | 高     |

**推荐方案**: Seata AT 模式 + 消息队列

### 23. 如何设计一个高可用的微服务系统?

**答案**:

**架构设计**:

1. **服务注册发现**: Nacos 集群
2. **负载均衡**: Dubbo/Gateway
3. **限流熔断**: Sentinel
4. **分布式事务**: Seata
5. **消息队列**: RocketMQ
6. **配置中心**: Nacos
7. **链路追踪**: Sleuth + Zipkin
8. **监控告警**: Prometheus + Grafana

**关键要素**:

- 服务多实例部署
- 数据库主从 + 读写分离
- 缓存集群 (Redis Cluster)
- 限流熔断
- 监控告警

### 24. Spring Cloud 和 Spring Cloud Alibaba 如何选择?

**答案**:

**选择 Spring Cloud**:

- 需要与 Spring 官方生态深度集成
- 使用 AWS、Azure 等云服务

**选择 Spring Cloud Alibaba**:

- 国内项目,需要中文文档和社区支持
- 需要更好的性能 (Dubbo)
- 需要分布式事务解决方案 (Seata)
- 需要高性能消息队列 (RocketMQ)

**推荐**: 两者可以混用,取长补短

---

**面试技巧**:

- 理解原理比记答案更重要
- 结合实际项目经验回答
- 主动扩展相关知识点
- 诚实面对不懂的问题

**下一步**: 继续学习 [FAQ 常见问题](/docs/springcloud-alibaba/faq)
