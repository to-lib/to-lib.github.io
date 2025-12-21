---
sidebar_position: 5
title: 架构设计
---

# 🎯 架构设计（专家级）

## 16. 什么是 CAP 理论？如何在实际系统中权衡？

**答案要点：**

**CAP 三要素：**

- **C（Consistency）一致性**：所有节点同一时刻数据相同
- **A（Availability）可用性**：每个请求都能得到响应
- **P（Partition tolerance）分区容错**：网络分区时系统仍能运行

**CAP 不可能三角：**

```
        C（一致性）
           /\
          /  \
         /    \
        /  CA  \
       /________\
      A          P
   (可用性)  (分区容错)

分布式系统必须选择 P，因此只能在 CP 和 AP 之间选择
```

**实际系统选择：**

| 系统 | 选择 | 说明 |
|------|------|------|
| ZooKeeper | CP | 强一致性，可能短暂不可用 |
| Eureka | AP | 高可用，允许数据不一致 |
| Redis Cluster | AP | 异步复制，可能丢数据 |
| MySQL 主从 | CP | 同步复制保证一致性 |

**BASE 理论（AP 的延伸）：**

```
BA - Basically Available（基本可用）
S  - Soft state（软状态）
E  - Eventually consistent（最终一致性）
```

**实际权衡示例：**

```java
// 电商下单场景
// 1. 库存扣减 - 需要强一致性（CP）
// 2. 订单创建 - 需要高可用（AP）
// 3. 通知发送 - 最终一致性即可

@Transactional
public Order createOrder(OrderRequest request) {
    // CP: 同步扣减库存
    inventoryService.deduct(request.getProductId(), request.getQuantity());
    
    // AP: 创建订单
    Order order = orderRepository.save(new Order(request));
    
    // 最终一致性: 异步发送通知
    messageQueue.send(new OrderCreatedEvent(order));
    
    return order;
}
```

**延伸：** 参考 [微服务 - 分布式理论](/docs/microservices/core-concepts)

---

## 17. 分布式事务有哪些解决方案？各有什么优缺点？

**答案要点：**

**分布式事务方案对比：**

| 方案 | 一致性 | 性能 | 复杂度 | 适用场景 |
|------|--------|------|--------|---------|
| 2PC | 强一致 | 低 | 中 | 数据库分布式事务 |
| TCC | 最终一致 | 中 | 高 | 资金交易 |
| Saga | 最终一致 | 高 | 中 | 长事务 |
| 本地消息表 | 最终一致 | 高 | 低 | 异步场景 |
| MQ 事务消息 | 最终一致 | 高 | 低 | 消息驱动 |

**TCC 实现示例：**

```java
// TCC: Try-Confirm-Cancel
public interface AccountService {
    // Try: 预留资源
    @TwoPhaseBusinessAction(name = "deduct", 
        commitMethod = "confirm", rollbackMethod = "cancel")
    boolean tryDeduct(BusinessActionContext context, 
                      @BusinessActionContextParameter("accountId") String accountId,
                      @BusinessActionContextParameter("amount") BigDecimal amount);
    
    // Confirm: 确认提交
    boolean confirm(BusinessActionContext context);
    
    // Cancel: 取消回滚
    boolean cancel(BusinessActionContext context);
}

@Service
public class AccountServiceImpl implements AccountService {
    
    @Override
    public boolean tryDeduct(BusinessActionContext context, 
                             String accountId, BigDecimal amount) {
        // 冻结金额
        accountDao.freeze(accountId, amount);
        return true;
    }
    
    @Override
    public boolean confirm(BusinessActionContext context) {
        String accountId = context.getActionContext("accountId");
        BigDecimal amount = context.getActionContext("amount");
        // 扣减冻结金额
        accountDao.deductFrozen(accountId, amount);
        return true;
    }
    
    @Override
    public boolean cancel(BusinessActionContext context) {
        String accountId = context.getActionContext("accountId");
        BigDecimal amount = context.getActionContext("amount");
        // 解冻金额
        accountDao.unfreeze(accountId, amount);
        return true;
    }
}
```

**本地消息表方案：**

```java
@Transactional
public void createOrder(OrderRequest request) {
    // 1. 创建订单
    Order order = orderRepository.save(new Order(request));
    
    // 2. 写入本地消息表（同一事务）
    LocalMessage message = new LocalMessage();
    message.setMessageId(UUID.randomUUID().toString());
    message.setContent(JSON.toJSONString(new OrderCreatedEvent(order)));
    message.setStatus("PENDING");
    localMessageRepository.save(message);
}

// 定时任务发送消息
@Scheduled(fixedRate = 1000)
public void sendPendingMessages() {
    List<LocalMessage> messages = localMessageRepository.findByStatus("PENDING");
    for (LocalMessage message : messages) {
        try {
            messageQueue.send(message.getContent());
            message.setStatus("SENT");
            localMessageRepository.save(message);
        } catch (Exception e) {
            // 重试
        }
    }
}
```

**延伸：** 参考 [微服务 - 分布式事务](/docs/microservices/design-patterns)

---

## 18. 如何设计一个高可用的系统？

**答案要点：**

**高可用设计原则：**

```
可用性 = MTBF / (MTBF + MTTR)

MTBF: 平均故障间隔时间
MTTR: 平均修复时间

99.9%  = 8.76 小时/年 停机
99.99% = 52.6 分钟/年 停机
```

**高可用架构设计：**

```
                    ┌─────────────┐
                    │   DNS/CDN   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
        │   LB-1    │ │  LB-2   │ │   LB-3    │  负载均衡
        └─────┬─────┘ └────┬────┘ └─────┬─────┘
              │            │            │
    ┌─────────┼────────────┼────────────┼─────────┐
    │         │            │            │         │
┌───▼───┐ ┌───▼───┐ ┌──────▼──────┐ ┌───▼───┐ ┌───▼───┐
│ App-1 │ │ App-2 │ │    App-3    │ │ App-4 │ │ App-5 │  应用集群
└───┬───┘ └───┬───┘ └──────┬──────┘ └───┬───┘ └───┬───┘
    │         │            │            │         │
    └─────────┼────────────┼────────────┼─────────┘
              │            │            │
        ┌─────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
        │  Redis-M  │ │ Redis-S │ │  Redis-S  │  缓存集群
        └───────────┘ └─────────┘ └───────────┘
              │
        ┌─────▼─────┐ ┌─────────┐ ┌───────────┐
        │  MySQL-M  │ │ MySQL-S │ │  MySQL-S  │  数据库集群
        └───────────┘ └─────────┘ └───────────┘
```

**限流熔断实现：**

```java
// Sentinel 限流配置
@SentinelResource(value = "getUser", 
    blockHandler = "getUserBlockHandler",
    fallback = "getUserFallback")
public User getUser(String id) {
    return userService.getUser(id);
}

// 限流处理
public User getUserBlockHandler(String id, BlockException e) {
    return new User("限流中，请稍后重试");
}

// 降级处理
public User getUserFallback(String id, Throwable e) {
    return new User("服务暂时不可用");
}
```

**延伸：** 参考 [微服务 - 服务治理](/docs/microservices/service-governance)

---

## 19. 如何设计一个秒杀系统？

**答案要点：**

**秒杀系统架构：**

```
用户请求 → CDN → 网关（限流）→ 秒杀服务 → Redis → 消息队列 → 订单服务
                    ↓
              静态页面缓存
```

**核心设计要点：**

```java
// 1. 库存预热到 Redis
@PostConstruct
public void initStock() {
    List<SeckillProduct> products = productService.getSeckillProducts();
    for (SeckillProduct product : products) {
        redisTemplate.opsForValue().set(
            "seckill:stock:" + product.getId(), 
            product.getStock()
        );
    }
}

// 2. Redis 原子扣减库存
public boolean deductStock(Long productId) {
    String key = "seckill:stock:" + productId;
    Long stock = redisTemplate.opsForValue().decrement(key);
    if (stock < 0) {
        // 库存不足，恢复
        redisTemplate.opsForValue().increment(key);
        return false;
    }
    return true;
}

// 3. 异步创建订单
@Transactional
public void createOrder(SeckillRequest request) {
    // 扣减 Redis 库存
    if (!deductStock(request.getProductId())) {
        throw new SeckillException("库存不足");
    }
    
    // 发送消息异步创建订单
    OrderMessage message = new OrderMessage(
        request.getUserId(), 
        request.getProductId()
    );
    kafkaTemplate.send("seckill-order", message);
}

// 4. 消费者处理订单
@KafkaListener(topics = "seckill-order")
public void handleOrder(OrderMessage message) {
    // 创建订单
    Order order = new Order();
    order.setUserId(message.getUserId());
    order.setProductId(message.getProductId());
    orderRepository.save(order);
    
    // 扣减数据库库存
    productRepository.deductStock(message.getProductId());
}
```

**防刷策略：**

```java
// 1. 用户限流
@RateLimiter(key = "seckill:user:#userId", rate = 1, interval = 1)
public void seckill(Long userId, Long productId) { }

// 2. IP 限流
@RateLimiter(key = "seckill:ip:#ip", rate = 10, interval = 1)
public void seckill(String ip, Long productId) { }

// 3. 验证码
// 4. 隐藏秒杀接口（动态 URL）
```

**延伸：** 参考 [微服务 - 高并发设计](/docs/microservices/design-patterns)

---

## 20. 微服务拆分的原则是什么？如何确定服务边界？

**答案要点：**

**服务拆分原则：**

| 原则 | 说明 |
|------|------|
| **单一职责** | 每个服务只负责一个业务领域 |
| **高内聚低耦合** | 服务内部高度相关，服务间依赖最小 |
| **业务边界清晰** | 基于领域驱动设计（DDD）划分 |
| **数据独立** | 每个服务拥有独立的数据存储 |
| **可独立部署** | 服务可以独立开发、测试、部署 |

**DDD 领域划分：**

```
电商系统领域划分

┌─────────────────────────────────────────────────────────┐
│                      核心域                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ 商品域   │  │ 订单域   │  │ 支付域   │  │ 库存域   │    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                      支撑域                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
│  │ 用户域   │  │ 营销域   │  │ 物流域   │                 │
│  └─────────┘  └─────────┘  └─────────┘                 │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                      通用域                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
│  │ 消息通知 │  │ 文件存储 │  │ 日志监控 │                 │
│  └─────────┘  └─────────┘  └─────────┘                 │
└─────────────────────────────────────────────────────────┘
```

**服务拆分反模式：**

```java
// ❌ 错误：分布式单体
// 服务间强依赖，同步调用链过长
OrderService → InventoryService → PaymentService → LogisticsService

// ✅ 正确：事件驱动解耦
OrderService --发布事件--> EventBus
                              ↓
              InventoryService（订阅）
              PaymentService（订阅）
              LogisticsService（订阅）
```

**延伸：** 参考 [微服务 - 核心概念](/docs/microservices/core-concepts)
