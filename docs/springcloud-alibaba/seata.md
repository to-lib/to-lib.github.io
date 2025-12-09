---
id: seata
title: Seata 分布式事务
sidebar_label: Seata
sidebar_position: 5
---

# Seata 分布式事务

> [!TIP] > **完整的分布式事务解决方案**: Seata 提供 AT、TCC、SAGA、XA 四种事务模式，解决微服务架构下的数据一致性问题。

## 1. Seata 简介

**Seata** (Simple Extensible Autonomous Transaction Architecture) 是阿里巴巴开源的分布式事务解决方案。

### 事务模式

| 模式 | 说明               | 使用场景               |
| ---- | ------------------ | ---------------------- |
| AT   | 自动补偿，无侵入   | 大部分业务场景         |
| TCC  | 手动补偿，性能高   | 对性能要求高的场景     |
| SAGA | 长事务，最终一致性 | 长流程业务             |
| XA   | 强一致性           | 对一致性要求极高的场景 |

## 2. Seata Server 安装

```bash
# 下载
wget https://github.com/seata/seata/releases/download/v1.7.0/seata-server-1.7.0.zip

# 解压
unzip seata-server-1.7.0.zip

# 启动
cd seata
sh bin/seata-server.sh
```

访问控制台：`http://localhost:7091`

## 3. AT 模式

### 工作原理

```
一阶段：
1. 解析 SQL，获取前后镜像
2. 执行业务 SQL
3. 记录 undo log
4. 提交本地事务
5. 向 TC（Seata Server）注册分支事务

二阶段提交：
- 删除 undo log

二阶段回滚：
- 根据 undo log 回滚
```

### 添加依赖

```xml
<dependency>
    <groupId>com.alibaba.cloud</groupId>
    <artifactId>spring-cloud-starter-alibaba-seata</artifactId>
</dependency>
```

### 配置

```yaml
seata:
  # 应用 ID
  application-id: order-service
  # 事务分组
  tx-service-group: default_tx_group
  registry:
    type: nacos
    nacos:
      server-addr: localhost:8848
      group: SEATA_GROUP
  config:
    type: nacos
    nacos:
      server-addr: localhost:8848
      group: SEATA_GROUP
```

### 使用

```java
@Service
public class OrderService {

    @Autowired
    private OrderRepository orderRepository;

    @Autowired
    private AccountClient accountClient;

    @Autowired
    private StockClient stockClient;

    @GlobalTransactional(
        name = "create-order",
        rollbackFor = Exception.class
    )
    public void createOrder(Order order) {
        // 1. 创建订单
        orderRepository.save(order);

        // 2. 扣减账户余额（远程调用）
        accountClient.deduct(order.getUserId(), order.getAmount());

        // 3. 扣减库存（远程调用）
        stockClient.reduce(order.getProductId(), order.getQuantity());

        // 任何一步失败，所有操作都会回滚
    }
}
```

### 创建 undo_log 表

```sql
CREATE TABLE `undo_log` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `branch_id` bigint(20) NOT NULL,
  `xid` varchar(100) NOT NULL,
  `context` varchar(128) NOT NULL,
  `rollback_info` longblob NOT NULL,
  `log_status` int(11) NOT NULL,
  `log_created` datetime NOT NULL,
  `log_modified` datetime NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `ux_undo_log` (`xid`,`branch_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
```

## 4. TCC 模式

### 工作原理

- **Try** - 资源检查和预留
- **Confirm** - 确认执行
- **Cancel** - 取消执行，释放资源

### 示例

```java
@LocalTCC
public interface AccountTccAction {

    @TwoPhaseBusinessAction(
        name = "deduct",
        commitMethod = "confirm",
        rollbackMethod = "cancel"
    )
    boolean prepare(
        BusinessActionContext context,
        @BusinessActionContextParameter(paramName = "userId") Long userId,
        @BusinessActionContextParameter(paramName = "amount") BigDecimal amount
    );

    boolean confirm(BusinessActionContext context);

    boolean cancel(BusinessActionContext context);
}

@Service
public class AccountTccActionImpl implements AccountTccAction {

    @Override
    public boolean prepare(BusinessActionContext context, Long userId, BigDecimal amount) {
        // Try: 冻结金额
        accountService.freeze(userId, amount);
        return true;
    }

    @Override
    public boolean confirm(BusinessActionContext context) {
        // Confirm: 扣减冻结金额
        Long userId = (Long) context.getActionContext("userId");
        BigDecimal amount = (BigDecimal) context.getActionContext("amount");
        accountService.deduct(userId, amount);
        return true;
    }

    @Override
    public boolean cancel(BusinessActionContext context) {
        // Cancel: 释放冻结金额
        Long userId = (Long) context.getActionContext("userId");
        BigDecimal amount = (BigDecimal) context.getActionContext("amount");
        accountService.unfreeze(userId, amount);
        return true;
    }
}
```

## 5. SAGA 模式

适合长事务、多步骤的业务流程：

```java
// 定义状态机（JSON）
{
  "Name": "orderSaga",
  "Nodes": [
    {
      "Name": "CreateOrder",
      "Type": "ServiceTask",
      "ServiceName": "orderService",
      "ServiceMethod": "create",
      "CompensateMethod": "delete"
    },
    {
      "Name": "DeductAccount",
      "Type": "ServiceTask",
      "ServiceName": "accountService",
      "ServiceMethod": "deduct",
      "CompensateMethod": "add"
    },
    {
      "Name": "ReduceStock",
      "Type": "ServiceTask",
      "ServiceName": "stockService",
      "ServiceMethod": "reduce",
      "CompensateMethod": "add"
    }
  ]
}
```

## 6. XA 模式

基于数据库的 XA 协议：

```yaml
seata:
  data-source-proxy-mode: XA
```

```java
@GlobalTransactional
public void createOrder(Order order) {
    // 业务代码
}
```

## 7. 最佳实践

### 模式选择

- **AT 模式** - 首选，简单易用
- **TCC 模式** - 性能要求高，愿意手动编码
- **SAGA 模式** - 长事务，跨组织
- **XA 模式** - 强一致性要求

### 幂等性

```java
@GlobalTransactional
public void createOrder(Order order) {
    // 1. 检查幂等性
    if (orderRepository.existsByOrderNo(order.getOrderNo())) {
        return;
    }

    // 2. 执行业务
    // ...
}
```

### 异常处理

```java
@GlobalTransactional(
    rollbackFor = Exception.class,
    noRollbackFor = BusinessException.class
)
public void createOrder(Order order) {
    // 业务代码
}
```

## 8. 总结

| 模式 | 一致性   | 性能 | 复杂度 |
| ---- | -------- | ---- | ------ |
| AT   | 最终一致 | 中   | 低     |
| TCC  | 最终一致 | 高   | 高     |
| SAGA | 最终一致 | 高   | 中     |
| XA   | 强一致   | 低   | 低     |

---

**关键要点**：

- Seata 提供完整的分布式事务解决方案
- AT 模式适合大部分场景
- 注意幂等性和异常处理
- 生产环境建议使用集群部署

**下一步**：学习 [RocketMQ 消息队列](./rocketmq)
