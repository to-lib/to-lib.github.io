---
sidebar_position: 15
title: "安全与 ACL"
description: "RocketMQ 访问控制（ACL）与生产安全加固指南"
---

# RocketMQ 安全与 ACL

RocketMQ 开源版的安全能力通常由以下几部分组成：

- **访问控制（ACL）**：限制谁能访问 Broker、能对哪些 Topic/ConsumerGroup 做什么操作。
- **网络隔离**：把 NameServer/Broker 放在内网，通过安全组/防火墙限制来源。
- **最小权限**：区分生产者/消费者/管理员账号，避免共享 root 权限。
- **审计与可观测性**：留存关键操作与异常日志，配合告警。

本文以 RocketMQ 4.x/5.x 开源版的常见实践为主（不同版本配置项可能略有差异）。

## 安全基线（上线前必做）

- **NameServer/Broker 仅内网可达**
- **禁用公网 10911/10909/9876 直连**（只允许跳板机/内网应用访问）
- **开启 ACL 并使用独立账号**（不要把 `admin` 账号硬编码到业务代码）
- **Dashboard/Exporter 等运维组件单独鉴权**（至少内网 + 反向代理鉴权）
- **消息体脱敏与加密**（对敏感字段做脱敏/加密，避免在消息中明文传递密钥/证件号）

## RocketMQ ACL（访问控制）

RocketMQ 的 ACL 机制通过在 Broker 侧开启鉴权，并配置 `plain_acl.yml`（或同类 ACL 配置文件）来实现。

### 1. Broker 开启 ACL

在 `broker.conf` 中开启 ACL（你仓库其他文档中也使用了该配置项）：

```properties
# broker.conf
aclEnable=true
```

启动 Broker 时确保加载该配置文件：

```bash
nohup sh bin/mqbroker -c conf/broker.conf &
```

### 2. 配置 `plain_acl.yml`

`plain_acl.yml` 默认放在 RocketMQ 的配置目录（不同版本/发行包路径可能不同；以官方包的 `conf` 目录为例）。

示例：

```yaml
# conf/plain_acl.yml
accounts:
  # 管理员账号（仅用于运维）
  - accessKey: admin
    secretKey: change_me_admin_secret
    admin: true
    whiteRemoteAddress:
      - 10.0.*
      - 192.168.*

  # 生产者账号（只允许发）
  - accessKey: order_producer
    secretKey: change_me_producer_secret
    defaultTopicPerm: PUB
    defaultGroupPerm: DENY
    whiteRemoteAddress:
      - 10.0.*
    topicPerms:
      - order_topic=PUB

  # 消费者账号（只允许订阅）
  - accessKey: order_consumer
    secretKey: change_me_consumer_secret
    defaultTopicPerm: DENY
    defaultGroupPerm: SUB
    whiteRemoteAddress:
      - 10.0.*
    groupPerms:
      - order_consumer_group=SUB

# 全局白名单（可选）
globalWhiteRemoteAddresses:
  - 127.0.0.1
```

说明：

- **`accessKey/secretKey`**：相当于用户名/密码。
- **`admin: true`**：运维账号，拥有管理权限（谨慎授予）。
- **`defaultTopicPerm/defaultGroupPerm`**：默认权限（常见取值：`PUB`/`SUB`/`DENY`）。
- **`topicPerms/groupPerms`**：按 Topic/ConsumerGroup 精细授权。
- **`whiteRemoteAddress`**：限制来源 IP 段，建议配合内网隔离使用。

### 3. 客户端接入（Java SDK）

生产者/消费者在构造时通过 `AclClientRPCHook` 注入凭证：

```java
// 生产者
DefaultMQProducer producer = new DefaultMQProducer(
    "ProducerGroup",
    new AclClientRPCHook(new SessionCredentials("order_producer", "change_me_producer_secret"))
);
producer.setNamesrvAddr("10.0.0.10:9876");
producer.start();

// 消费者
DefaultMQPushConsumer consumer = new DefaultMQPushConsumer(
    "order_consumer_group",
    new AclClientRPCHook(new SessionCredentials("order_consumer", "change_me_consumer_secret"))
);
consumer.setNamesrvAddr("10.0.0.10:9876");
consumer.subscribe("order_topic", "*");
consumer.start();
```

建议：

- **凭证不要写死在代码里**，用配置中心/环境变量/密钥管理系统注入。
- **按应用/业务域拆分账号**，出现泄漏时便于快速止损。

## 运维组件的安全（Dashboard/Exporter）

### RocketMQ Dashboard

常见风险：

- Dashboard 可执行运维操作（创建 Topic、重置 Offset、查询消息等）。

建议：

- **仅内网访问**（安全组 + 防火墙）。
- 通过 **Nginx/Ingress 增加统一鉴权**（BasicAuth/OIDC/企业 SSO）。
- 若启用了 ACL，Dashboard 侧也需要配置对应的 AK/SK，并限制其来源 IP。

### Prometheus Exporter

你的监控文档中已提供 Exporter 配置示例。启用 ACL 后：

- Exporter 需要配置 `enableACL/accessKey/secretKey`。
- 建议使用**只读/最小权限账号**（不要复用管理员账号）。

## 最佳实践清单

- **账号与权限**
  - 生产者账号仅 `PUB`
  - 消费者账号仅 `SUB`
  - 运维账号单独管理，减少使用频率
- **密钥治理**
  - 定期轮换 `secretKey`
  - 泄漏应急：立即吊销/替换账号，并通过 `whiteRemoteAddress` 临时收敛来源
- **Topic/Group 规范化**
  - Topic/Group 命名带业务域，便于授权与审计
- **网络层**
  - 仅允许应用所在网段访问 Broker
  - 禁止跨网段直连（尤其是公网）

## 常见问题

### 1) 开启 ACL 后报鉴权失败/无权限

排查方向：

- **客户端是否正确注入 `AclClientRPCHook`**
- **`plain_acl.yml` 是否被 Broker 正确加载**（看 Broker 日志是否有 ACL 相关加载信息）
- **权限是否授对了 Topic/Group**（尤其是 `defaultTopicPerm/defaultGroupPerm`）
- **来源 IP 是否命中 `whiteRemoteAddress`**

### 2) ACL 开启后 Dashboard 无法使用

通常是：

- Dashboard 没配置 AK/SK
- Dashboard 所在机器 IP 不在白名单

## 下一步

- 📊 [监控与运维](/docs/rocketmq/monitoring)
- 🛠️ [排障手册](/docs/rocketmq/troubleshooting)
- ✅ [最佳实践](/docs/rocketmq/best-practices)
