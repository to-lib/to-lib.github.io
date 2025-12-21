---
sidebar_position: 25
title: 系统设计面试题
slug: /interview/system-design-interview-questions
---

# 🏗️ 系统设计面试题

> [!TIP]
> 系统设计面试考察的是你在面对复杂、大规模系统时的架构能力。重点在于**权衡(Trade-off)**、**扩展性**和**可行性**。

## 🎯 基础概念

### 1. 什么是 CAP 定理？

**答案**:

CAP 定理指出，在分布式系统中，无法同时满足以下三点，最多只能满足其中两点：

- **Consistency (一致性)**: 所有节点在同一时间读到相同的数据。
- **Availability (可用性)**: 每次请求都能得到响应（不保证数据最新）。
- **Partition Tolerance (分区容错性)**: 系统在网络分区（节点间通信失败）时仍能继续运行。

**权衡**:

- **CP (一致性 + 分区容错)**: 牺牲可用性。如 HBase, Redis (默认)。适用于对数据一致性要求高的场景（如金融）。
- **AP (可用性 + 分区容错)**: 牺牲一致性（保证最终一致性）。如 Cassandra, DynamoDB。适用于追求高可用的场景（如社交网络）。

### 2. 负载均衡有哪些算法？

**答案**:

1.  **轮询 (Round Robin)**: 依次分配请求。
2.  **加权轮询 (Weighted Round Robin)**: 根据服务器性能加权。
3.  **最少连接 (Least Connections)**: 分配给当前连接数最少的服务器。
4.  **IP Hash**: 根据 IP 哈希值分配，确保同一 IP 访问同一服务器（会话保持）。
5.  **一致性哈希 (Consistent Hashing)**: 减少节点增减时的缓存抖动，常用于分布式缓存。

### 3. 常见的缓存策略有哪些？

**答案**:

- **Cache Aside (旁路缓存)**: 应用先读缓存，不命中读库并回填；写时先更新库，再删缓存。最常用。
- **Read/Write Through**: 应用只与缓存交互，缓存负责与数据库同步。
- **Write Behind (异步写回)**: 应用只写缓存，缓存异步批量同步数据库。性能最高，但有丢失数据风险。

## 🏛️ 经典设计题

### 4. 设计一个 URL 短链生成器 (TinyURL)

**核心要点**:

1.  **API 设计**:
    - `createShortURL(originalURL) -> shortURL`
    - `getOriginalURL(shortURL) -> originalURL` (HTTP 301/302 重定向)
2.  **唯一 ID 生成**:
    - **数据库自增 ID**: 简单，但需考虑分库分表。
    - **分布式 ID (Snowflake)**: 高性能，有序。
    - **Redis 自增**: 性能极高。
3.  **Base62 编码**: 将 ID 转为 [a-z, A-Z, 0-9] 字符串 (62 进制)，缩短长度。
4.  **存储**:
    - K-V 数据库 (Redis/DynamoDB) 适合，读写快。
    - RDBMS (MySQL) 加上索引也可以支撑很大规模。

### 5. 设计 Twitter (Feeds 流系统)

**核心模式**:

1.  **Pull 模型 (拉模式)**:
    - 用户查看 Feeds 时，查询关注列表，拉取他们最新的推文进行聚合排序。
    - 优点：实现简单，节约存储。
    - 缺点：读取延迟高。
2.  **Push 模型 (推模式/写扩散)**:
    - 大 V 发推文时，直接写入所有粉丝的 Feeds 列表 (Inbox)。
    - 优点：读取快（O(1)）。
    - 缺点：写入成本大，大 V 粉丝多时会有“惊群效应”。
3.  **混合模式**:
    - 普通用户使用 Push。
    - 大 V (粉丝 > 100w) 使用 Pull。

## 💾 数据库与存储

### 6. SQL vs NoSQL 如何选择？

**答案**:

- **SQL (MySQL, PostgreSQL)**:
  - 结构化数据，关系复杂。
  - 需要 ACID 事务支持。
  - 数据量适中，或者可以通过分库分表解决。
- **NoSQL (MongoDB, Redis, Cassandra)**:
  - 非结构化/半结构化数据。
  - 海量数据，高吞吐量写入。
  - 灵活性高，Schema-less。
  - 对一致性要求不苛刻 (最终一致性)。

### 7. 数据复制与分片

- **主从复制 (Replication)**: 提高读性能，数据备份。
- **分片 (Sharding)**: 水平切分，解决单机存储和写入瓶颈。
  - 垂直分片：按业务/列拆分。
  - 水平分片：按 ID 范围/哈希拆分。

## 🔒 认证与安全

### 8. Session vs JWT (JSON Web Token)

**答案**:

- **Session**:
  - 服务端存储状态。
  - 客户端存 SessionID (Cookie)。
  - 优点：安全，服务端可控（可强制下线）。
  - 缺点：分布式环境需要共享 Session (Redis)。
- **JWT**:
  - 无状态，信息自包含在 Token 中。
  - 优点：扩展性好，无需查库。
  - 缺点：Token 一旦签发无法撤销（需配合黑名单），体积较大。

---

**面试技巧**:

- **Clarify Constraints**: 即使是开放性问题，也要先问清楚 QPS、数据量、延迟要求等约束条件。
- **High Level Design**: 先画出整体架构图，再深入细节。
- **Address Bottlenecks**: 主动识别单点故障和性能瓶颈，并提出解决方案。
