---
sidebar_position: 13
title: "Flink CDC"
description: "Flink CDC 变更数据捕获详解"
---

# Flink CDC

## 什么是 Flink CDC？

Flink CDC（Change Data Capture）是基于数据库日志的变更数据捕获工具，可以实时将数据库变更同步到 Flink 进行处理。

### 核心优势

- **实时性**：毫秒级数据同步
- **全量+增量**：自动切换，无需手动干预
- **精确一次**：保证数据一致性
- **无侵入**：不影响源数据库性能

## 支持的数据源

| 数据源         | 版本支持      | 特性                    |
| -------------- | ------------- | ----------------------- |
| **MySQL**      | 5.6, 5.7, 8.0 | 全量+增量，支持所有类型 |
| **PostgreSQL** | 9.6+          | 逻辑复制                |
| **Oracle**     | 11, 12, 19    | LogMiner / XStream      |
| **MongoDB**    | 3.6+          | Change Streams          |
| **SQL Server** | 2012+         | CDC 功能                |
| **TiDB**       | 4.0+          | TiCDC 兼容              |

## 添加依赖

```xml
<!-- MySQL CDC -->
<dependency>
    <groupId>com.ververica</groupId>
    <artifactId>flink-connector-mysql-cdc</artifactId>
    <version>2.4.0</version>
</dependency>

<!-- PostgreSQL CDC -->
<dependency>
    <groupId>com.ververica</groupId>
    <artifactId>flink-connector-postgres-cdc</artifactId>
    <version>2.4.0</version>
</dependency>
```

## MySQL CDC 配置

### 数据库准备

```sql
-- 创建 CDC 用户
CREATE USER 'flink'@'%' IDENTIFIED BY 'password';

-- 授权
GRANT SELECT, RELOAD, SHOW DATABASES, REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'flink'@'%';
FLUSH PRIVILEGES;

-- 检查 binlog 配置
SHOW VARIABLES LIKE 'log_bin';           -- 应为 ON
SHOW VARIABLES LIKE 'binlog_format';     -- 应为 ROW
SHOW VARIABLES LIKE 'binlog_row_image';  -- 应为 FULL
```

### my.cnf 配置

```ini
[mysqld]
server-id = 1
log_bin = mysql-bin
binlog_format = ROW
binlog_row_image = FULL
expire_logs_days = 3
```

## SQL 方式使用

### 创建 CDC 源表

```sql
CREATE TABLE orders (
    order_id INT,
    user_id INT,
    product_name STRING,
    price DECIMAL(10, 2),
    order_status STRING,
    create_time TIMESTAMP(3),
    update_time TIMESTAMP(3),
    PRIMARY KEY (order_id) NOT ENFORCED
) WITH (
    'connector' = 'mysql-cdc',
    'hostname' = 'localhost',
    'port' = '3306',
    'username' = 'flink',
    'password' = 'password',
    'database-name' = 'mydb',
    'table-name' = 'orders',
    'server-time-zone' = 'Asia/Shanghai'
);
```

### 多表同步

```sql
-- 使用正则匹配多表
CREATE TABLE all_orders (
    db_name STRING METADATA FROM 'database_name' VIRTUAL,
    table_name STRING METADATA FROM 'table_name' VIRTUAL,
    order_id INT,
    user_id INT,
    amount DECIMAL(10, 2),
    PRIMARY KEY (order_id) NOT ENFORCED
) WITH (
    'connector' = 'mysql-cdc',
    'hostname' = 'localhost',
    'port' = '3306',
    'username' = 'flink',
    'password' = 'password',
    'database-name' = 'shop_.*',
    'table-name' = 'orders'
);
```

### 实时数据同步

```sql
-- 同步到 Kafka
INSERT INTO kafka_orders
SELECT * FROM orders;

-- 同步到 ClickHouse
INSERT INTO clickhouse_orders
SELECT order_id, user_id, price, order_status, create_time
FROM orders;

-- 同步到 Elasticsearch
INSERT INTO es_orders
SELECT order_id, user_id, product_name, price
FROM orders
WHERE order_status = 'COMPLETED';
```

## DataStream API 使用

### 基础用法

```java
import com.ververica.cdc.connectors.mysql.source.MySqlSource;
import com.ververica.cdc.debezium.JsonDebeziumDeserializationSchema;

MySqlSource<String> source = MySqlSource.<String>builder()
    .hostname("localhost")
    .port(3306)
    .databaseList("mydb")
    .tableList("mydb.orders")
    .username("flink")
    .password("password")
    .deserializer(new JsonDebeziumDeserializationSchema())
    .build();

DataStreamSource<String> stream = env.fromSource(
    source,
    WatermarkStrategy.noWatermarks(),
    "MySQL CDC Source"
);

stream.print();
env.execute("MySQL CDC Job");
```

### 自定义反序列化

```java
public class OrderDeserializer
        implements DebeziumDeserializationSchema<Order> {

    @Override
    public void deserialize(SourceRecord record, Collector<Order> out) {
        Struct value = (Struct) record.value();
        String op = value.getString("op"); // c=create, u=update, d=delete

        Struct after = value.getStruct("after");
        if (after != null) {
            Order order = new Order();
            order.setOrderId(after.getInt32("order_id"));
            order.setUserId(after.getInt32("user_id"));
            order.setAmount(after.getFloat64("amount"));
            order.setOperation(op);
            out.collect(order);
        }
    }

    @Override
    public TypeInformation<Order> getProducedType() {
        return TypeInformation.of(Order.class);
    }
}
```

## 高级配置

### 启动模式

```java
MySqlSource.<String>builder()
    // 初始读取：全量 + 增量
    .startupOptions(StartupOptions.initial())

    // 从最新位置开始（跳过全量）
    .startupOptions(StartupOptions.latest())

    // 从指定时间戳开始
    .startupOptions(StartupOptions.timestamp(1678886400000L))

    // 从指定 binlog 位置开始
    .startupOptions(StartupOptions.specificOffset("mysql-bin.000003", 4L))
    .build();
```

### 分片并行读取

```java
MySqlSource.<String>builder()
    .hostname("localhost")
    .port(3306)
    .databaseList("mydb")
    .tableList("mydb.orders")
    .username("flink")
    .password("password")
    // 并行快照读取
    .splitSize(8096)            // 每个分片的记录数
    .fetchSize(1024)            // 每次拉取的记录数
    .connectTimeout(Duration.ofSeconds(30))
    .deserializer(new JsonDebeziumDeserializationSchema())
    .build();
```

### 心跳配置

```sql
CREATE TABLE orders (...) WITH (
    'connector' = 'mysql-cdc',
    ...
    'debezium.heartbeat.interval.ms' = '60000',
    'debezium.snapshot.mode' = 'initial'
);
```

## 实战案例

### 实时数据同步到数据仓库

```sql
-- 源表
CREATE TABLE source_orders (
    id INT,
    user_id INT,
    amount DECIMAL(10,2),
    status STRING,
    created_at TIMESTAMP(3),
    updated_at TIMESTAMP(3),
    PRIMARY KEY (id) NOT ENFORCED
) WITH (
    'connector' = 'mysql-cdc',
    'hostname' = 'mysql-prod',
    'database-name' = 'ecommerce',
    'table-name' = 'orders',
    'username' = 'cdc_user',
    'password' = 'password'
);

-- 目标表（ClickHouse）
CREATE TABLE sink_orders (
    id INT,
    user_id INT,
    amount DECIMAL(10,2),
    status STRING,
    created_at TIMESTAMP(3),
    updated_at TIMESTAMP(3),
    PRIMARY KEY (id) NOT ENFORCED
) WITH (
    'connector' = 'jdbc',
    'url' = 'jdbc:clickhouse://clickhouse:8123/analytics',
    'table-name' = 'orders',
    'driver' = 'com.clickhouse.jdbc.ClickHouseDriver'
);

-- 实时同步
INSERT INTO sink_orders SELECT * FROM source_orders;
```

### 实时宽表构建

```sql
-- 订单主表
CREATE TABLE orders (...) WITH ('connector' = 'mysql-cdc', ...);

-- 用户维表
CREATE TABLE users (...) WITH ('connector' = 'jdbc', ...);

-- 商品维表
CREATE TABLE products (...) WITH ('connector' = 'jdbc', ...);

-- 构建宽表
INSERT INTO order_wide_table
SELECT
    o.order_id,
    o.order_time,
    o.amount,
    u.user_name,
    u.user_level,
    p.product_name,
    p.category
FROM orders o
LEFT JOIN users FOR SYSTEM_TIME AS OF o.proc_time AS u
    ON o.user_id = u.user_id
LEFT JOIN products FOR SYSTEM_TIME AS OF o.proc_time AS p
    ON o.product_id = p.product_id;
```

## 最佳实践

### 性能优化

```java
// 1. 合理设置并行度
env.setParallelism(4);

// 2. 配置 RocksDB 状态后端
env.setStateBackend(new EmbeddedRocksDBStateBackend());

// 3. 增量检查点
env.getCheckpointConfig().enableUnalignedCheckpoints();
```

### 注意事项

| 问题       | 解决方案                   |
| ---------- | -------------------------- |
| 全量阶段慢 | 增加 splitSize，提高并行度 |
| 内存不足   | 使用 RocksDB 后端          |
| 数据延迟   | 检查源库负载，优化网络     |
| 断点续传   | 启用检查点，配置状态后端   |

## 下一步

- 🔌 [连接器](/docs/flink/connectors) - 更多连接器配置
- 📊 [Table API & SQL](/docs/flink/table-sql) - SQL 详解
- 🚀 [监控与运维](/docs/flink/monitoring) - 生产监控
