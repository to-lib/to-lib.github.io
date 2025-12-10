---
sidebar_position: 6
title: "Table API 与 SQL"
description: "Flink Table API 和 SQL 声明式数据处理"
---

# Table API 与 SQL

## 概述

Flink Table API 和 SQL 提供了统一的声明式 API，可以用类似关系型数据库的方式处理流和批数据。

## 环境配置

### Maven 依赖

```xml
<dependency>
    <groupId>org.apache.flink</groupId>
    <artifactId>flink-table-api-java-bridge</artifactId>
    <version>${flink.version}</version>
</dependency>
<dependency>
    <groupId>org.apache.flink</groupId>
    <artifactId>flink-table-planner-loader</artifactId>
    <version>${flink.version}</version>
</dependency>
```

### 创建表环境

```java
// 流处理表环境
StreamExecutionEnvironment env =
    StreamExecutionEnvironment.getExecutionEnvironment();
StreamTableEnvironment tableEnv = StreamTableEnvironment.create(env);

// 纯 Table API 环境
EnvironmentSettings settings = EnvironmentSettings
    .newInstance()
    .inStreamingMode()
    .build();
TableEnvironment tableEnv = TableEnvironment.create(settings);
```

## 创建表

### 从 DataStream 创建

```java
// 从 POJO 流创建
DataStream<User> userStream = ...;
Table userTable = tableEnv.fromDataStream(userStream);

// 指定列名
Table table = tableEnv.fromDataStream(
    userStream,
    $("id"), $("name"), $("age"), $("proctime").proctime()
);
```

### 使用 DDL 创建

```java
tableEnv.executeSql(
    "CREATE TABLE orders (" +
    "   order_id STRING," +
    "   user_id STRING," +
    "   amount DECIMAL(10, 2)," +
    "   order_time TIMESTAMP(3)," +
    "   WATERMARK FOR order_time AS order_time - INTERVAL '5' SECOND" +
    ") WITH (" +
    "   'connector' = 'kafka'," +
    "   'topic' = 'orders'," +
    "   'properties.bootstrap.servers' = 'localhost:9092'," +
    "   'format' = 'json'" +
    ")"
);
```

## Table API 操作

### 基本查询

```java
Table orders = tableEnv.from("orders");

// 选择列
Table result = orders.select($("order_id"), $("amount"));

// 过滤
Table filtered = orders.filter($("amount").isGreater(100));

// 别名
Table aliased = orders
    .select($("order_id").as("id"), $("amount").as("total"));
```

### 聚合操作

```java
// 简单聚合
Table totals = orders
    .groupBy($("user_id"))
    .select($("user_id"), $("amount").sum().as("total_amount"));

// 窗口聚合
Table windowedResult = orders
    .window(Tumble.over(lit(1).hours()).on($("order_time")).as("w"))
    .groupBy($("user_id"), $("w"))
    .select(
        $("user_id"),
        $("w").start().as("window_start"),
        $("w").end().as("window_end"),
        $("amount").sum().as("total_amount")
    );
```

### 表连接

```java
Table orders = tableEnv.from("orders");
Table users = tableEnv.from("users");

// 内连接
Table joined = orders
    .join(users)
    .where($("orders.user_id").isEqual($("users.id")))
    .select($("order_id"), $("users.name"), $("amount"));

// 左外连接
Table leftJoined = orders
    .leftOuterJoin(users, $("orders.user_id").isEqual($("users.id")))
    .select($("order_id"), $("users.name"), $("amount"));
```

## Flink SQL

### 基本查询

```java
// 执行查询
Table result = tableEnv.sqlQuery(
    "SELECT user_id, SUM(amount) as total " +
    "FROM orders " +
    "GROUP BY user_id"
);

// 执行 DDL/DML
tableEnv.executeSql("INSERT INTO output_table SELECT * FROM orders");
```

### 窗口查询

```sql
-- 滚动窗口
SELECT
    user_id,
    TUMBLE_START(order_time, INTERVAL '1' HOUR) AS window_start,
    TUMBLE_END(order_time, INTERVAL '1' HOUR) AS window_end,
    SUM(amount) AS total_amount
FROM orders
GROUP BY user_id, TUMBLE(order_time, INTERVAL '1' HOUR);

-- 滑动窗口
SELECT
    user_id,
    HOP_START(order_time, INTERVAL '5' MINUTE, INTERVAL '1' HOUR) AS window_start,
    SUM(amount) AS total_amount
FROM orders
GROUP BY user_id, HOP(order_time, INTERVAL '5' MINUTE, INTERVAL '1' HOUR);

-- 会话窗口
SELECT
    user_id,
    SESSION_START(order_time, INTERVAL '30' MINUTE) AS session_start,
    COUNT(*) AS order_count
FROM orders
GROUP BY user_id, SESSION(order_time, INTERVAL '30' MINUTE);
```

### Top-N 查询

```sql
SELECT * FROM (
    SELECT
        user_id,
        total_amount,
        ROW_NUMBER() OVER (ORDER BY total_amount DESC) AS rank
    FROM (
        SELECT user_id, SUM(amount) AS total_amount
        FROM orders
        GROUP BY user_id
    )
) WHERE rank <= 10;
```

### Join 操作

```sql
-- 常规 Join
SELECT o.order_id, u.name, o.amount
FROM orders o
JOIN users u ON o.user_id = u.id;

-- 时间窗口 Join
SELECT o.order_id, p.payment_id, o.amount
FROM orders o, payments p
WHERE o.order_id = p.order_id
  AND p.payment_time BETWEEN o.order_time AND o.order_time + INTERVAL '1' HOUR;

-- Lookup Join（维表关联）
SELECT o.order_id, p.product_name, o.amount
FROM orders o
JOIN products FOR SYSTEM_TIME AS OF o.order_time AS p
  ON o.product_id = p.id;
```

## 内置函数

### 字符串函数

```sql
SELECT
    UPPER(name) AS upper_name,
    LOWER(name) AS lower_name,
    CONCAT(first_name, ' ', last_name) AS full_name,
    SUBSTRING(name, 1, 5) AS short_name
FROM users;
```

### 时间函数

```sql
SELECT
    order_time,
    DATE_FORMAT(order_time, 'yyyy-MM-dd') AS date_str,
    YEAR(order_time) AS year,
    MONTH(order_time) AS month,
    CURRENT_TIMESTAMP AS now
FROM orders;
```

### 聚合函数

```sql
SELECT
    user_id,
    COUNT(*) AS order_count,
    SUM(amount) AS total_amount,
    AVG(amount) AS avg_amount,
    MIN(amount) AS min_amount,
    MAX(amount) AS max_amount
FROM orders
GROUP BY user_id;
```

## 连接器

### Kafka 连接器

```sql
CREATE TABLE kafka_source (
    id STRING,
    data STRING,
    ts TIMESTAMP(3),
    WATERMARK FOR ts AS ts - INTERVAL '5' SECOND
) WITH (
    'connector' = 'kafka',
    'topic' = 'input-topic',
    'properties.bootstrap.servers' = 'localhost:9092',
    'properties.group.id' = 'my-group',
    'scan.startup.mode' = 'earliest-offset',
    'format' = 'json'
);
```

### JDBC 连接器

```sql
CREATE TABLE jdbc_table (
    id INT,
    name STRING,
    PRIMARY KEY (id) NOT ENFORCED
) WITH (
    'connector' = 'jdbc',
    'url' = 'jdbc:mysql://localhost:3306/mydb',
    'table-name' = 'users',
    'username' = 'root',
    'password' = 'password'
);
```

### 文件系统连接器

```sql
CREATE TABLE file_source (
    id STRING,
    name STRING,
    dt STRING
) PARTITIONED BY (dt) WITH (
    'connector' = 'filesystem',
    'path' = 'hdfs:///data/input',
    'format' = 'parquet'
);
```

## 结果输出

### 转换为 DataStream

```java
Table resultTable = tableEnv.sqlQuery("SELECT * FROM orders");

// 转换为 DataStream
DataStream<Row> resultStream = tableEnv.toDataStream(resultTable);

// 转换为 changelog 流（有更新操作时）
DataStream<Row> changelogStream = tableEnv.toChangelogStream(resultTable);
```

### 输出到 Sink

```java
tableEnv.executeSql(
    "INSERT INTO output_table " +
    "SELECT user_id, SUM(amount) FROM orders GROUP BY user_id"
);
```

## 下一步

- 💻 [DataStream API](./datastream-api.md) - 底层流处理 API
- 🔧 [状态管理](./state-management.md) - 有状态计算
- 🚀 [性能优化](./performance.md) - 性能调优指南
