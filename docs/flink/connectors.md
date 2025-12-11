---
sidebar_position: 12
title: "连接器"
description: "Flink 数据源和接收器连接器详解"
---

# Flink 连接器

## 概述

Flink 连接器用于与外部系统进行数据交互，包括数据源（Source）和数据接收器（Sink）。

## Kafka 连接器

### 添加依赖

```xml
<dependency>
    <groupId>org.apache.flink</groupId>
    <artifactId>flink-connector-kafka</artifactId>
    <version>${flink.version}</version>
</dependency>
```

### Kafka Source

```java
KafkaSource<String> source = KafkaSource.<String>builder()
    .setBootstrapServers("localhost:9092")
    .setTopics("input-topic")
    .setGroupId("my-group")
    .setStartingOffsets(OffsetsInitializer.earliest())
    .setValueOnlyDeserializer(new SimpleStringSchema())
    .build();

DataStream<String> stream = env.fromSource(
    source,
    WatermarkStrategy.noWatermarks(),
    "Kafka Source"
);
```

### Kafka Sink

```java
KafkaSink<String> sink = KafkaSink.<String>builder()
    .setBootstrapServers("localhost:9092")
    .setRecordSerializer(KafkaRecordSerializationSchema.builder()
        .setTopic("output-topic")
        .setValueSerializationSchema(new SimpleStringSchema())
        .build()
    )
    .setDeliveryGuarantee(DeliveryGuarantee.AT_LEAST_ONCE)
    .build();

stream.sinkTo(sink);
```

### Kafka SQL 连接器

```sql
CREATE TABLE kafka_source (
    id STRING,
    name STRING,
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

CREATE TABLE kafka_sink (
    id STRING,
    result STRING,
    ts TIMESTAMP(3)
) WITH (
    'connector' = 'kafka',
    'topic' = 'output-topic',
    'properties.bootstrap.servers' = 'localhost:9092',
    'format' = 'json'
);
```

## JDBC 连接器

### 添加依赖

```xml
<dependency>
    <groupId>org.apache.flink</groupId>
    <artifactId>flink-connector-jdbc</artifactId>
    <version>3.1.0-1.17</version>
</dependency>
<dependency>
    <groupId>mysql</groupId>
    <artifactId>mysql-connector-java</artifactId>
    <version>8.0.28</version>
</dependency>
```

### JDBC Source（Table API）

```sql
CREATE TABLE mysql_source (
    id INT,
    name STRING,
    age INT,
    PRIMARY KEY (id) NOT ENFORCED
) WITH (
    'connector' = 'jdbc',
    'url' = 'jdbc:mysql://localhost:3306/mydb',
    'table-name' = 'users',
    'username' = 'root',
    'password' = 'password'
);
```

### JDBC Sink

```java
stream.addSink(JdbcSink.sink(
    "INSERT INTO users (id, name, age) VALUES (?, ?, ?)",
    (statement, user) -> {
        statement.setInt(1, user.getId());
        statement.setString(2, user.getName());
        statement.setInt(3, user.getAge());
    },
    JdbcExecutionOptions.builder()
        .withBatchSize(1000)
        .withBatchIntervalMs(200)
        .withMaxRetries(5)
        .build(),
    new JdbcConnectionOptions.JdbcConnectionOptionsBuilder()
        .withUrl("jdbc:mysql://localhost:3306/mydb")
        .withDriverName("com.mysql.cj.jdbc.Driver")
        .withUsername("root")
        .withPassword("password")
        .build()
));
```

### JDBC Lookup Join

```sql
-- 维表定义
CREATE TABLE products (
    id INT,
    name STRING,
    price DECIMAL(10, 2),
    PRIMARY KEY (id) NOT ENFORCED
) WITH (
    'connector' = 'jdbc',
    'url' = 'jdbc:mysql://localhost:3306/mydb',
    'table-name' = 'products',
    'lookup.cache.max-rows' = '5000',
    'lookup.cache.ttl' = '10min'
);

-- Lookup Join
SELECT o.order_id, p.name, p.price
FROM orders AS o
JOIN products FOR SYSTEM_TIME AS OF o.proc_time AS p
  ON o.product_id = p.id;
```

## Elasticsearch 连接器

### 添加依赖

```xml
<dependency>
    <groupId>org.apache.flink</groupId>
    <artifactId>flink-connector-elasticsearch7</artifactId>
    <version>${flink.version}</version>
</dependency>
```

### Elasticsearch Sink

```java
ElasticsearchSink<Event> esSink = new Elasticsearch7SinkBuilder<Event>()
    .setHosts(new HttpHost("localhost", 9200, "http"))
    .setEmitter((element, context, indexer) -> {
        indexer.add(Requests.indexRequest()
            .index("events")
            .id(element.getId())
            .source(Map.of(
                "id", element.getId(),
                "name", element.getName(),
                "timestamp", element.getTimestamp()
            ))
        );
    })
    .setBulkFlushMaxActions(1000)
    .build();

stream.sinkTo(esSink);
```

### Elasticsearch SQL

```sql
CREATE TABLE es_sink (
    id STRING,
    name STRING,
    ts TIMESTAMP(3),
    PRIMARY KEY (id) NOT ENFORCED
) WITH (
    'connector' = 'elasticsearch-7',
    'hosts' = 'http://localhost:9200',
    'index' = 'events'
);
```

## 文件系统连接器

### 文件 Source

```java
FileSource<String> source = FileSource
    .forRecordStreamFormat(
        new TextLineInputFormat(),
        new Path("hdfs:///data/input")
    )
    .monitorContinuously(Duration.ofSeconds(10))
    .build();

DataStream<String> stream = env.fromSource(
    source,
    WatermarkStrategy.noWatermarks(),
    "File Source"
);
```

### 文件 Sink（流式写入）

```java
FileSink<String> sink = FileSink
    .forRowFormat(
        new Path("hdfs:///data/output"),
        new SimpleStringEncoder<String>("UTF-8")
    )
    .withRollingPolicy(
        DefaultRollingPolicy.builder()
            .withRolloverInterval(Duration.ofMinutes(15))
            .withInactivityInterval(Duration.ofMinutes(5))
            .withMaxPartSize(MemorySize.ofMebiBytes(1024))
            .build()
    )
    .withBucketAssigner(new DateTimeBucketAssigner<>("yyyy-MM-dd--HH"))
    .build();

stream.sinkTo(sink);
```

### 文件系统 SQL

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

CREATE TABLE file_sink (
    id STRING,
    name STRING,
    dt STRING
) PARTITIONED BY (dt) WITH (
    'connector' = 'filesystem',
    'path' = 'hdfs:///data/output',
    'format' = 'parquet',
    'sink.partition-commit.policy.kind' = 'success-file'
);
```

## Redis 连接器

### 自定义 Redis Sink

```java
public class RedisSink extends RichSinkFunction<Event> {
    private transient Jedis jedis;

    @Override
    public void open(Configuration parameters) {
        jedis = new Jedis("localhost", 6379);
    }

    @Override
    public void invoke(Event event, Context context) {
        jedis.hset("events", event.getId(), event.toJson());
    }

    @Override
    public void close() {
        if (jedis != null) {
            jedis.close();
        }
    }
}
```

## 连接器配置汇总

| 连接器        | 依赖                          | 特点                 |
| ------------- | ----------------------------- | -------------------- |
| Kafka         | flink-connector-kafka         | 最常用，支持精确一次 |
| JDBC          | flink-connector-jdbc          | 支持各种关系型数据库 |
| Elasticsearch | flink-connector-elasticsearch | 全文搜索、日志分析   |
| Filesystem    | flink-connector-files         | HDFS/S3/本地文件     |
| Pulsar        | flink-connector-pulsar        | 新一代消息系统       |
| HBase         | flink-connector-hbase         | NoSQL 数据库         |

## 下一步

- 📊 [Table API & SQL](/docs/flink/table-sql) - SQL 连接器使用
- 🚀 [部署与运维](/docs/flink/deployment) - 生产部署
- 📋 [最佳实践](/docs/flink/best-practices) - 开发规范
