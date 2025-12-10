---
sidebar_position: 4
title: "快速开始"
description: "快速搭建和运行 Apache Flink 应用"
---

# Flink 快速开始

## 环境准备

### 系统要求

- Java 8 或 Java 11（推荐 Java 11）
- Maven 3.x 或 Gradle
- Linux、macOS 或 Windows

### 安装 Flink

#### 下载 Flink

```bash
# 下载 Flink 1.17
wget https://archive.apache.org/dist/flink/flink-1.17.1/flink-1.17.1-bin-scala_2.12.tgz

# 解压
tar -xzf flink-1.17.1-bin-scala_2.12.tgz
cd flink-1.17.1
```

#### 启动本地集群

```bash
# 启动集群
./bin/start-cluster.sh

# 查看进程
jps
# 应该看到 StandaloneSessionClusterEntrypoint 和 TaskManagerRunner

# 访问 Web UI: http://localhost:8081
```

## 创建 Maven 项目

### pom.xml 配置

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.example</groupId>
    <artifactId>flink-quickstart</artifactId>
    <version>1.0-SNAPSHOT</version>

    <properties>
        <flink.version>1.17.1</flink.version>
        <java.version>11</java.version>
        <maven.compiler.source>${java.version}</maven.compiler.source>
        <maven.compiler.target>${java.version}</maven.compiler.target>
    </properties>

    <dependencies>
        <!-- Flink 核心依赖 -->
        <dependency>
            <groupId>org.apache.flink</groupId>
            <artifactId>flink-streaming-java</artifactId>
            <version>${flink.version}</version>
            <scope>provided</scope>
        </dependency>

        <!-- Flink 客户端 -->
        <dependency>
            <groupId>org.apache.flink</groupId>
            <artifactId>flink-clients</artifactId>
            <version>${flink.version}</version>
            <scope>provided</scope>
        </dependency>

        <!-- 本地运行需要 -->
        <dependency>
            <groupId>org.apache.flink</groupId>
            <artifactId>flink-runtime-web</artifactId>
            <version>${flink.version}</version>
            <scope>provided</scope>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-shade-plugin</artifactId>
                <version>3.4.1</version>
                <executions>
                    <execution>
                        <phase>package</phase>
                        <goals>
                            <goal>shade</goal>
                        </goals>
                        <configuration>
                            <transformers>
                                <transformer implementation="org.apache.maven.plugins.shade.resource.ManifestResourceTransformer">
                                    <mainClass>com.example.WordCount</mainClass>
                                </transformer>
                            </transformers>
                        </configuration>
                    </execution>
                </executions>
            </plugin>
        </plugins>
    </build>
</project>
```

## 第一个 Flink 程序

### WordCount 示例

```java
package com.example;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Collector;

public class WordCount {

    public static void main(String[] args) throws Exception {
        // 1. 创建执行环境
        final StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();

        // 2. 读取数据源
        DataStream<String> text = env.fromElements(
            "Hello World",
            "Hello Flink",
            "Hello Streaming"
        );

        // 3. 转换处理
        DataStream<Tuple2<String, Integer>> counts = text
            .flatMap(new Tokenizer())
            .keyBy(value -> value.f0)
            .sum(1);

        // 4. 输出结果
        counts.print();

        // 5. 执行作业
        env.execute("WordCount Example");
    }

    public static class Tokenizer
            implements FlatMapFunction<String, Tuple2<String, Integer>> {

        @Override
        public void flatMap(String value, Collector<Tuple2<String, Integer>> out) {
            String[] words = value.toLowerCase().split("\\s+");
            for (String word : words) {
                if (word.length() > 0) {
                    out.collect(new Tuple2<>(word, 1));
                }
            }
        }
    }
}
```

### 运行结果

```
(hello,1)
(world,1)
(hello,2)
(flink,1)
(hello,3)
(streaming,1)
```

## Socket 流处理示例

### 启动 Socket 服务

```bash
# 终端 1：启动 netcat
nc -lk 9999
```

### Socket WordCount

```java
public class SocketWordCount {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();

        // 从 Socket 读取数据
        DataStream<String> text = env.socketTextStream("localhost", 9999);

        DataStream<Tuple2<String, Integer>> counts = text
            .flatMap(new Tokenizer())
            .keyBy(value -> value.f0)
            .sum(1);

        counts.print();

        env.execute("Socket WordCount");
    }
}
```

## 提交到集群

### 打包

```bash
mvn clean package -DskipTests
```

### 提交作业

```bash
# 提交到本地集群
./bin/flink run target/flink-quickstart-1.0-SNAPSHOT.jar

# 指定并行度
./bin/flink run -p 4 target/flink-quickstart-1.0-SNAPSHOT.jar

# 后台运行
./bin/flink run -d target/flink-quickstart-1.0-SNAPSHOT.jar
```

### 查看和管理作业

```bash
# 列出运行中的作业
./bin/flink list

# 取消作业
./bin/flink cancel <jobId>

# 创建保存点
./bin/flink savepoint <jobId> /path/to/savepoints
```

## 停止集群

```bash
./bin/stop-cluster.sh
```

## 常见问题

### 内存不足

```bash
# 修改 conf/flink-conf.yaml
taskmanager.memory.process.size: 2048m
jobmanager.memory.process.size: 1024m
```

### 类找不到

确保依赖的 scope 设置正确：

- `provided`：集群已有的依赖
- `compile`：需要打包的依赖

## 下一步

- 💻 [DataStream API](./datastream-api.md) - 深入学习流处理 API
- 📊 [Table API & SQL](./table-sql.md) - 使用 SQL 进行数据处理
- 🎯 [核心概念](./core-concepts.md) - 理解 Flink 的核心概念
