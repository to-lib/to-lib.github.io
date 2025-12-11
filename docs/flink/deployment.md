---
sidebar_position: 13
title: "部署与运维"
description: "Flink 集群部署与运维指南"
---

# Flink 部署与运维

## 部署模式

### Standalone 模式

独立集群部署，适合开发测试：

```bash
# 启动集群
./bin/start-cluster.sh

# 提交作业
./bin/flink run myJob.jar

# 停止集群
./bin/stop-cluster.sh
```

**配置 flink-conf.yaml**：

```yaml
jobmanager.rpc.address: master-node
jobmanager.rpc.port: 6123
jobmanager.memory.process.size: 1600m
taskmanager.memory.process.size: 4096m
taskmanager.numberOfTaskSlots: 4
parallelism.default: 2
```

### YARN 模式

在 Hadoop YARN 上运行 Flink：

```bash
# Session 模式：预启动集群
./bin/yarn-session.sh -n 4 -jm 1024m -tm 4096m -s 2

# Per-Job 模式：每个作业独立集群
./bin/flink run -m yarn-cluster -yjm 1024m -ytm 4096m myJob.jar

# Application 模式（推荐）
./bin/flink run-application -t yarn-application myJob.jar
```

### Kubernetes 模式

使用 Kubernetes 部署 Flink：

```yaml
# flink-configuration-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: flink-config
data:
  flink-conf.yaml: |
    jobmanager.rpc.address: flink-jobmanager
    taskmanager.numberOfTaskSlots: 2
    blob.server.port: 6124
    jobmanager.rpc.port: 6123
```

```yaml
# flink-jobmanager-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flink-jobmanager
spec:
  replicas: 1
  selector:
    matchLabels:
      app: flink
      component: jobmanager
  template:
    spec:
      containers:
        - name: jobmanager
          image: flink:1.17
          args: ["jobmanager"]
          ports:
            - containerPort: 6123
            - containerPort: 8081
          env:
            - name: FLINK_PROPERTIES
              value: |
                jobmanager.rpc.address: flink-jobmanager
```

**Kubernetes 部署命令**：

```bash
# Native Kubernetes
./bin/flink run-application \
    --target kubernetes-application \
    -Dkubernetes.cluster-id=my-flink-cluster \
    -Dkubernetes.container.image=my-flink-image:latest \
    local:///opt/flink/usrlib/my-flink-job.jar
```

## 高可用配置

### ZooKeeper HA

```yaml
# flink-conf.yaml
high-availability: zookeeper
high-availability.storageDir: hdfs:///flink/ha
high-availability.zookeeper.quorum: zk1:2181,zk2:2181,zk3:2181
high-availability.zookeeper.path.root: /flink
high-availability.cluster-id: /cluster-1
```

### Kubernetes HA

```yaml
high-availability: kubernetes
high-availability.storageDir: s3://bucket/flink/ha
kubernetes.cluster-id: my-cluster
```

## 内存配置

### TaskManager 内存

```yaml
# 进程总内存
taskmanager.memory.process.size: 4096m

# Flink 内存
taskmanager.memory.flink.size: 3072m

# 框架堆内存
taskmanager.memory.framework.heap.size: 128m

# 任务堆内存
taskmanager.memory.task.heap.size: 1024m

# 托管内存（用于 RocksDB 等）
taskmanager.memory.managed.size: 512m

# 网络内存
taskmanager.memory.network.min: 64m
taskmanager.memory.network.max: 1024m
```

### JobManager 内存

```yaml
jobmanager.memory.process.size: 1600m
jobmanager.memory.heap.size: 1024m
```

## 监控配置

### Metrics Reporter

```yaml
# Prometheus
metrics.reporter.promgateway.factory.class: org.apache.flink.metrics.prometheus.PrometheusPushGatewayReporterFactory
metrics.reporter.promgateway.host: prometheus-gateway
metrics.reporter.promgateway.port: 9091
metrics.reporter.promgateway.interval: 60 SECONDS

# InfluxDB
metrics.reporter.influxdb.factory.class: org.apache.flink.metrics.influxdb.InfluxdbReporterFactory
metrics.reporter.influxdb.host: influxdb
metrics.reporter.influxdb.port: 8086
metrics.reporter.influxdb.db: flink
```

### 重要监控指标

| 指标                     | 描述         | 告警阈值     |
| ------------------------ | ------------ | ------------ |
| `numRecordsInPerSecond`  | 输入吞吐量   | 根据业务     |
| `numRecordsOutPerSecond` | 输出吞吐量   | 根据业务     |
| `currentInputWatermark`  | 当前水印     | 延迟过大告警 |
| `lastCheckpointDuration` | 检查点耗时   | > 5min       |
| `lastCheckpointSize`     | 检查点大小   | 增长过快     |
| `fullRestarts`           | 全量重启次数 | > 0          |

## 日志配置

### log4j2.properties

```properties
rootLogger.level = INFO
rootLogger.appenderRef.file.ref = MainAppender

appender.main.name = MainAppender
appender.main.type = RollingFile
appender.main.fileName = ${sys:log.file}
appender.main.filePattern = ${sys:log.file}.%i
appender.main.layout.type = PatternLayout
appender.main.layout.pattern = %d{yyyy-MM-dd HH:mm:ss,SSS} %-5p %-60c %x - %m%n
appender.main.policies.type = Policies
appender.main.policies.size.type = SizeBasedTriggeringPolicy
appender.main.policies.size.size = 100MB
```

## 常用运维命令

### 作业管理

```bash
# 列出作业
flink list

# 取消作业
flink cancel <jobId>

# 从保存点恢复
flink run -s <savepointPath> myJob.jar

# 触发保存点
flink savepoint <jobId> <savepointDir>

# 修改并行度（需要保存点）
flink modify <jobId> -p 8
```

### 集群状态

```bash
# 查看 TaskManager
curl http://localhost:8081/taskmanagers

# 查看作业详情
curl http://localhost:8081/jobs/<jobId>

# 查看检查点统计
curl http://localhost:8081/jobs/<jobId>/checkpoints
```

## 故障排查

### 常见问题

| 问题       | 可能原因        | 解决方案             |
| ---------- | --------------- | -------------------- |
| OOM 错误   | 内存配置不足    | 增加堆内存或托管内存 |
| 检查点超时 | 状态过大/网络慢 | 使用增量检查点       |
| 背压严重   | 下游处理慢      | 优化算子/增加并行度  |
| 数据倾斜   | Key 分布不均    | 添加随机前缀         |

### 日志分析

```bash
# 查看 JobManager 日志
tail -f log/flink-*-jobmanager-*.log

# 查看 TaskManager 日志
tail -f log/flink-*-taskmanager-*.log

# 搜索异常
grep -i "exception" log/*.log
```

## 下一步

- 🚀 [性能优化](/docs/flink/performance-optimization) - 调优指南
- 📋 [最佳实践](/docs/flink/best-practices) - 开发规范
- ❓ [常见问题](/docs/flink/faq) - FAQ
