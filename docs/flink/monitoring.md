---
sidebar_position: 15
title: "监控与运维"
description: "Flink 生产环境监控与运维实践"
---

# Flink 监控与运维

> 适用版本：Apache Flink v2.2.0

## 内置 Metrics 系统

### Metrics 类型

| 类型          | 描述     | 示例           |
| ------------- | -------- | -------------- |
| **Counter**   | 计数器   | 处理记录数     |
| **Gauge**     | 瞬时值   | 当前队列大小   |
| **Meter**     | 速率     | 每秒处理记录数 |
| **Histogram** | 分布统计 | 延迟分布       |

### 自定义 Metrics

```java
public class MetricsFunction extends RichMapFunction<Event, Result> {
    private transient Counter processedCounter;
    private transient Meter throughputMeter;
    private transient Histogram latencyHistogram;

    @Override
    public void open(Configuration parameters) {
        MetricGroup group = getRuntimeContext().getMetricGroup();

        // 计数器
        processedCounter = group.counter("processedEvents");

        // 速率
        throughputMeter = group.meter("throughput", new MeterView(60));

        // 直方图
        latencyHistogram = group.histogram("latency",
            new DescriptiveStatisticsHistogram(1000));

        // 仪表盘
        group.gauge("queueSize", () -> getQueueSize());
    }

    @Override
    public Result map(Event event) {
        long start = System.currentTimeMillis();
        Result result = process(event);

        processedCounter.inc();
        throughputMeter.markEvent();
        latencyHistogram.update(System.currentTimeMillis() - start);

        return result;
    }
}
```

## Prometheus 集成

### 配置 Metrics Reporter

```yaml
# flink-conf.yaml
metrics.reporter.promgateway.factory.class: org.apache.flink.metrics.prometheus.PrometheusPushGatewayReporterFactory
metrics.reporter.promgateway.host: prometheus-pushgateway
metrics.reporter.promgateway.port: 9091
metrics.reporter.promgateway.jobName: flink-job
metrics.reporter.promgateway.randomJobNameSuffix: true
metrics.reporter.promgateway.deleteOnShutdown: false
metrics.reporter.promgateway.interval: 30 SECONDS

# 或使用 Prometheus Pull 模式
metrics.reporter.prom.factory.class: org.apache.flink.metrics.prometheus.PrometheusReporterFactory
metrics.reporter.prom.port: 9999
```

### 添加依赖

```xml
<dependency>
    <groupId>org.apache.flink</groupId>
    <artifactId>flink-metrics-prometheus</artifactId>
    <version>${flink.version}</version>
</dependency>
```

## Grafana Dashboard

### 关键监控面板

#### 作业概览

```promql
# 作业运行状态
flink_jobmanager_job_uptime{job_name="$job_name"}

# 作业重启次数
flink_jobmanager_job_numRestarts{job_name="$job_name"}

# 检查点成功率
rate(flink_jobmanager_job_numberOfCompletedCheckpoints[5m]) /
rate(flink_jobmanager_job_numberOfInProgressCheckpoints[5m])
```

#### 吞吐量监控

```promql
# 每秒输入记录数
rate(flink_taskmanager_job_task_numRecordsIn[1m])

# 每秒输出记录数
rate(flink_taskmanager_job_task_numRecordsOut[1m])

# 每秒处理字节数
rate(flink_taskmanager_job_task_numBytesIn[1m])
```

#### 延迟监控

```promql
# 端到端延迟
flink_taskmanager_job_latency_source_id_operator_id_operator_subtask_index_latency{
    quantile="0.99"
}

# 水印延迟
time() * 1000 - flink_taskmanager_job_task_currentInputWatermark
```

#### 背压监控

```promql
# 背压率
flink_taskmanager_job_task_isBackPressured

# 输出缓冲区使用率
flink_taskmanager_job_task_buffers_outPoolUsage
```

### Dashboard JSON 模板

```json
{
  "title": "Flink Job Monitoring",
  "panels": [
    {
      "title": "Records Throughput",
      "type": "graph",
      "targets": [
        {
          "expr": "rate(flink_taskmanager_job_task_numRecordsIn[1m])",
          "legendFormat": "{{task_name}} - In"
        },
        {
          "expr": "rate(flink_taskmanager_job_task_numRecordsOut[1m])",
          "legendFormat": "{{task_name}} - Out"
        }
      ]
    },
    {
      "title": "Checkpoint Duration",
      "type": "graph",
      "targets": [
        {
          "expr": "flink_jobmanager_job_lastCheckpointDuration",
          "legendFormat": "Duration (ms)"
        }
      ]
    }
  ]
}
```

## 告警配置

### Prometheus AlertManager 规则

```yaml
groups:
  - name: flink-critical
    rules:
      # 作业失败告警
      - alert: FlinkJobFailed
        expr: flink_jobmanager_job_uptime == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Flink 作业 {{ $labels.job_name }} 失败"
          description: "作业已经停止运行超过 1 分钟"

      # 检查点失败告警
      - alert: FlinkCheckpointFailed
        expr: increase(flink_jobmanager_job_numberOfFailedCheckpoints[10m]) > 3
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Flink 检查点频繁失败"
          description: "10分钟内检查点失败 {{ $value }} 次"

      # 检查点时间过长
      - alert: FlinkCheckpointTooSlow
        expr: flink_jobmanager_job_lastCheckpointDuration > 600000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "检查点耗时过长"
          description: "检查点耗时 {{ $value }}ms，超过 10 分钟"

  - name: flink-performance
    rules:
      # 背压告警
      - alert: FlinkHighBackpressure
        expr: flink_taskmanager_job_task_isBackPressured > 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "检测到高背压"
          description: "Task {{ $labels.task_name }} 背压率 {{ $value }}"

      # 延迟告警
      - alert: FlinkHighLatency
        expr: flink_taskmanager_job_latency_source_id_operator_id_latency{quantile="0.99"} > 10000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "端到端延迟过高"
          description: "P99 延迟 {{ $value }}ms"

      # 消费延迟告警
      - alert: FlinkKafkaLag
        expr: flink_taskmanager_job_task_operator_KafkaSourceReader_KafkaConsumer_records_lag_max > 100000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Kafka 消费延迟过大"
          description: "消费延迟 {{ $value }} 条消息"
```

## 常见运维操作

### 作业管理

```bash
# 查看作业列表
flink list

# 查看作业详情
flink info <job-id>

# 取消作业
flink cancel <job-id>

# 创建保存点
flink savepoint <job-id> hdfs:///savepoints

# 从保存点恢复
flink run -s hdfs:///savepoints/savepoint-xxx job.jar
```

### 扩缩容

```bash
# 1. 创建保存点
flink savepoint <job-id> hdfs:///savepoints

# 2. 取消作业
flink cancel <job-id>

# 3. 修改并行度后恢复
flink run -p 8 -s hdfs:///savepoints/savepoint-xxx job.jar
```

### 版本升级

```bash
# 1. 创建保存点
flink savepoint <job-id> hdfs:///savepoints

# 2. 停止旧版本作业
flink cancel <job-id>

# 3. 部署新版本
flink run -s hdfs:///savepoints/savepoint-xxx new-job.jar
```

## 故障排查

### 常见问题诊断

| 问题       | 诊断方法             | 解决方案                 |
| ---------- | -------------------- | ------------------------ |
| OOM        | 查看 GC 日志、堆转储 | 增加内存、优化状态       |
| 背压       | Web UI 背压指标      | 增加并行度、优化算子     |
| 检查点超时 | 检查点日志           | 使用增量检查点、增大超时 |
| Kafka 延迟 | Consumer Lag         | 增加并行度、检查网络     |

### 日志分析

```bash
# 查看 JobManager 日志
tail -f log/flink-*-jobmanager-*.log

# 查看 TaskManager 日志
tail -f log/flink-*-taskmanager-*.log

# 搜索异常
grep -r "Exception" log/

# 搜索检查点日志
grep -r "Checkpoint" log/flink-*-jobmanager-*.log
```

### 堆转储分析

```bash
# 生成堆转储
jmap -dump:format=b,file=heap.hprof <pid>

# 使用 MAT 或 VisualVM 分析
```

## Web UI 监控

### 关键页面

| 页面         | 用途                 |
| ------------ | -------------------- |
| Overview     | 集群概览、资源使用   |
| Jobs         | 作业列表、运行状态   |
| Job Details  | 算子拓扑、各算子状态 |
| Checkpoints  | 检查点历史、耗时     |
| Metrics      | 详细指标查看         |
| Backpressure | 背压状态分析         |

### REST API

```bash
# 获取作业列表
curl http://localhost:8081/jobs

# 获取作业详情
curl http://localhost:8081/jobs/<job-id>

# 获取检查点统计
curl http://localhost:8081/jobs/<job-id>/checkpoints

# 触发保存点
curl -X POST http://localhost:8081/jobs/<job-id>/savepoints \
  -d '{"target-directory": "hdfs:///savepoints"}'
```

## 下一步

- 🚀 [性能优化](/docs/flink/performance-optimization) - 调优指南
- 📋 [最佳实践](/docs/flink/best-practices) - 开发规范
- 🔧 [部署与运维](/docs/flink/deployment) - 部署配置
