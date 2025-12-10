---
id: monitoring
title: 监控与运维
sidebar_label: 监控与运维
sidebar_position: 12
---

# Spring Cloud Alibaba 监控与运维

> [!IMPORTANT]
> **生产环境必备**: 完善的监控体系是保障系统稳定运行的基础,本文介绍 Spring Cloud Alibaba 各组件的监控和运维方案。

## 1. Nacos 监控

### Nacos 控制台

**访问**: `http://localhost:8848/nacos`

**核心功能**:

- 服务列表查看
- 服务实例健康状态
- 配置管理
- 命名空间管理

### 关键指标

| 指标               | 说明             | 正常范围    |
| ------------------ | ---------------- | ----------- |
| 服务实例数         | 注册的服务实例数 | -           |
| 健康实例比例       | 健康实例/总实例  | > 80%       |
| 配置变更次数       | 配置修改频率     | -           |
| 服务订阅数         | 服务被订阅次数   | -           |

### Metrics 接口

Nacos 提供 Prometheus 格式的 Metrics:

```bash
# Nacos Metrics
curl http://localhost:8848/nacos/actuator/prometheus
```

### 日志监控

**日志路径**: `${nacos.home}/logs/`

```bash
# 关键日志文件
nacos.log          # 主日志
naming-server.log  # 服务注册日志
config-server.log  # 配置中心日志
```

## 2. Sentinel 监控

### Sentinel Dashboard

**启动控制台**:

```bash
java -Dserver.port=8080 -jar sentinel-dashboard.jar
```

**访问**: `http://localhost:8080`

**核心功能**:

- 实时监控
- 规则配置
- 簇点链路
- 机器列表

### 实时监控指标

| 指标         | 说明           |
| ------------ | -------------- |
| QPS          | 每秒请求数     |
| 通过 QPS     | 通过的请求数   |
| 拒绝 QPS     | 被限流的请求数 |
| 异常 QPS     | 异常的请求数   |
| 平均响应时间 | 响应时间       |

### Metrics 接口

```java
@RestController
public class MetricsController {

    @GetMapping("/metrics")
    public Map<String, Object> getMetrics() {
        Map<String, Object> metrics = new HashMap<>();
        
        // 获取所有资源的统计信息
        for (String resource : InMemoryMetricsRepository.getInstance().listResourcesOfType(MetricEvent.PASS)) {
            metrics.put(resource, InMemoryMetricsRepository.getInstance()
                .queryByAppAndResourceBetween(resource, 0, System.currentTimeMillis()));
        }
        
        return metrics;
    }
}
```

## 3. Seata 监控

### Seata Server 控制台

**访问**: `http://localhost:7091`

**核心功能**:

- 全局事务查询
- 分支事务查询
- 全局锁查询

### 关键指标

| 指标           | 说明               | 告警阈值      |
| -------------- | ------------------ | ------------- |
| 活跃全局事务数 | 正在执行的全局事务 | > 1000        |
| 全局事务超时数 | 超时的全局事务     | > 0           |
| 事务提交成功率 | 提交成功/总事务    | < 95%         |
| 事务回滚率     | 回滚/总事务        | > 5%          |

### 日志监控

```bash
# Seata Server 日志
${seata.home}/logs/seata_gc.log
${seata.home}/logs/seata.log
```

### Metrics 集成

```yaml
# application.yml
seata:
  metrics:
    enabled: true
    registry-type: compact
    exporter-list: prometheus
    exporter-prometheus-port: 9898
```

访问 Metrics:

```bash
curl http://localhost:9898/metrics
```

## 4. RocketMQ 监控

### RocketMQ Dashboard

**下载和启动**:

```bash
git clone https://github.com/apache/rocketmq-dashboard.git
cd rocketmq-dashboard
mvn clean package -Dmaven.test.skip=true
java -jar target/rocketmq-dashboard-1.0.0.jar
```

**访问**: `http://localhost:8080`

### 关键指标

| 指标         | 说明             | 告警阈值    |
| ------------ | ---------------- | ----------- |
| 消息堆积数   | 未消费的消息数   | > 10000     |
| 消费 TPS     | 每秒消费消息数   | -           |
| 生产 TPS     | 每秒生产消息数   | -           |
| 消费延迟     | 消息延迟时间     | > 1s        |

### 消息堆积监控

```java
@Component
public class RocketMQMonitor {

    @Autowired
    private DefaultMQAdminExt mqAdminExt;

    public long getMessageBacklog(String consumerGroup, String topic) {
        try {
            ConsumeStats consumeStats = mqAdminExt.examineConsumeStats(consumerGroup);
            long diff = consumeStats.computeTotalDiff();
            return diff;
        } catch (Exception e) {
            log.error("获取消息堆积失败", e);
            return -1;
        }
    }
}
```

## 5. Dubbo 监控

### Dubbo Admin

**启动**:

```bash
git clone https://github.com/apache/dubbo-admin.git
cd dubbo-admin
mvn clean package
cd dubbo-admin-distribution/target
java -jar dubbo-admin-0.6.0.jar
```

**访问**: `http://localhost:8080`

### 关键指标

| 指标         | 说明             |
| ------------ | ---------------- |
| 服务提供者数 | Provider 数量    |
| 服务消费者数 | Consumer 数量    |
| 调用成功率   | 成功/总调用      |
| 平均响应时间 | 平均 RT          |
| 调用 QPS     | 每秒调用数       |

## 6. Spring Boot Actuator

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>

<dependency>
    <groupId>io.micrometer</groupId>
    <artifactId>micrometer-registry-prometheus</artifactId>
</dependency>
```

### 配置

```yaml
management:
  endpoints:
    web:
      exposure:
        include: "*"
  endpoint:
    health:
      show-details: always
  metrics:
    export:
      prometheus:
        enabled: true
```

### 核心端点

| 端点          | 说明           | URL                       |
| ------------- | -------------- | ------------------------- |
| health        | 健康检查       | /actuator/health          |
| info          | 应用信息       | /actuator/info            |
| metrics       | 度量指标       | /actuator/metrics         |
| prometheus    | Prometheus格式 | /actuator/prometheus      |
| env           | 环境变量       | /actuator/env             |
| beans         | Spring Beans   | /actuator/beans           |

### 自定义健康检查

```java
@Component
public class CustomHealthIndicator implements HealthIndicator {

    @Override
    public Health health() {
        // 检查数据库连接
        if (checkDatabase()) {
            return Health.up()
                .withDetail("database", "MySQL")
                .withDetail("status", "UP")
                .build();
        }
        
        return Health.down()
            .withDetail("database", "MySQL")
            .withDetail("error", "连接失败")
            .build();
    }
    
    private boolean checkDatabase() {
        // 实际检查逻辑
        return true;
    }
}
```

## 7. Prometheus + Grafana

### Prometheus 配置

```yaml title="prometheus.yml"
global:
  scrape_interval: 15s

scrape_configs:
  # Nacos
  - job_name: 'nacos'
    static_configs:
      - targets: ['localhost:8848']
    metrics_path: '/nacos/actuator/prometheus'

  # Sentinel
  - job_name: 'sentinel'
    static_configs:
      - targets: ['localhost:8719']

  # Spring Boot 应用
  - job_name: 'spring-boot-apps'
    static_configs:
      - targets:
        - 'localhost:8081'  # user-service
        - 'localhost:8082'  # order-service
    metrics_path: '/actuator/prometheus'

  # Seata
  - job_name: 'seata'
    static_configs:
      - targets: ['localhost:9898']
```

### Grafana Dashboard

**推荐 Dashboard**:

1. **Spring Boot Dashboard**: ID 6756
2. **JVM Dashboard**: ID 4701
3. **MySQL Dashboard**: ID 7362
4. **Redis Dashboard**: ID 11835

**导入 Dashboard**:

```
Grafana → Create → Import → 输入 Dashboard ID
```

### 关键指标面板

**JVM 监控**:

- Heap Memory Usage
- Non-Heap Memory Usage
- GC Count
- GC Time
- Thread Count

**应用监控**:

- QPS
- Response Time (P50, P95, P99)
- Error Rate
- HTTP Status Codes

**数据库监控**:

- Connection Pool Size
- Active Connections
- Query Time

## 8. 日志收集 (ELK)

### 架构

```
应用服务 → Filebeat → Logstash → Elasticsearch → Kibana
```

### Logback 配置

```xml title="logback-spring.xml"
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <!-- 控制台输出 -->
    <appender name="CONSOLE" class="ch.qos.logback.core.ConsoleAppender">
        <encoder>
            <pattern>%d{yyyy-MM-dd HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg%n</pattern>
        </encoder>
    </appender>

    <!-- 文件输出 -->
    <appender name="FILE" class="ch.qos.logback.core.rolling.RollingFileAppender">
        <file>logs/application.log</file>
        <rollingPolicy class="ch.qos.logback.core.rolling.TimeBasedRollingPolicy">
            <fileNamePattern>logs/application.%d{yyyy-MM-dd}.log</fileNamePattern>
            <maxHistory>30</maxHistory>
        </rollingPolicy>
        <encoder>
            <pattern>%d{yyyy-MM-dd HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg%n</pattern>
        </encoder>
    </appender>

    <!-- JSON 格式 (用于 ELK) -->
    <appender name="JSON" class="ch.qos.logback.core.rolling.RollingFileAppender">
        <file>logs/application-json.log</file>
        <rollingPolicy class="ch.qos.logback.core.rolling.TimeBasedRollingPolicy">
            <fileNamePattern>logs/application-json.%d{yyyy-MM-dd}.log</fileNamePattern>
            <maxHistory>30</maxHistory>
        </rollingPolicy>
        <encoder class="net.logstash.logback.encoder.LogstashEncoder"/>
    </appender>

    <root level="INFO">
        <appender-ref ref="CONSOLE"/>
        <appender-ref ref="FILE"/>
        <appender-ref ref="JSON"/>
    </root>
</configuration>
```

## 9. 链路追踪 (Sleuth + Zipkin)

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-sleuth</artifactId>
</dependency>

<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-sleuth-zipkin</artifactId>
</dependency>
```

### 配置

```yaml
spring:
  zipkin:
    base-url: http://localhost:9411
    sender:
      type: web
  sleuth:
    sampler:
      probability: 1.0  # 采样率 100%
```

### Zipkin Server

```bash
# Docker 启动
docker run -d -p 9411:9411 openzipkin/zipkin
```

**访问**: `http://localhost:9411`

## 10. 告警配置

### Prometheus Alertmanager

```yaml title="alert.rules.yml"
groups:
  - name: application_alerts
    rules:
      # 错误率告警
      - alert: HighErrorRate
        expr: rate(http_server_requests_seconds_count{status=~"5.."}[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "高错误率告警"
          description: "服务 {{ $labels.application }} 错误率超过 1%"

      # 响应时间告警
      - alert: SlowResponse
        expr: http_server_requests_seconds{quantile="0.95"} > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "响应时间过慢"
          description: "服务 {{ $labels.application }} P95 响应时间超过 1s"

      # 服务下线告警
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "服务下线"
          description: "服务 {{ $labels.job }} 已下线"
```

### Alertmanager 配置

```yaml title="alertmanager.yml"
global:
  resolve_timeout: 5m

route:
  receiver: 'default'
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h

receivers:
  - name: 'default'
    email_configs:
      - to: 'ops@example.com'
        from: 'alert@example.com'
        smarthost: 'smtp.example.com:587'
        auth_username: 'alert@example.com'
        auth_password: 'password'
    webhook_configs:
      - url: 'http://localhost:8080/alert/webhook'
```

## 11. 监控最佳实践

### 核心指标 (RED)

- **Rate**: 请求速率 (QPS)
- **Error**: 错误率
- **Duration**: 响应时间

### 黄金指标 (USE)

- **Utilization**: 资源利用率
- **Saturation**: 资源饱和度
- **Errors**: 错误

### 告警原则

1. **准确性**: 避免误报和漏报
2. **及时性**: 快速发现问题
3. **可操作性**: 告警要能指导处理
4. **分级**: Critical > Warning > Info

### 日志规范

**日志级别使用**:

- **ERROR**: 影响业务的错误
- **WARN**: 潜在问题
- **INFO**: 重要业务流程
- **DEBUG**: 调试信息

**日志格式**:

```java
// ✅ 好的日志
log.info("用户登录成功, userId={}, ip={}", userId, ip);

// ❌ 不好的日志
log.info("用户" + userId + "登录了");
```

## 12. 运维工具脚本

### 健康检查脚本

```bash
#!/bin/bash

# 检查服务健康状态
check_health() {
    SERVICE=$1
    PORT=$2
    
    HEALTH=$(curl -s http://localhost:${PORT}/actuator/health | jq -r '.status')
    
    if [ "$HEALTH" == "UP" ]; then
        echo "✅ $SERVICE is UP"
    else
        echo "❌ $SERVICE is DOWN"
        # 发送告警
        send_alert "$SERVICE is DOWN"
    fi
}

check_health "user-service" 8081
check_health "order-service" 8082
check_health "api-gateway" 8080
```

### 日志分析脚本

```bash
#!/bin/bash

# 分析错误日志
analyze_errors() {
    LOG_FILE=$1
    TIME_RANGE="5 minutes ago"
    
    echo "=== 最近 5 分钟错误统计 ==="
    grep -i "error" $LOG_FILE | \
        awk -v d="$(date --date="$TIME_RANGE" +%s)" '$0 > d' | \
        wc -l
    
    echo "=== TOP 10 错误类型 ==="
    grep -i "error" $LOG_FILE | \
        awk '{print $NF}' | \
        sort | uniq -c | \
        sort -rn | head -10
}

analyze_errors "/var/log/application.log"
```

## 13. 总结

### 监控覆盖

- [x] 基础设施监控 (CPU、内存、磁盘、网络)
- [x] 应用监控 (QPS、RT、错误率)
- [x] 中间件监控 (Nacos、Sentinel、RocketMQ)
- [x] 数据库监控 (MySQL、Redis)
- [x] 日志收集和分析
- [x] 链路追踪

### 关键要点

- 监控要覆盖全链路
- 告警要及时准确
- 日志要结构化
- 定期review监控指标

---

**下一步**: 学习 [FAQ 常见问题](./faq)
