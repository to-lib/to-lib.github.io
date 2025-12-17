---
sidebar_position: 19
---

# 可观测性（Metrics / Logs / Traces）

> [!IMPORTANT]
> **生产环境建议把“指标、日志、链路追踪”作为一个整体来设计**：
> - 指标用于发现趋势与告警
> - 日志用于定位细节
> - 链路追踪用于跨服务还原调用路径
>
> 本文以 Spring Boot 3.x 为主，默认使用 Actuator + Micrometer 体系。

## 目标与基本原则

- **统一三件套**：Metrics、Logs、Traces
- **日志可关联**：日志中必须能看到 `traceId` / `spanId`
- **面向 SLO**：围绕延迟、错误率、吞吐量、饱和度（RED/USE）
- **默认可观测**：在 Web、DB、MQ、HTTP Client 等关键链路自动埋点

## 1. Metrics：指标采集与导出

### 1.1 引入依赖

```xml
<!-- Actuator：提供 /actuator/metrics 等端点 -->
<dependency>
  <groupId>org.springframework.boot</groupId>
  <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>

<!-- Prometheus：导出 /actuator/prometheus（推荐） -->
<dependency>
  <groupId>io.micrometer</groupId>
  <artifactId>micrometer-registry-prometheus</artifactId>
</dependency>
```

### 1.2 常用配置

```yaml
management:
  endpoints:
    web:
      exposure:
        include: health,info,metrics,prometheus,loggers,env,configprops
  endpoint:
    health:
      show-details: when_authorized
  metrics:
    tags:
      application: ${spring.application.name}
```

> [!TIP]
> 指标相关更详细的 Actuator/Micrometer 使用，可以参考：
> - `/docs/springboot/health-monitoring`

### 1.3 自定义业务指标

```java
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import org.springframework.stereotype.Component;

@Component
public class OrderMetrics {

  private final Counter orderCreated;

  public OrderMetrics(MeterRegistry registry) {
    this.orderCreated = Counter.builder("biz.order.created")
        .description("Orders created")
        .register(registry);
  }

  public void onOrderCreated() {
    orderCreated.increment();
  }
}
```

## 2. Logs：日志规范与结构化

### 2.1 日志级别与输出建议

- **INFO**：关键业务事件（下单成功、支付完成）
- **WARN**：可恢复/预期外但可处理（参数不合法、第三方超时重试）
- **ERROR**：需要关注的异常（影响请求成功率、数据一致性）
- **DEBUG**：开发排查（不要在生产长期开启大量 DEBUG）

### 2.2 建议使用 logback-spring.xml

Spring Boot 默认使用 Logback；建议使用 `logback-spring.xml`（支持 profile、Spring 属性）。

### 2.3 结构化日志（JSON）

如果你需要把日志投递到 ELK/EFK/ClickHouse 等，建议输出 JSON 结构化日志，常见做法是使用 logstash encoder。

```xml
<dependency>
  <groupId>net.logstash.logback</groupId>
  <artifactId>logstash-logback-encoder</artifactId>
  <version>7.4</version>
</dependency>
```

> [!NOTE]
> 结构化日志的字段建议：`timestamp`、`level`、`logger`、`message`、`traceId`、`spanId`、`userId`、`requestId`、`costMs`。

## 3. Traces：链路追踪（Micrometer Tracing / OpenTelemetry）

### 3.1 依赖选择

Spring Boot 3.x 推荐使用 Micrometer Tracing（底层可桥接 OpenTelemetry）。

```xml
<dependency>
  <groupId>io.micrometer</groupId>
  <artifactId>micrometer-tracing-bridge-otel</artifactId>
</dependency>

<dependency>
  <groupId>io.opentelemetry</groupId>
  <artifactId>opentelemetry-exporter-otlp</artifactId>
</dependency>
```

### 3.2 基本配置（OTLP 导出）

```yaml
management:
  tracing:
    sampling:
      probability: 1.0

otel:
  exporter:
    otlp:
      endpoint: http://localhost:4317
```

> [!WARNING]
> `sampling.probability=1.0` 只适用于开发/压测。生产通常建议从 `0.01 ~ 0.2` 起步，结合错误采样、慢请求采样。

### 3.3 日志与 Trace 关联（MDC）

目标：每条请求日志都能看到 `traceId/spanId`。

在 Logback pattern 中加入 MDC 字段（不同框架字段名可能略有不同）：

```xml
<pattern>%d{yyyy-MM-dd HH:mm:ss.SSS} %-5level [%thread] %logger{36} - %msg traceId=%X{traceId} spanId=%X{spanId}%n</pattern>
```

> [!TIP]
> 如果你输出 JSON 日志，同样把 `traceId/spanId` 作为 JSON 字段输出。

### 3.4 自定义 Span（关键业务埋点）

```java
import io.micrometer.tracing.Tracer;
import io.micrometer.tracing.Span;
import org.springframework.stereotype.Service;

@Service
public class PaymentService {

  private final Tracer tracer;

  public PaymentService(Tracer tracer) {
    this.tracer = tracer;
  }

  public void pay() {
    Span span = tracer.nextSpan().name("biz.pay").start();
    try (Tracer.SpanInScope ws = tracer.withSpan(span)) {
      // 业务逻辑
    } finally {
      span.end();
    }
  }
}
```

## 4. 可观测性落地清单

- **Web**：HTTP 入口统一打点（延迟、状态码、错误率）
- **DB**：慢 SQL、连接池指标（HikariCP）、事务错误
- **MQ**：生产/消费延迟与失败数
- **HTTP Client**：对外调用的耗时与错误（含重试次数）
- **日志字段**：必须可关联（traceId/spanId + requestId）
- **告警**：建议至少覆盖：
  - 5xx 错误率
  - P95/P99 延迟
  - JVM 内存/GC
  - 线程池/连接池饱和

## 5. 相关文档

- `/docs/springboot/health-monitoring`
- `/docs/springboot/performance-optimization`
- `/docs/springboot/deployment`
