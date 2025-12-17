---
title: Micrometer Tracing 链路追踪
sidebar_label: Micrometer Tracing
sidebar_position: 9.1
---

# Micrometer Tracing 链路追踪

> [!TIP]
> **Spring Boot 3.x 推荐方案**：从 Spring Boot 3 开始，Spring Cloud Sleuth 已进入维护结束阶段，Spring 官方推荐使用 **Micrometer Tracing** 作为新的链路追踪与上下文传播方案。

## 1. 背景：Sleuth → Micrometer Tracing

- **Spring Boot 2.x**：常用组合为 `Sleuth + Zipkin`。
- **Spring Boot 3.x**：使用 **Micrometer Tracing**，通过桥接（Bridge）接入 Brave / OpenTelemetry，并输出到 Zipkin / OTLP 等后端。

> [!IMPORTANT]
> 如果你的项目是 Spring Boot 3.x / Spring Cloud 2023.x，建议直接阅读本文并以 Micrometer Tracing 为主；Sleuth 文档可作为老项目参考。

## 2. 核心概念

Micrometer Tracing 仍然沿用常见分布式追踪概念：

- **Trace**：一次请求在分布式系统中的完整调用链。
- **Span**：调用链中的一个片段（一次 RPC、一次 DB 调用、一次消息消费等）。
- **TraceId / SpanId**：用于跨服务关联与定位。

## 3. 基本接入（Zipkin 示例）

### 3.1 依赖

典型依赖组合（Maven）：

- `spring-boot-starter-actuator`
- `micrometer-tracing-bridge-brave`
- `zipkin-reporter-brave`

> [!TIP]
> 追踪实现可选择 Brave 或 OpenTelemetry；本文用 Zipkin + Brave 举例，适合入门与排障。

### 3.2 配置

```yaml
management:
  tracing:
    sampling:
      probability: 1.0
  zipkin:
    tracing:
      endpoint: http://localhost:9411/api/v2/spans
```

- `probability`：采样率。开发环境可设为 `1.0`，生产通常为 `0.01-0.1`。
- `endpoint`：Zipkin 接收 spans 的地址。

## 4. 日志关联（MDC）

开启 tracing 后，通常会把 `traceId` / `spanId` 写入 MDC，便于日志检索与链路还原。

Logback 示例（`logback-spring.xml`）：

```xml
<pattern>%d{yyyy-MM-dd HH:mm:ss.SSS} [%X{traceId},%X{spanId}] %-5level %logger{36} - %msg%n</pattern>
```

> [!TIP]
> 日志里“有没有 traceId”是判断链路追踪是否生效的最快方式之一。

## 5. 上下文传播（跨线程/异步）

链路追踪常见问题是“换线程后 TraceId 丢失”。典型场景：

- `@Async`
- 自建线程池（`ExecutorService`）
- 消息消费回调

思路是 **在任务提交与执行之间传播上下文**。

> [!IMPORTANT]
> 如果你在异步场景中发现 `traceId` 变了或消失，优先检查线程池是否做了上下文传播配置。

## 6. 常见问题

### 6.1 日志里没有 traceId/spanId

- 确认 tracing 相关依赖已引入（桥接实现 + exporter）。
- 确认项目中存在对 Web 请求或客户端调用的 tracing instrumentation（通常依赖齐全后自动生效）。
- 确认日志 pattern 使用的是 `%X{traceId}` / `%X{spanId}`。

### 6.2 Zipkin UI 查不到数据

- 采样率太低：先把 `probability` 调到 `1.0`。
- endpoint 配错：确认是 `http://zipkin:9411/api/v2/spans`。
- 网络不通：服务容器/集群内是否能访问 Zipkin。

---

**相关文档**：

- [Sleuth 链路追踪（旧）](/docs/springcloud/sleuth)
- [Zipkin 追踪](/docs/springcloud/zipkin)
- [Spring Cloud 常见问题](/docs/springcloud/faq)
