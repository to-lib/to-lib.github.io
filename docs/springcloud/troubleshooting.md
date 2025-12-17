---
title: Spring Cloud 排障手册
sidebar_label: 排障手册
sidebar_position: 21.5
---

# Spring Cloud 排障手册

> [!TIP]
> 本页以“**现象 → 可能原因 → 排查步骤 → 解决方案**”的方式，汇总 Spring Cloud 常见线上问题。

## 1. 注册中心相关

### 1.1 服务已启动但 Eureka 控制台不显示实例

**可能原因**：

- 服务没有启用注册客户端或依赖未引入。
- `eureka.client.service-url.defaultZone` 配置错误（路径缺少 `/eureka/`）。
- 网络不可达（容器网络 / 安全组 / 防火墙）。

**排查步骤**：

- 检查配置是否正确：

```yaml
eureka:
  client:
    service-url:
      defaultZone: http://localhost:8761/eureka/
```

- 检查服务启动日志中是否出现注册成功相关日志。
- 直接访问 `http://<eureka-server>:8761/eureka/apps` 看是否有应用注册。

**解决方案**：

- 修正 `defaultZone`。
- 开发环境可降低心跳与拉取间隔，加快可见性验证。

### 1.2 Eureka 显示自我保护模式（EMERGENCY）

**可能原因**：

- 网络抖动导致心跳大量丢失。
- 心跳/过期参数配置不合理。

**排查步骤**：

- 检查 Eureka Server 与客户端网络连通性、DNS、时钟同步。
- 查看客户端的心跳是否稳定。

**解决方案**：

- 生产环境建议保持自我保护开启，优先解决网络/心跳问题。
- 开发环境可暂时关闭自我保护（仅用于本地学习验证）。

## 2. Gateway / 路由相关

### 2.1 Gateway 返回 503 Service Unavailable

**可能原因**：

- `lb://` 走服务发现，但后端服务未注册/实例为 0。
- 缺少客户端负载均衡依赖（Spring Cloud LoadBalancer）。
- 服务名不一致（注册名与路由中写的不一致）。

**排查步骤**：

- 在注册中心确认后端实例存在。
- 检查 gateway 配置的 `uri`：

```yaml
spring:
  cloud:
    gateway:
      routes:
        - id: user-service
          uri: lb://user-service
          predicates:
            - Path=/api/users/**
```

- 检查是否引入了 LoadBalancer 依赖（尤其是 Spring Boot 3.x 项目）。

**解决方案**：

- 修正服务名与路由配置。
- 引入 `spring-cloud-starter-loadbalancer`。

### 2.2 Gateway 路由不匹配（404 / 路由不生效）

**可能原因**：

- `Path` 断言写错（大小写、通配符）。
- `StripPrefix` 使用不当导致后端路径不一致。

**排查步骤**：

- 先把路由配置简化到最小可用（只保留 `Path`），验证是否能转发。
- 临时加一个全局日志过滤器打印 `request.getPath()`，确认实际请求路径。

**解决方案**：

- 修正断言与过滤器。
- 为不同服务统一 API 前缀规则，避免后续维护成本。

## 3. OpenFeign / 服务调用相关

### 3.1 Feign 调用超时或偶发慢

**可能原因**：

- 服务端响应慢（数据库、下游依赖）。
- 客户端超时设置不合理。
- 连接池不足。

**排查步骤**：

- 先用链路追踪定位慢点（Boot 3.x 推荐 [Micrometer Tracing](/docs/springcloud/micrometer-tracing)）。
- 对比服务端日志与客户端日志的时间点。
- 打开 Feign 日志（仅在排障阶段启用）。

**解决方案**：

- 合理设置 `connectTimeout/readTimeout`。
- 调整连接池或切换到 OkHttp/HttpClient。
- 对核心调用增加熔断/降级（Resilience4j）。

### 3.2 Feign 请求头丢失（Authorization / TraceId 等）

**可能原因**：

- 未配置 `RequestInterceptor` 透传。
- 异步调用导致上下文丢失。

**排查步骤**：

- 抓包或打印 `RequestTemplate` 的 headers。
- 检查线程切换点（`@Async`、线程池、消息回调）。

**解决方案**：

- 使用 `RequestInterceptor` 透传必要请求头。
- 对线程池做上下文传播配置。

## 4. Config / 动态刷新相关

### 4.1 配置不生效或读取不到

**可能原因**：

- 配置中心地址、profile、label 不匹配。
- 客户端启动时序问题（需要更早加载）。

**排查步骤**：

- 直接访问 Config Server 的 REST API 验证配置是否能返回。
- 检查客户端启动日志中是否成功拉取远端配置。

**解决方案**：

- 修正 profile/label 与文件命名。
- 生产环境避免让关键服务完全依赖配置中心单点。

### 4.2 @RefreshScope 刷新后部分配置仍未更新

**可能原因**：

- 并非所有配置都支持运行时刷新（例如连接池、端口等）。

**排查步骤**：

- 确认变更的配置属于可刷新范围。
- 确认 refresh/busrefresh 端点暴露且实际调用成功。

**解决方案**：

- 将不可刷新配置的变更纳入“滚动发布/重启”流程。

## 5. 链路追踪相关

### 5.1 日志里没有 traceId/spanId

**可能原因**：

- 依赖未引入或 exporter 未配置。
- 日志 pattern 未输出 MDC。

**排查步骤**：

- Boot 3.x：确认已按 [Micrometer Tracing](/docs/springcloud/micrometer-tracing) 配置。
- 检查 `logback-spring.xml` pattern 是否包含 `%X{traceId}` / `%X{spanId}`。

**解决方案**：

- 修正依赖与配置。
- 在开发环境将采样率暂时调到 `1.0` 以便验证。

---

**相关文档**：

- [快速参考](/docs/springcloud/quick-reference)
- [常见问题](/docs/springcloud/faq)
- [Micrometer Tracing](/docs/springcloud/micrometer-tracing)
