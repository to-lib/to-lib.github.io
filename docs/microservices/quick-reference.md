---
sidebar_position: 11
title: 快速参考
description: 微服务快速参考 - 术语、工具、命令速查
---

# 快速参考

## 术语速查表

| 术语 | 英文 | 说明 |
| ---- | ---- | ---- |
| 微服务 | Microservices | 将应用拆分为小型独立服务的架构风格 |
| API 网关 | API Gateway | 微服务的统一入口，处理路由、认证等 |
| 服务注册 | Service Registration | 服务启动时向注册中心注册地址 |
| 服务发现 | Service Discovery | 从注册中心获取服务实例列表 |
| 负载均衡 | Load Balancing | 将请求分发到多个服务实例 |
| 断路器 | Circuit Breaker | 防止级联故障的保护机制 |
| 限流 | Rate Limiting | 限制请求速率保护服务 |
| 熔断 | Fuse | 服务故障时快速失败 |
| 降级 | Degradation | 服务不可用时返回备用响应 |
| 服务网格 | Service Mesh | 处理服务间通信的基础设施层 |
| Sidecar | Sidecar | 与服务一起部署的代理容器 |
| 分布式追踪 | Distributed Tracing | 跟踪请求在多服务间的调用链 |
| 链路追踪 | Trace | 一次完整请求的调用链 |
| Span | Span | 调用链中的一个操作单元 |
| 最终一致性 | Eventual Consistency | 数据最终会达到一致状态 |
| Saga | Saga | 分布式事务的补偿模式 |
| CQRS | CQRS | 命令查询职责分离 |
| 事件溯源 | Event Sourcing | 将状态变化存储为事件序列 |
| DDD | Domain-Driven Design | 领域驱动设计 |
| 限界上下文 | Bounded Context | DDD 中模型的边界 |
| 聚合 | Aggregate | DDD 中数据一致性的边界 |
| mTLS | Mutual TLS | 双向 TLS 认证 |
| JWT | JSON Web Token | JSON 格式的令牌 |
| OAuth2 | OAuth 2.0 | 授权框架 |

## 常用工具和框架

### 服务框架

| 工具 | 语言 | 说明 |
| ---- | ---- | ---- |
| Spring Cloud | Java | 最流行的微服务框架 |
| Spring Cloud Alibaba | Java | 阿里巴巴微服务解决方案 |
| Dubbo | Java | 高性能 RPC 框架 |
| gRPC | 多语言 | Google 开源的 RPC 框架 |
| Go Micro | Go | Go 语言微服务框架 |
| Dapr | 多语言 | 分布式应用运行时 |

### 服务治理

| 工具 | 功能 | 说明 |
| ---- | ---- | ---- |
| Nacos | 注册/配置 | 阿里巴巴开源 |
| Consul | 注册/配置 | HashiCorp 开源 |
| Eureka | 注册 | Netflix 开源 |
| Apollo | 配置 | 携程开源 |
| Sentinel | 限流熔断 | 阿里巴巴开源 |
| Resilience4j | 限流熔断 | 轻量级容错库 |

### API 网关

| 工具 | 说明 |
| ---- | ---- |
| Spring Cloud Gateway | Spring 生态网关 |
| Kong | 高性能 API 网关 |
| APISIX | Apache 开源网关 |
| Nginx | 反向代理/网关 |
| Envoy | 云原生代理 |

### 服务网格

| 工具 | 说明 |
| ---- | ---- |
| Istio | 最流行的服务网格 |
| Linkerd | 轻量级服务网格 |
| Consul Connect | HashiCorp 服务网格 |

### 可观测性

| 工具 | 功能 | 说明 |
| ---- | ---- | ---- |
| Prometheus | 指标 | 时序数据库 |
| Grafana | 可视化 | 监控面板 |
| Jaeger | 追踪 | 分布式追踪 |
| Zipkin | 追踪 | 分布式追踪 |
| ELK Stack | 日志 | 日志聚合分析 |
| SkyWalking | APM | 应用性能监控 |

### 消息队列

| 工具 | 特点 |
| ---- | ---- |
| Kafka | 高吞吐量 |
| RabbitMQ | 功能丰富 |
| RocketMQ | 阿里开源 |
| Pulsar | 云原生 |

### 容器编排

| 工具 | 说明 |
| ---- | ---- |
| Kubernetes | 容器编排平台 |
| Docker | 容器化技术 |
| Helm | K8s 包管理 |

## 常用命令速查

### Docker 命令

```bash
# 构建镜像
docker build -t myapp:v1 .

# 运行容器
docker run -d -p 8080:8080 --name myapp myapp:v1

# 查看日志
docker logs -f myapp

# 进入容器
docker exec -it myapp /bin/sh

# 查看容器
docker ps -a

# 停止/删除容器
docker stop myapp
docker rm myapp
```

### Kubernetes 命令

```bash
# 部署应用
kubectl apply -f deployment.yaml

# 查看 Pod
kubectl get pods -n namespace

# 查看日志
kubectl logs -f pod-name -n namespace

# 进入 Pod
kubectl exec -it pod-name -n namespace -- /bin/sh

# 查看服务
kubectl get svc -n namespace

# 扩缩容
kubectl scale deployment myapp --replicas=3

# 滚动更新
kubectl set image deployment/myapp myapp=myapp:v2

# 回滚
kubectl rollout undo deployment/myapp

# 查看事件
kubectl get events -n namespace
```

### Nacos 命令

```bash
# 启动 Nacos（单机）
sh startup.sh -m standalone

# 注册服务
curl -X POST 'http://localhost:8848/nacos/v1/ns/instance' \
  -d 'serviceName=myapp&ip=127.0.0.1&port=8080'

# 查询服务
curl 'http://localhost:8848/nacos/v1/ns/instance/list?serviceName=myapp'

# 发布配置
curl -X POST 'http://localhost:8848/nacos/v1/cs/configs' \
  -d 'dataId=myapp&group=DEFAULT_GROUP&content=key=value'

# 获取配置
curl 'http://localhost:8848/nacos/v1/cs/configs?dataId=myapp&group=DEFAULT_GROUP'
```

## 常用配置速查

### Spring Cloud Gateway

```yaml
spring:
  cloud:
    gateway:
      routes:
        - id: user-service
          uri: lb://user-service
          predicates:
            - Path=/api/users/**
          filters:
            - StripPrefix=1
            - name: RequestRateLimiter
              args:
                redis-rate-limiter.replenishRate: 10
                redis-rate-limiter.burstCapacity: 20
```

### Nacos 配置

```yaml
spring:
  cloud:
    nacos:
      discovery:
        server-addr: localhost:8848
        namespace: dev
      config:
        server-addr: localhost:8848
        namespace: dev
        file-extension: yaml
```

### Sentinel 配置

```yaml
spring:
  cloud:
    sentinel:
      transport:
        dashboard: localhost:8080
      datasource:
        flow:
          nacos:
            server-addr: localhost:8848
            dataId: ${spring.application.name}-flow-rules
            groupId: SENTINEL_GROUP
            rule-type: flow
```

### Resilience4j 配置

```yaml
resilience4j:
  circuitbreaker:
    instances:
      default:
        slidingWindowSize: 10
        failureRateThreshold: 50
        waitDurationInOpenState: 30s
  retry:
    instances:
      default:
        maxAttempts: 3
        waitDuration: 1s
  ratelimiter:
    instances:
      default:
        limitForPeriod: 10
        limitRefreshPeriod: 1s
```

### OpenTelemetry 配置

```yaml
otel:
  exporter:
    otlp:
      endpoint: http://localhost:4317
  resource:
    attributes:
      service.name: ${spring.application.name}
```

## HTTP 状态码速查

| 状态码 | 含义 | 使用场景 |
| ------ | ---- | ------- |
| 200 | OK | 请求成功 |
| 201 | Created | 资源创建成功 |
| 204 | No Content | 删除成功 |
| 400 | Bad Request | 请求参数错误 |
| 401 | Unauthorized | 未认证 |
| 403 | Forbidden | 无权限 |
| 404 | Not Found | 资源不存在 |
| 409 | Conflict | 资源冲突 |
| 429 | Too Many Requests | 请求过多（限流） |
| 500 | Internal Server Error | 服务器错误 |
| 502 | Bad Gateway | 网关错误 |
| 503 | Service Unavailable | 服务不可用 |
| 504 | Gateway Timeout | 网关超时 |

## 端口速查

| 服务 | 默认端口 |
| ---- | ------- |
| Nacos | 8848 |
| Consul | 8500 |
| Eureka | 8761 |
| Sentinel Dashboard | 8080 |
| Prometheus | 9090 |
| Grafana | 3000 |
| Jaeger UI | 16686 |
| Zipkin | 9411 |
| Elasticsearch | 9200 |
| Kibana | 5601 |
| Kafka | 9092 |
| RabbitMQ | 5672 (AMQP), 15672 (管理) |
| Redis | 6379 |
| MySQL | 3306 |
| PostgreSQL | 5432 |
