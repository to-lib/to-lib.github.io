---
title: Gateway API网关
sidebar_label: Gateway
sidebar_position: 5
---

# Gateway API 网关

> [!TIP] > **统一入口**: Spring Cloud Gateway 是新一代 API 网关，基于 Spring WebFlux，提供路由、过滤、限流等功能，是微服务架构的统一入口。

## 1. Gateway 简介

### 什么是 Spring Cloud Gateway？

**Spring Cloud Gateway** 是 Spring Cloud 官方推出的第二代网关框架，用于替代 Zuul。它基于 Spring 5、Spring Boot 2 和 Project Reactor 构建。

### 核心概念

- **Route（路由）** - 网关的基本构建块，包含 ID、目标 URI、断言集合和过滤器集合
- **Predicate（断言）** - 匹配来自 HTTP 请求的任何内容，如请求头、请求参数
- **Filter（过滤器）** - 可以在发送请求前后修改请求和响应

### Gateway vs Zuul

| 特性     | Gateway           | Zuul 1.x          |
| -------- | ----------------- | ----------------- |
| 底层实现 | WebFlux（响应式） | Servlet（阻塞式） |
| 性能     | 高                | 中                |
| 限流     | 内置              | 需要自己实现      |
| 维护状态 | 活跃开发          | 停止维护          |

## 2. 快速开始

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-gateway</artifactId>
</dependency>

<!-- 如果需要服务发现 -->
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
</dependency>
```

### 基本配置

```yaml
server:
  port: 8080

spring:
  application:
    name: api-gateway
  cloud:
    gateway:
      routes:
        # 用户服务路由
        - id: user-service
          uri: http://localhost:8081
          predicates:
            - Path=/api/users/**
          filters:
            - StripPrefix=1

        # 订单服务路由
        - id: order-service
          uri: http://localhost:8082
          predicates:
            - Path=/api/orders/**
          filters:
            - StripPrefix=1
```

## 3. 路由配置

### 基于 YAML 配置

```yaml
spring:
  cloud:
    gateway:
      routes:
        - id: user-service
          # 目标服务地址
          uri: lb://user-service
          # 断言配置
          predicates:
            - Path=/api/users/**
            - Method=GET,POST
            - Header=X-Request-Id, \d+
          # 过滤器配置
          filters:
            - StripPrefix=1
            - AddRequestHeader=X-Gateway, gateway
```

### 基于 Java 配置

```java
@Configuration
public class GatewayConfig {

    @Bean
    public RouteLocator customRouteLocator(RouteLocatorBuilder builder) {
        return builder.routes()
            .route("user-service", r -> r
                .path("/api/users/**")
                .filters(f -> f
                    .stripPrefix(1)
                    .addRequestHeader("X-Gateway", "gateway"))
                .uri("lb://user-service"))
            .route("order-service", r -> r
                .path("/api/orders/**")
                .filters(f -> f.stripPrefix(1))
                .uri("lb://order-service"))
            .build();
    }
}
```

### 基于服务发现

```yaml
spring:
  cloud:
    gateway:
      discovery:
        locator:
          # 启用服务发现路由
          enabled: true
          # 服务名小写
          lower-case-service-id: true
```

访问方式：`http://gateway:8080/user-service/users/1`

## 4. 断言（Predicates）

### 路径断言

```yaml
predicates:
  # 精确匹配
  - Path=/api/users
  # 通配符匹配
  - Path=/api/users/**
  # 正则表达式
  - Path=/api/users/{id:\d+}
```

### 请求方法断言

```yaml
predicates:
  - Method=GET,POST
```

### 请求头断言

```yaml
predicates:
  # 存在某个头
  - Header=X-Request-Id
  # 头的值匹配正则
  - Header=X-Request-Id, \d+
```

### 请求参数断言

```yaml
predicates:
  # 存在某个参数
  - Query=token
  # 参数值匹配正则
  - Query=name, [a-z]+
```

### 时间断言

```yaml
predicates:
  # 在某个时间之后
  - After=2023-01-01T00:00:00+08:00[Asia/Shanghai]
  # 在某个时间之前
  - Before=2024-01-01T00:00:00+08:00[Asia/Shanghai]
  # 在某个时间范围内
  - Between=2023-01-01T00:00:00+08:00[Asia/Shanghai], 2024-01-01T00:00:00+08:00[Asia/Shanghai]
```

### 自定义断言

```java
@Component
public class CustomRoutePredicateFactory
    extends AbstractRoutePredicateFactory<CustomRoutePredicateFactory.Config> {

    public CustomRoutePredicateFactory() {
        super(Config.class);
    }

    @Override
    public Predicate<ServerWebExchange> apply(Config config) {
        return exchange -> {
            String userType = exchange.getRequest()
                .getHeaders()
                .getFirst("User-Type");
            return config.getUserType().equals(userType);
        };
    }

    public static class Config {
        private String userType;
        // getter/setter
    }
}
```

## 5. 过滤器（Filters）

### 内置过滤器

**AddRequestHeader** - 添加请求头

```yaml
filters:
  - AddRequestHeader=X-Request-Source, gateway
```

**AddRequestParameter** - 添加请求参数

```yaml
filters:
  - AddRequestParameter=source, gateway
```

**AddResponseHeader** - 添加响应头

```yaml
filters:
  - AddResponseHeader=X-Response-Time, ${timestamp}
```

**StripPrefix** - 移除路径前缀

```yaml
# /api/users/1 -> /users/1
filters:
  - StripPrefix=1
```

**PrefixPath** - 添加路径前缀

```yaml
# /users/1 -> /api/users/1
filters:
  - PrefixPath=/api
```

**RewritePath** - 重写路径

```yaml
filters:
  - RewritePath=/api/(?<segment>.*), /${segment}
```

**SetStatus** - 设置响应状态码

```yaml
filters:
  - SetStatus=401
```

**Retry** - 重试

```yaml
filters:
  - name: Retry
    args:
      retries: 3
      statuses: BAD_GATEWAY, GATEWAY_TIMEOUT
      methods: GET,POST
      backoff:
        firstBackoff: 10ms
        maxBackoff: 50ms
        factor: 2
```

### 全局过滤器

```java
@Component
public class LoggingGlobalFilter implements GlobalFilter, Ordered {

    private static final Logger log = LoggerFactory.getLogger(LoggingGlobalFilter.class);

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        ServerHttpRequest request = exchange.getRequest();
        log.info("请求路径: {}", request.getPath());
        log.info("请求方法: {}", request.getMethod());

        return chain.filter(exchange).then(Mono.fromRunnable(() -> {
            ServerHttpResponse response = exchange.getResponse();
            log.info("响应状态码: {}", response.getStatusCode());
        }));
    }

    @Override
    public int getOrder() {
        return -1;  // 优先级，数字越小优先级越高
    }
}
```

### 自定义过滤器工厂

```java
@Component
public class AuthGatewayFilterFactory
    extends AbstractGatewayFilterFactory<AuthGatewayFilterFactory.Config> {

    public AuthGatewayFilterFactory() {
        super(Config.class);
    }

    @Override
    public GatewayFilter apply(Config config) {
        return (exchange, chain) -> {
            ServerHttpRequest request = exchange.getRequest();
            String token = request.getHeaders().getFirst("Authorization");

            if (token == null || !config.getValidToken().equals(token)) {
                exchange.getResponse().setStatusCode(HttpStatus.UNAUTHORIZED);
                return exchange.getResponse().setComplete();
            }

            return chain.filter(exchange);
        };
    }

    public static class Config {
        private String validToken;
        // getter/setter
    }
}
```

## 6. 限流

### 使用 RequestRateLimiter

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
            - name: RequestRateLimiter
              args:
                # 令牌桶填充速率（每秒）
                redis-rate-limiter.replenishRate: 10
                # 令牌桶容量
                redis-rate-limiter.burstCapacity: 20
                # 限流 key 解析器
                key-resolver: "#{@ipKeyResolver}"

  redis:
    host: localhost
    port: 6379
```

### 自定义 KeyResolver

```java
@Configuration
public class KeyResolverConfig {

    // 基于 IP 限流
    @Bean
    public KeyResolver ipKeyResolver() {
        return exchange -> Mono.just(
            exchange.getRequest()
                .getRemoteAddress()
                .getAddress()
                .getHostAddress()
        );
    }

    // 基于用户 ID 限流
    @Bean
    public KeyResolver userKeyResolver() {
        return exchange -> Mono.just(
            exchange.getRequest()
                .getHeaders()
                .getFirst("User-Id")
        );
    }

    // 基于接口路径限流
    @Bean
    public KeyResolver pathKeyResolver() {
        return exchange -> Mono.just(
            exchange.getRequest().getPath().value()
        );
    }
}
```

## 7. 熔断

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-circuitbreaker-reactor-resilience4j</artifactId>
</dependency>
```

### 配置熔断

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
            - name: CircuitBreaker
              args:
                name: userServiceCB
                fallbackUri: forward:/fallback/user
```

### 降级处理

```java
@RestController
public class FallbackController {

    @GetMapping("/fallback/user")
    public Mono<Map<String, Object>> userFallback() {
        Map<String, Object> result = new HashMap<>();
        result.put("code", 503);
        result.put("message", "用户服务暂时不可用");
        return Mono.just(result);
    }
}
```

## 8. 跨域配置

```yaml
spring:
  cloud:
    gateway:
      globalcors:
        cors-configurations:
          "[/**]":
            allowed-origins: "http://localhost:3000"
            allowed-methods:
              - GET
              - POST
              - PUT
              - DELETE
            allowed-headers: "*"
            allow-credentials: true
            max-age: 3600
```

## 9. 实战示例

### 统一认证网关

```java
@Component
public class AuthFilter implements GlobalFilter, Ordered {

    private static final List<String> WHITE_LIST = Arrays.asList(
        "/api/auth/login",
        "/api/auth/register"
    );

    @Autowired
    private JwtTokenUtil jwtTokenUtil;

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        ServerHttpRequest request = exchange.getRequest();
        String path = request.getPath().value();

        // 白名单放行
        if (WHITE_LIST.contains(path)) {
            return chain.filter(exchange);
        }

        // 获取 token
        String token = request.getHeaders().getFirst("Authorization");
        if (token == null || !token.startsWith("Bearer ")) {
            exchange.getResponse().setStatusCode(HttpStatus.UNAUTHORIZED);
            return exchange.getResponse().setComplete();
        }

        // 验证 token
        token = token.substring(7);
        if (!jwtTokenUtil.validateToken(token)) {
            exchange.getResponse().setStatusCode(HttpStatus.UNAUTHORIZED);
            return exchange.getResponse().setComplete();
        }

        // 将用户信息添加到请求头
        String userId = jwtTokenUtil.getUserIdFromToken(token);
        ServerHttpRequest mutatedRequest = request.mutate()
            .header("X-User-Id", userId)
            .build();

        return chain.filter(exchange.mutate().request(mutatedRequest).build());
    }

    @Override
    public int getOrder() {
        return -100;
    }
}
```

## 10. 最佳实践

### 路由配置

- 使用服务发现自动路由
- 合理使用断言减少匹配开销
- 路由配置集中管理

### 过滤器使用

- 全局过滤器处理通用逻辑（认证、日志等）
- 局部过滤器处理特定路由逻辑
- 注意过滤器执行顺序

### 限流策略

- 根据实际情况选择限流粒度（IP、用户、接口）
- 合理设置令牌桶参数
- 使用 Redis 集群保证高可用

### 熔断降级

- 为关键服务配置熔断
- 提供友好的降级响应
- 监控熔断器状态

## 11. 总结

| 功能   | 说明             |
| ------ | ---------------- |
| 路由   | 根据规则转发请求 |
| 断言   | 匹配请求条件     |
| 过滤器 | 修改请求和响应   |
| 限流   | 保护后端服务     |
| 熔断   | 快速失败降级     |

---

**关键要点**：

- Gateway 基于响应式编程，性能优于 Zuul
- 断言用于匹配请求，过滤器用于处理请求
- 支持限流、熔断等高级功能
- 可以作为统一认证网关

**下一步**：学习 [Feign 声明式调用](/docs/springcloud/feign)
