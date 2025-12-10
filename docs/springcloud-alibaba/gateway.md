---
id: gateway
title: Spring Cloud Gateway 集成
sidebar_label: Gateway 网关
sidebar_position: 8
---

# Spring Cloud Gateway 集成

> [!TIP]
> **统一网关**: Spring Cloud Gateway 作为微服务的统一入口,与 Nacos、Sentinel 完美集成,提供路由、负载均衡、限流等功能。

## 1. Gateway 简介

**Spring Cloud Gateway** 是 Spring Cloud 官方推出的网关组件,基于 Spring WebFlux 构建。

### 核心功能

- **路由管理** - 将请求路由到具体的微服务
- **负载均衡** - 自动负载均衡
- **过滤器** - 请求和响应的处理
- **限流熔断** - 与 Sentinel 集成
- **动态路由** - 与 Nacos 集成,支持动态配置

## 2. 快速开始

### 添加依赖

```xml
<dependencies>
    <!-- Spring Cloud Gateway -->
    <dependency>
        <groupId>org.springframework.cloud</groupId>
        <artifactId>spring-cloud-starter-gateway</artifactId>
    </dependency>

    <!-- Nacos 服务发现 -->
    <dependency>
        <groupId>com.alibaba.cloud</groupId>
        <artifactId>spring-cloud-starter-alibaba-nacos-discovery</artifactId>
    </dependency>

    <!-- Nacos 配置中心 -->
    <dependency>
        <groupId>com.alibaba.cloud</groupId>
        <artifactId>spring-cloud-starter-alibaba-nacos-config</artifactId>
    </dependency>

    <!-- LoadBalancer -->
    <dependency>
        <groupId>org.springframework.cloud</groupId>
        <artifactId>spring-cloud-starter-loadbalancer</artifactId>
    </dependency>
</dependencies>
```

### 基础配置

```yaml
server:
  port: 8080

spring:
  application:
    name: api-gateway
  cloud:
    nacos:
      discovery:
        server-addr: localhost:8848
    gateway:
      # 启用服务发现路由
      discovery:
        locator:
          enabled: true
          # 服务名小写
          lower-case-service-id: true
      routes:
        # 用户服务路由
        - id: user-service
          uri: lb://user-service
          predicates:
            - Path=/api/users/**
          filters:
            - StripPrefix=1

        # 订单服务路由
        - id: order-service
          uri: lb://order-service
          predicates:
            - Path=/api/orders/**
          filters:
            - StripPrefix=1
```

### 启动类

```java
package com.example.gateway;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;

@SpringBootApplication
@EnableDiscoveryClient
public class GatewayApplication {
    public static void main(String[] args) {
        SpringApplication.run(GatewayApplication.class, args);
    }
}
```

## 3. 路由配置详解

### 路由组成

```yaml
spring:
  cloud:
    gateway:
      routes:
        - id: route-id              # 路由唯一标识
          uri: lb://service-name    # 目标服务 (lb = LoadBalancer)
          predicates:               # 断言,判断是否匹配
            - Path=/api/**
          filters:                  # 过滤器,处理请求/响应
            - StripPrefix=1
          order: 1                  # 路由优先级
```

### 常用断言 (Predicates)

**Path 路径匹配**:

```yaml
predicates:
  - Path=/api/users/{id}
```

**Method 方法匹配**:

```yaml
predicates:
  - Method=GET,POST
```

**Header 请求头匹配**:

```yaml
predicates:
  - Header=X-Request-Id, \d+
```

**Query 参数匹配**:

```yaml
predicates:
  - Query=token
```

**时间匹配**:

```yaml
predicates:
  - After=2024-01-01T00:00:00+08:00[Asia/Shanghai]
  - Before=2024-12-31T23:59:59+08:00[Asia/Shanghai]
```

**组合使用**:

```yaml
predicates:
  - Path=/api/users/**
  - Method=GET
  - Header=Authorization
```

### 常用过滤器 (Filters)

**StripPrefix** - 去除路径前缀:

```yaml
# /api/users/1 -> /users/1
filters:
  - StripPrefix=1
```

**AddRequestHeader** - 添加请求头:

```yaml
filters:
  - AddRequestHeader=X-Request-Source, gateway
```

**AddRequestParameter** - 添加请求参数:

```yaml
filters:
  - AddRequestParameter=source, gateway
```

**AddResponseHeader** - 添加响应头:

```yaml
filters:
  - AddResponseHeader=X-Response-Gateway, true
```

**PrefixPath** - 添加路径前缀:

```yaml
# /users/1 -> /api/users/1
filters:
  - PrefixPath=/api
```

**RewritePath** - 重写路径:

```yaml
# /red/blue -> /blue
filters:
  - RewritePath=/red(?<segment>/?.*), $\{segment}
```

## 4. 与 Nacos 动态路由

### 配置在 Nacos

在 Nacos 配置中心创建配置:

**Data ID**: `api-gateway.yaml`

```yaml
spring:
  cloud:
    gateway:
      routes:
        - id: user-service-route
          uri: lb://user-service
          predicates:
            - Path=/api/users/**
          filters:
            - StripPrefix=1
            - name: RequestRateLimiter
              args:
                redis-rate-limiter.replenishRate: 10
                redis-rate-limiter.burstCapacity: 20

        - id: order-service-route
          uri: lb://order-service
          predicates:
            - Path=/api/orders/**
          filters:
            - StripPrefix=1
```

### 动态刷新

修改 Nacos 配置后,网关会自动加载新的路由配置。

## 5. 与 Sentinel 集成限流

### 添加依赖

```xml
<dependency>
    <groupId>com.alibaba.cloud</groupId>
    <artifactId>spring-cloud-alibaba-sentinel-gateway</artifactId>
</dependency>

<dependency>
    <groupId>com.alibaba.cloud</groupId>
    <artifactId>spring-cloud-starter-alibaba-sentinel</artifactId>
</dependency>
```

### 配置

```yaml
spring:
  cloud:
    sentinel:
      transport:
        dashboard: localhost:8080
      # 网关限流配置
      scg:
        fallback:
          mode: response
          response-status: 429
          response-body: '{"code":429,"message":"Too Many Requests"}'
```

### 网关限流规则

```java
package com.example.gateway.config;

import com.alibaba.csp.sentinel.adapter.gateway.common.SentinelGatewayConstants;
import com.alibaba.csp.sentinel.adapter.gateway.common.api.ApiDefinition;
import com.alibaba.csp.sentinel.adapter.gateway.common.api.ApiPathPredicateItem;
import com.alibaba.csp.sentinel.adapter.gateway.common.api.ApiPredicateItem;
import com.alibaba.csp.sentinel.adapter.gateway.common.api.GatewayApiDefinitionManager;
import com.alibaba.csp.sentinel.adapter.gateway.common.rule.GatewayFlowRule;
import com.alibaba.csp.sentinel.adapter.gateway.common.rule.GatewayRuleManager;
import org.springframework.context.annotation.Configuration;

import javax.annotation.PostConstruct;
import java.util.HashSet;
import java.util.Set;

@Configuration
public class GatewaySentinelConfig {

    @PostConstruct
    public void init() {
        initCustomizedApis();
        initGatewayRules();
    }

    // 定义 API 分组
    private void initCustomizedApis() {
        Set<ApiDefinition> definitions = new HashSet<>();
        
        // 用户服务 API
        ApiDefinition api1 = new ApiDefinition("user_api")
            .setPredicateItems(new HashSet<ApiPredicateItem>() {{
                add(new ApiPathPredicateItem().setPattern("/api/users/**")
                    .setMatchStrategy(SentinelGatewayConstants.URL_MATCH_STRATEGY_PREFIX));
            }});
        
        // 订单服务 API
        ApiDefinition api2 = new ApiDefinition("order_api")
            .setPredicateItems(new HashSet<ApiPredicateItem>() {{
                add(new ApiPathPredicateItem().setPattern("/api/orders/**")
                    .setMatchStrategy(SentinelGatewayConstants.URL_MATCH_STRATEGY_PREFIX));
            }});
        
        definitions.add(api1);
        definitions.add(api2);
        
        GatewayApiDefinitionManager.loadApiDefinitions(definitions);
    }

    // 配置限流规则
    private void initGatewayRules() {
        Set<GatewayFlowRule> rules = new HashSet<>();
        
        // 用户服务限流: 每秒 10 个请求
        rules.add(new GatewayFlowRule("user_api")
            .setCount(10)
            .setIntervalSec(1)
        );
        
        // 订单服务限流: 每秒 5 个请求
        rules.add(new GatewayFlowRule("order_api")
            .setCount(5)
            .setIntervalSec(1)
        );
        
        GatewayRuleManager.loadRules(rules);
    }
}
```

## 6. 自定义过滤器

### 全局过滤器

```java
package com.example.gateway.filter;

import org.springframework.cloud.gateway.filter.GatewayFilterChain;
import org.springframework.cloud.gateway.filter.GlobalFilter;
import org.springframework.core.Ordered;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

@Component
public class AuthFilter implements GlobalFilter, Ordered {

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        ServerHttpRequest request = exchange.getRequest();
        
        // 获取请求头中的 token
        String token = request.getHeaders().getFirst("Authorization");
        
        // 验证 token (简化示例)
        if (token == null || !token.startsWith("Bearer ")) {
            exchange.getResponse().setStatusCode(
                org.springframework.http.HttpStatus.UNAUTHORIZED
            );
            return exchange.getResponse().setComplete();
        }
        
        // token 验证通过,继续执行
        return chain.filter(exchange);
    }

    @Override
    public int getOrder() {
        return -100;  // 优先级,越小越先执行
    }
}
```

### 局部过滤器

```java
package com.example.gateway.filter;

import org.springframework.cloud.gateway.filter.GatewayFilter;
import org.springframework.cloud.gateway.filter.factory.AbstractGatewayFilterFactory;
import org.springframework.stereotype.Component;

@Component
public class LogGatewayFilterFactory 
    extends AbstractGatewayFilterFactory<LogGatewayFilterFactory.Config> {

    public LogGatewayFilterFactory() {
        super(Config.class);
    }

    @Override
    public GatewayFilter apply(Config config) {
        return (exchange, chain) -> {
            System.out.println("Pre Filter: " + config.getMessage());
            
            return chain.filter(exchange).then(Mono.fromRunnable(() -> {
                System.out.println("Post Filter: " + config.getMessage());
            }));
        };
    }

    public static class Config {
        private String message;

        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
    }
}
```

使用:

```yaml
filters:
  - Log=Hello Gateway
```

## 7. 跨域配置

```yaml
spring:
  cloud:
    gateway:
      globalcors:
        cors-configurations:
          '[/**]':
            allowed-origins: "*"
            allowed-methods:
              - GET
              - POST
              - PUT
              - DELETE
            allowed-headers: "*"
            allow-credentials: true
            max-age: 3600
```

## 8. 熔断降级

### 使用 Sentinel

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
            - name: Sentinel
              args:
                mode: response
                fallback-response-status: 503
                fallback-response-body: '{"message":"服务暂时不可用"}'
```

## 9. 请求日志

```java
package com.example.gateway.filter;

import lombok.extern.slf4j.Slf4j;
import org.springframework.cloud.gateway.filter.GatewayFilterChain;
import org.springframework.cloud.gateway.filter.GlobalFilter;
import org.springframework.core.Ordered;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

@Slf4j
@Component
public class LoggingFilter implements GlobalFilter, Ordered {

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        ServerHttpRequest request = exchange.getRequest();
        
        long startTime = System.currentTimeMillis();
        
        log.info("请求路径: {}", request.getURI().getPath());
        log.info("请求方法: {}", request.getMethod());
        log.info("请求参数: {}", request.getQueryParams());
        
        return chain.filter(exchange).then(Mono.fromRunnable(() -> {
            long duration = System.currentTimeMillis() - startTime;
            log.info("响应状态: {}", exchange.getResponse().getStatusCode());
            log.info("请求耗时: {}ms", duration);
        }));
    }

    @Override
    public int getOrder() {
        return Ordered.LOWEST_PRECEDENCE;
    }
}
```

## 10. 最佳实践

### 路由配置建议

- **使用服务名**: `uri: lb://service-name` 而不是固定 IP
- **合理分组**: 将相关的路由放在一起
- **设置优先级**: 使用 `order` 控制路由匹配顺序

### 过滤器使用建议

- **全局过滤器**: 用于通用逻辑 (认证、日志等)
- **局部过滤器**: 用于特定路由的处理
- **注意顺序**: 通过 `Ordered` 接口控制执行顺序

### 性能优化

- **合理限流**: 根据实际情况设置限流规则
- **连接池配置**: 调整 WebClient 连接池大小
- **超时配置**: 设置合理的超时时间

```yaml
spring:
  cloud:
    gateway:
      httpclient:
        # 连接超时
        connect-timeout: 3000
        # 响应超时
        response-timeout: 5s
        # 连接池配置
        pool:
          max-connections: 500
          max-idle-time: 30s
```

## 11. 完整示例

```yaml
server:
  port: 8080

spring:
  application:
    name: api-gateway
  cloud:
    nacos:
      discovery:
        server-addr: localhost:8848
      config:
        server-addr: localhost:8848
        file-extension: yaml
    sentinel:
      transport:
        dashboard: localhost:8080
      scg:
        fallback:
          mode: response
          response-status: 429
    gateway:
      discovery:
        locator:
          enabled: true
          lower-case-service-id: true
      routes:
        # 用户服务
        - id: user-service
          uri: lb://user-service
          predicates:
            - Path=/api/users/**
          filters:
            - StripPrefix=1
            - AddRequestHeader=X-Gateway, api-gateway
          order: 1

        # 订单服务
        - id: order-service
          uri: lb://order-service
          predicates:
            - Path=/api/orders/**
          filters:
            - StripPrefix=1
            - AddRequestHeader=X-Gateway, api-gateway
          order: 2

      # 全局 CORS
      globalcors:
        cors-configurations:
          '[/**]':
            allowed-origins: "*"
            allowed-methods: "*"
            allowed-headers: "*"

      # 超时配置
      httpclient:
        connect-timeout: 3000
        response-timeout: 5s
```

## 12. 总结

| 功能     | 说明           |
| -------- | -------------- |
| 路由管理 | 统一请求入口   |
| 负载均衡 | 自动负载均衡   |
| 限流熔断 | 与 Sentinel 集成|
| 动态配置 | 与 Nacos 集成  |
| 过滤器   | 灵活的请求处理 |

---

**关键要点**:

- Gateway 是微服务的统一入口
- 与 Nacos 集成实现动态路由
- 与 Sentinel 集成实现限流熔断
- 通过过滤器实现认证、日志等通用功能

**下一步**: 学习 [最佳实践](/docs/springcloud-alibaba/best-practices)
