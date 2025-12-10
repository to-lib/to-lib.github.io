---
id: service-governance
title: 微服务治理高级
sidebar_label: 服务治理
sidebar_position: 13
---

# 微服务治理高级

> [!TIP]
> **企业级治理方案**: 深入探讨微服务治理的高级主题,包括服务路由、灰度发布、链路追踪等。

## 1. 服务路由

### 基于标签路由

**添加服务标签**:

```yaml
spring:
  cloud:
    nacos:
      discovery:
        metadata:
          version: v1
          region: cn-hangzhou
          env: prod
```

**自定义路由规则**:

```java
@Component
public class TagBasedLoadBalancer implements ReactorLoadBalancer<ServiceInstance> {

    @Override
    public Mono<Response<ServiceInstance>> choose(Request request) {
        return Mono.fromCallable(() -> {
            List<ServiceInstance> instances = getInstances();
            
            // 获取请求中的标签
            String tag = request.getContext().toString();
            
            // 根据标签过滤实例
            List<ServiceInstance> filtered = instances.stream()
                .filter(instance -> matchTag(instance, tag))
                .collect(Collectors.toList());
            
            if (filtered.isEmpty()) {
                filtered = instances;
            }
            
            // 随机选择一个实例
            ServiceInstance instance = filtered.get(
                ThreadLocalRandom.current().nextInt(filtered.size())
            );
            
            return new DefaultResponse(instance);
        });
    }
}
```

### 流量染色

```java
@Component
public class TrafficColorFilter implements GlobalFilter, Ordered {

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        ServerHttpRequest request = exchange.getRequest();
        
        // 提取用户ID或其他标识
        String userId = request.getHeaders().getFirst("User-Id");
        
        // 根据规则染色
        String color = getTrafficColor(userId);
        
        // 将颜色信息传递到下游
        ServerHttpRequest newRequest = request.mutate()
            .header("X-Traffic-Color", color)
            .build();
        
        return chain.filter(exchange.mutate().request(newRequest).build());
    }

    private String getTrafficColor(String userId) {
        // 规则: userId末位为0-4走灰度,5-9走正式
        if (userId != null && !userId.isEmpty()) {
            int lastDigit = Character.getNumericValue(userId.charAt(userId.length() - 1));
            return lastDigit < 5 ? "gray" : "normal";
        }
        return "normal";
    }

    @Override
    public int getOrder() {
        return -100;
    }
}
```

## 2. 灰度发布方案

### 基于权重的灰度

```java
@Configuration
public class GrayReleaseConfig {

    @Bean
    public ReactorLoadBalancer<ServiceInstance> weightBasedLoadBalancer(
            ObjectProvider<ServiceInstanceListSupplier> provider) {
        return new WeightBasedLoadBalancer(provider, "user-service");
    }
}

class WeightBasedLoadBalancer implements ReactorLoadBalancer<ServiceInstance> {

    @Override
    public Mono<Response<ServiceInstance>> choose(Request request) {
        return serviceInstanceListSupplierProvider.getIfAvailable().get()
            .next()
            .map(this::processInstanceResponse);
    }

    private Response<ServiceInstance> processInstanceResponse(
            List<ServiceInstance> instances) {
        
        if (instances.isEmpty()) {
            return new EmptyResponse();
        }

        // 计算权重总和
        int totalWeight = instances.stream()
            .mapToInt(this::getWeight)
            .sum();

        // 随机选择
        int randomWeight = ThreadLocalRandom.current().nextInt(totalWeight);
        int currentWeight = 0;

        for (ServiceInstance instance : instances) {
            currentWeight += getWeight(instance);
            if (randomWeight < currentWeight) {
                return new DefaultResponse(instance);
            }
        }

        return new DefaultResponse(instances.get(0));
    }

    private int getWeight(ServiceInstance instance) {
        String weight = instance.getMetadata().get("weight");
        return weight != null ? Integer.parseInt(weight) : 100;
    }
}
```

### A/B Testing

```java
@Component
public class ABTestingFilter implements GlobalFilter {

    private static final String AB_TEST_HEADER = "X-AB-Test";

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        String userId = exchange.getRequest().getHeaders().getFirst("User-Id");
        
        // A/B分组策略
        String group = getABTestGroup(userId);
        
        // 添加分组标识到请求头
        ServerHttpRequest request = exchange.getRequest().mutate()
            .header(AB_TEST_HEADER, group)
            .build();

        return chain.filter(exchange.mutate().request(request).build());
    }

    private String getABTestGroup(String userId) {
        if (userId == null) {
            return "A";
        }
        
        // 根据用户ID hash分组
        int hash = userId.hashCode();
        return Math.abs(hash % 2) == 0 ? "A" : "B";
    }
}
```

## 3. 链路追踪 (Sleuth + Zipkin)

### 集成 Sleuth

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

```yaml
spring:
  zipkin:
    base-url: http://localhost:9411
    sender:
      type: web
  sleuth:
    sampler:
      probability: 1.0  # 100% 采样
```

### 自定义 Span

```java
@Service
public class UserService {

    @Autowired
    private Tracer tracer;

    public User getUser(Long id) {
        // 创建自定义 Span
        Span span = tracer.nextSpan().name("getUserFromCache");
        
        try (Tracer.SpanInScope ws = tracer.withSpan(span.start())) {
            // 添加标签
            span.tag("user.id", id.toString());
            
            // 业务逻辑
            User user = getUserFromCache(id);
            
            if (user == null) {
                span.tag("cache.miss", "true");
                user = getUserFromDatabase(id);
            } else {
                span.tag("cache.hit", "true");
            }
            
            return user;
        } finally {
            span.end();
        }
    }
}
```

### 跨服务追踪

Sleuth 自动在 HTTP 请求头中传递追踪信息:

```
X-B3-TraceId: 80f198ee56343ba864fe8b2a57d3eff7
X-B3-SpanId: e457b5a2e4d86bd1
X-B3-ParentSpanId: 05e3ac9a4f6e3b90
X-B3-Sampled: 1
```

## 4. 服务限流降级

### 多级限流

**网关层限流**:

```java
@Configuration
public class GatewayRateLimiterConfig {

    @Bean
    public KeyResolver userKeyResolver() {
        return exchange -> Mono.just(
            exchange.getRequest().getHeaders().getFirst("User-Id")
        );
    }

    @Bean
    public KeyResolver ipKeyResolver() {
        return exchange -> Mono.just(
            exchange.getRequest().getRemoteAddress().getAddress().getHostAddress()
        );
    }
}
```

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
                key-resolver: '#{@userKeyResolver}'
                redis-rate-limiter.replenishRate: 10
                redis-rate-limiter.burstCapacity: 20
```

**服务层限流**:

```java
@Service
public class OrderService {

    @SentinelResource(
        value = "createOrder",
        blockHandler = "handleBlock",
        fallback = "handleFallback"
    )
    public Order createOrder(Order order) {
        return orderRepository.save(order);
    }

    public Order handleBlock(Order order, BlockException ex) {
        return new Order("限流中,请稍后重试");
    }

    public Order handleFallback(Order order, Throwable ex) {
        return new Order("服务降级");
    }
}
```

### 熔断策略

```java
@Configuration
public class CircuitBreakerConfig {

    @Bean
    public Customizer<ReactiveResilience4JCircuitBreakerFactory> defaultCustomizer() {
        return factory -> factory.configureDefault(id -> new Resilience4JConfigBuilder(id)
            .circuitBreakerConfig(CircuitBreakerConfig.custom()
                .slidingWindowSize(10)
                .failureRateThreshold(50)
                .waitDurationInOpenState(Duration.ofSeconds(10))
                .permittedNumberOfCallsInHalfOpenState(3)
                .build())
            .timeLimiterConfig(TimeLimiterConfig.custom()
                .timeoutDuration(Duration.ofSeconds(4))
                .build())
            .build());
    }
}
```

## 5. 服务降级

### 自动降级

```java
@Service
public class ProductService {

    @Autowired
    private RemoteProductClient productClient;

    @Autowired
    private RedisTemplate redisTemplate;

    @HystrixCommand(
        fallbackMethod = "getProductFromCache",
        commandProperties = {
            @HystrixProperty(name = "execution.isolation.thread.timeoutInMilliseconds", value = "3000")
        }
    )
    public Product getProduct(Long id) {
        return productClient.getProduct(id);
    }

    public Product getProductFromCache(Long id) {
        // 从缓存获取
        Object cached = redisTemplate.opsForValue().get("product:" + id);
        if (cached != null) {
            return (Product) cached;
        }
        
        // 返回降级数据
        return new Product(id, "商品暂时不可用", BigDecimal.ZERO);
    }
}
```

### 手动降级开关

```java
@Component
public class ManualDegradeSwitch {

    private final AtomicBoolean degradeSwitch = new AtomicBoolean(false);

    public boolean isDegraded() {
        return degradeSwitch.get();
    }

    public void enableDegrade() {
        degradeSwitch.set(true);
        log.warn("手动开启降级");
    }

    public void disableDegrade() {
        degradeSwitch.set(false);
        log.info("关闭降级");
    }
}

@Service
public class OrderService {

    @Autowired
    private ManualDegradeSwitch degradeSwitch;

    public Order createOrder(Order order) {
        if (degradeSwitch.isDegraded()) {
            return createOrderDegraded(order);
        }
        return createOrderNormal(order);
    }
}
```

## 6. 服务网格探索 (Service Mesh)

### Istio 简介

**核心功能**:

- 流量管理
- 安全
- 可观测性
- 策略执行

**架构**:

```
Data Plane (Envoy Sidecars)
    ↓
Control Plane (Istiod)
    ↓
配置、发现、证书管理
```

### 与 Spring Cloud Alibaba 集成

**部署方式**:

1. **使用 Istio 替代部分组件**:
   - Istio Pilot 替代 Nacos 服务发现
   - Istio Mixer替代 Sentinel

2. **混合使用**:
   - Nacos 配置管理
   - Istio 流量管理
   - Sentinel 限流

**示例配置**:

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: user-service
spec:
  hosts:
    - user-service
  http:
    - match:
        - headers:
            version:
              exact: v2
      route:
        - destination:
            host: user-service
            subset: v2
    - route:
        - destination:
            host: user-service
            subset: v1
```

## 7. 最佳实践总结

### 治理策略选择

| 场景         | 推荐方案          |
| ------------ | ----------------- |
| 灰度发布     | 基于权重 + 标签   |
| A/B Testing  | 流量染色          |
| 链路追踪     | Sleuth + Zipkin   |
| 服务限流     | 多级限流          |
| 服务降级     | 自动 + 手动开关   |

### 监控告警

- 关键指标监控
- 异常告警
- 容量规划

---

**关键要点**:

- 服务治理需要全链路考虑
- 灰度发布降低风险
- 链路追踪定位问题
- 多级防护保证稳定性

**下一步**: 学习 [版本升级指南](/docs/springcloud-alibaba/migration-guide)
