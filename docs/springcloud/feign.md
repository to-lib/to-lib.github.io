---
title: Feign 声明式HTTP客户端
sidebar_label: Feign
sidebar_position: 6
---

# Feign 声明式 HTTP 客户端

> [!TIP] > **简化服务调用**: Feign 通过声明式接口简化了 HTTP 客户端的开发，集成了 Ribbon 负载均衡和 Hystrix 熔断器。

## 1. Feign 简介

### 什么是 Feign？

**Feign** 是一个声明式的 Web 服务客户端，只需创建一个接口并添加注解即可完成对 Web 服务的调用。

### 核心特性

- **声明式** - 通过接口和注解定义服务调用
- **集成 Ribbon** - 自动实现负载均衡
- **集成 Hystrix** - 支持熔断降级
- **可插拔** - 支持多种编码器、解码器
- **日志** - 提供详细的请求日志

## 2. 快速开始

### 添加依赖

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-openfeign</artifactId>
</dependency>
```

### 启用 Feign

```java
@SpringBootApplication
@EnableFeignClients
public class OrderServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(OrderServiceApplication.class, args);
    }
}
```

### 定义 Feign 客户端

```java
@FeignClient(name = "user-service")
public interface UserClient {

    @GetMapping("/users/{id}")
    User getUser(@PathVariable("id") Long id);

    @GetMapping("/users")
    List<User> listUsers(@RequestParam("page") int page);

    @PostMapping("/users")
    User createUser(@RequestBody User user);

    @PutMapping("/users/{id}")
    User updateUser(@PathVariable("id") Long id, @RequestBody User user);

    @DeleteMapping("/users/{id}")
    void deleteUser(@PathVariable("id") Long id);
}
```

### 使用 Feign 客户端

```java
@Service
public class OrderService {

    @Autowired
    private UserClient userClient;

    public Order createOrder(Long userId, Order order) {
        // 调用用户服务
        User user = userClient.getUser(userId);
        order.setUser(user);
        // 创建订单...
        return order;
    }
}
```

## 3. 配置

### 基本配置

```yaml
feign:
  client:
    config:
      # 默认配置，对所有 Feign 客户端生效
      default:
        # 连接超时时间（毫秒）
        connect-timeout: 5000
        # 读取超时时间（毫秒）
        read-timeout: 5000
        # 日志级别
        logger-level: full

      # 针对特定服务的配置
      user-service:
        connect-timeout: 3000
        read-timeout: 3000
```

### Java 配置

```java
@Configuration
public class FeignConfig {

    @Bean
    public Logger.Level feignLoggerLevel() {
        return Logger.Level.FULL;
    }

    @Bean
    public RequestInterceptor requestInterceptor() {
        return template -> {
            // 添加请求头
            template.header("X-Source", "order-service");
        };
    }

    @Bean
    public Retryer feignRetryer() {
        // 最大重试次数5次，初始间隔100ms，最大间隔1s
        return new Retryer.Default(100, 1000, 5);
    }
}
```

使用配置：

```java
@FeignClient(name = "user-service", configuration = FeignConfig.class)
public interface UserClient {
    // ...
}
```

## 4. 请求和响应处理

### 请求参数

**路径参数**

```java
@GetMapping("/users/{id}")
User getUser(@PathVariable("id") Long id);
```

**查询参数**

```java
@GetMapping("/users")
List<User> searchUsers(
    @RequestParam("keyword") String keyword,
    @RequestParam("page") int page,
    @RequestParam("size") int size
);
```

**请求体**

```java
@PostMapping("/users")
User createUser(@RequestBody User user);
```

**请求头**

```java
@GetMapping("/users/{id}")
User getUser(
    @PathVariable("id") Long id,
    @RequestHeader("Authorization") String token
);
```

### 对象参数

```java
// 使用 @SpringQueryMap 将对象转为查询参数
@GetMapping("/users/search")
List<User> search(@SpringQueryMap UserQuery query);

public class UserQuery {
    private String keyword;
    private Integer page;
    private Integer size;
    // getter/setter
}
```

## 5. 负载均衡

Feign 集成了 Ribbon，自动实现负载均衡：

```yaml
# Ribbon 配置
user-service:
  ribbon:
    # 负载均衡策略
    NFLoadBalancerRuleClassName: com.netflix.loadbalancer.RandomRule
    # 连接超时
    ConnectTimeout: 1000
    # 读取超时
    ReadTimeout: 3000
    # 最大自动重试次数
    MaxAutoRetries: 1
    # 切换实例的重试次数
    MaxAutoRetriesNextServer: 1
```

## 6. 熔断降级

### 启用 Hystrix

```yaml
feign:
  hystrix:
    enabled: true
```

或使用 Sentinel：

```yaml
feign:
  sentinel:
    enabled: true
```

### Fallback 降级

```java
// 1. 定义降级类
@Component
public class UserClientFallback implements UserClient {

    @Override
    public User getUser(Long id) {
        User user = new User();
        user.setId(id);
        user.setName("默认用户");
        return user;
    }

    @Override
    public List<User> listUsers(int page) {
        return Collections.emptyList();
    }

    // 其他方法...
}

// 2. 指定降级类
@FeignClient(name = "user-service", fallback = UserClientFallback.class)
public interface UserClient {
    // ...
}
```

### FallbackFactory 降级（获取异常信息）

```java
// 1. 定义降级工厂
@Component
public class UserClientFallbackFactory implements FallbackFactory<UserClient> {

    private static final Logger log = LoggerFactory.getLogger(UserClientFallbackFactory.class);

    @Override
    public UserClient create(Throwable cause) {
        return new UserClient() {
            @Override
            public User getUser(Long id) {
                log.error("获取用户失败: {}", cause.getMessage());
                User user = new User();
                user.setId(id);
                user.setName("默认用户");
                return user;
            }

            // 其他方法...
        };
    }
}

// 2. 指定降级工厂
@FeignClient(name = "user-service", fallbackFactory = UserClientFallbackFactory.class)
public interface UserClient {
    // ...
}
```

## 7. 拦截器

### RequestInterceptor

```java
@Component
public class FeignRequestInterceptor implements RequestInterceptor {

    @Override
    public void apply(RequestTemplate template) {
        // 添加通用请求头
        template.header("X-Request-Id", UUID.randomUUID().toString());
        template.header("X-Timestamp", String.valueOf(System.currentTimeMillis()));

        // 传递认证信息
        ServletRequestAttributes attributes =
            (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        if (attributes != null) {
            HttpServletRequest request = attributes.getRequest();
            String token = request.getHeader("Authorization");
            if (token != null) {
                template.header("Authorization", token);
            }
        }
    }
}
```

## 8. 日志配置

### 日志级别

```java
@Configuration
public class FeignConfig {

    @Bean
    public Logger.Level feignLoggerLevel() {
        // NONE: 不记录任何日志（默认）
        // BASIC: 仅记录请求方法、URL、响应状态码和执行时间
        // HEADERS: 在 BASIC 基础上，记录请求和响应的头信息
        // FULL: 记录请求和响应的头、正文和元数据
        return Logger.Level.FULL;
    }
}
```

### 日志配置

```yaml
logging:
  level:
    # 开启 Feign 客户端的日志
    com.example.client.UserClient: DEBUG
```

## 9. 文件上传

```java
@FeignClient(name = "file-service", configuration = MultipartSupportConfig.class)
public interface FileClient {

    @PostMapping(value = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    String upload(@RequestPart("file") MultipartFile file);
}

@Configuration
public class MultipartSupportConfig {

    @Bean
    public Encoder feignFormEncoder() {
        return new SpringFormEncoder(new SpringEncoder(new ObjectFactory<HttpMessageConverters>() {
            @Override
            public HttpMessageConverters getObject() {
                return new HttpMessageConverters(new RestTemplate().getMessageConverters());
            }
        }));
    }
}
```

## 10. 文件下载

```java
@FeignClient(name = "file-service")
public interface FileClient {

    @GetMapping("/download/{id}")
    Response download(@PathVariable("id") String id);
}

// 使用
public void downloadFile(String id) {
    Response response = fileClient.download(id);
    Response.Body body = response.body();

    try (InputStream inputStream = body.asInputStream();
         FileOutputStream outputStream = new FileOutputStream("file.jpg")) {
        byte[] buffer = new byte[1024];
        int length;
        while ((length = inputStream.read(buffer)) > 0) {
            outputStream.write(buffer, 0, length);
        }
    } catch (IOException e) {
        e.printStackTrace();
    }
}
```

## 11. 性能优化

### 连接池配置

```xml
<!-- 使用 HttpClient -->
<dependency>
    <groupId>io.github.openfeign</groupId>
    <artifactId>feign-httpclient</artifactId>
</dependency>
```

```yaml
feign:
  httpclient:
    enabled: true
    # 最大连接数
    max-connections: 200
    # 每个路由的最大连接数
    max-connections-per-route: 50
    # 连接超时（毫秒）
    connection-timeout: 2000
```

### 使用 OkHttp

```xml
<dependency>
    <groupId>io.github.openfeign</groupId>
    <artifactId>feign-okhttp</artifactId>
</dependency>
```

```yaml
feign:
  okhttp:
    enabled: true
  httpclient:
    enabled: false
```

## 12. 最佳实践

### 接口设计

- 定义统一的响应格式
- 合理使用请求方法（GET、POST、PUT、DELETE）
- 避免在接口中返回复杂对象

### 超时配置

```yaml
feign:
  client:
    config:
      default:
        # 连接超时
        connect-timeout: 2000
        # 读取超时
        read-timeout: 5000

# Hystrix 超时要大于 Feign + Ribbon 的超时
hystrix:
  command:
    default:
      execution:
        isolation:
          thread:
            timeoutInMilliseconds: 10000
```

### 降级策略

- 为关键接口配置降级
- 降级逻辑要简单快速
- 记录降级日志便于排查

### 异常处理

```java
@Component
public class FeignErrorDecoder implements ErrorDecoder {

    @Override
    public Exception decode(String methodKey, Response response) {
        if (response.status() == 404) {
            return new NotFoundException("资源不存在");
        }
        if (response.status() == 500) {
            return new InternalServerErrorException("服务器内部错误");
        }
        return new RuntimeException("调用失败");
    }
}
```

## 13. 常见问题

### 超时问题

检查配置优先级：

1. Hystrix 超时
2. Ribbon 超时
3. Feign 超时

### 请求头丢失

使用 `RequestInterceptor` 传递请求头

### 性能问题

- 使用连接池
- 合理配置超时时间
- 启用 GZIP 压缩

## 14. 总结

| 特性       | 说明                  |
| ---------- | --------------------- |
| 声明式调用 | 通过接口定义服务调用  |
| 负载均衡   | 集成 Ribbon           |
| 熔断降级   | 集成 Hystrix/Sentinel |
| 可插拔     | 支持多种编解码器      |

---

**关键要点**：

- Feign 简化了 HTTP 客户端开发
- 集成了负载均衡和熔断功能
- 合理配置超时时间很重要
- 为关键接口配置降级

**下一步**：学习 [Ribbon 负载均衡](/docs/springcloud/ribbon)
