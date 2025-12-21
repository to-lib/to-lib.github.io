---
sidebar_position: 100
title: Spring Boot 面试题
---

# Spring Boot 面试题精选

> [!TIP]
> 本文精选了 30+ 道 Spring Boot 高频面试题，涵盖核心概念、自动配置、Web 开发、数据访问、安全、监控等核心主题。

## 目录

- [🎯 核心概念](#-核心概念)
- [🎯 自动配置](#-自动配置)
- [🎯 Web 开发](#-web-开发)
- [🎯 数据访问](#-数据访问)
- [🎯 安全与监控](#-安全与监控)
- [🎯 部署与运维](#-部署与运维)

---

## 🎯 核心概念

### 1. 什么是 Spring Boot？它解决了什么问题？

**答案要点：**

Spring Boot 是基于 Spring 框架的快速开发脚手架，主要解决以下问题：

| 问题         | Spring Boot 解决方案           |
| ------------ | ------------------------------ |
| 配置繁琐     | 自动配置（Auto-Configuration） |
| 依赖管理复杂 | Starter 依赖简化               |
| 部署麻烦     | 内嵌服务器，可执行 JAR         |
| 缺乏标准化   | 约定优于配置                   |

**核心特性：**

```java
// 一个注解启动整个应用
@SpringBootApplication
public class MyApplication {
    public static void main(String[] args) {
        SpringApplication.run(MyApplication.class, args);
    }
}
```

**延伸：** 参考 [Spring Boot 核心概念](/docs/springboot/core-concepts)

---

### 2. @SpringBootApplication 注解包含哪些注解？

**答案要点：**

```java
@SpringBootApplication
    ├── @SpringBootConfiguration  // 标识为配置类
    │       └── @Configuration
    ├── @EnableAutoConfiguration  // 启用自动配置
    │       └── @Import(AutoConfigurationImportSelector.class)
    └── @ComponentScan            // 组件扫描
```

**各注解作用：**

| 注解                       | 作用                   |
| -------------------------- | ---------------------- |
| `@SpringBootConfiguration` | 标识当前类为配置类     |
| `@EnableAutoConfiguration` | 启用自动配置机制       |
| `@ComponentScan`           | 扫描当前包及子包的组件 |

---

### 3. Spring Boot 的启动流程是怎样的？

**答案要点：**

```
1. SpringApplication.run()
    ↓
2. 创建 SpringApplication 实例
    - 推断应用类型（Servlet/Reactive/None）
    - 加载 ApplicationContextInitializer
    - 加载 ApplicationListener
    ↓
3. 运行 run() 方法
    - 创建并配置 Environment
    - 创建 ApplicationContext
    - 准备上下文（prepareContext）
    - 刷新上下文（refreshContext）
    - 执行 Runner（CommandLineRunner/ApplicationRunner）
```

**代码示例：**

```java
@Component
public class MyRunner implements CommandLineRunner {
    @Override
    public void run(String... args) throws Exception {
        System.out.println("应用启动完成后执行");
    }
}
```

---

## 🎯 自动配置

### 4. Spring Boot 自动配置原理是什么？

**答案要点：**

**自动配置加载流程：**

```
@EnableAutoConfiguration
    ↓
AutoConfigurationImportSelector.selectImports()
    ↓
SpringFactoriesLoader.loadFactoryNames()
    ↓
读取 META-INF/spring.factories
    ↓
过滤条件注解（@ConditionalOnXxx）
    ↓
加载符合条件的自动配置类
```

**spring.factories 示例：**

```properties
# META-INF/spring.factories
org.springframework.boot.autoconfigure.EnableAutoConfiguration=\
org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration,\
org.springframework.boot.autoconfigure.web.servlet.WebMvcAutoConfiguration
```

**延伸：** 参考 [Spring Boot 自动配置](/docs/springboot/auto-configuration)

---

### 5. 常用的条件注解有哪些？

**答案要点：**

| 注解                           | 条件                  |
| ------------------------------ | --------------------- |
| `@ConditionalOnClass`          | 类路径存在指定类      |
| `@ConditionalOnMissingClass`   | 类路径不存在指定类    |
| `@ConditionalOnBean`           | 容器中存在指定 Bean   |
| `@ConditionalOnMissingBean`    | 容器中不存在指定 Bean |
| `@ConditionalOnProperty`       | 配置属性满足条件      |
| `@ConditionalOnWebApplication` | 是 Web 应用           |

**示例：**

```java
@Configuration
@ConditionalOnClass(DataSource.class)
@ConditionalOnProperty(prefix = "spring.datasource", name = "url")
public class DataSourceAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public DataSource dataSource() {
        return DataSourceBuilder.create().build();
    }
}
```

---

### 6. 如何自定义一个 Starter？

**答案要点：**

**步骤：**

1. 创建自动配置类
2. 创建配置属性类
3. 创建 spring.factories 文件
4. 打包发布

**代码示例：**

```java
// 1. 配置属性类
@ConfigurationProperties(prefix = "my.service")
public class MyServiceProperties {
    private String name = "default";
    private boolean enabled = true;
    // getters/setters
}

// 2. 自动配置类
@Configuration
@EnableConfigurationProperties(MyServiceProperties.class)
@ConditionalOnClass(MyService.class)
public class MyServiceAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    @ConditionalOnProperty(prefix = "my.service", name = "enabled",
                           havingValue = "true", matchIfMissing = true)
    public MyService myService(MyServiceProperties properties) {
        return new MyService(properties.getName());
    }
}

// 3. META-INF/spring.factories
org.springframework.boot.autoconfigure.EnableAutoConfiguration=\
com.example.MyServiceAutoConfiguration
```

---

## 🎯 Web 开发

### 7. Spring Boot 如何处理静态资源？

**答案要点：**

**默认静态资源路径（优先级从高到低）：**

```
classpath:/META-INF/resources/
classpath:/resources/
classpath:/static/
classpath:/public/
```

**自定义配置：**

```yaml
spring:
  web:
    resources:
      static-locations: classpath:/custom-static/
      cache:
        period: 3600
```

---

### 8. @RestController 和 @Controller 的区别？

**答案要点：**

| 注解              | 返回值处理    | 适用场景           |
| ----------------- | ------------- | ------------------ |
| `@Controller`     | 返回视图名称  | 传统 MVC，返回页面 |
| `@RestController` | 返回 JSON/XML | RESTful API        |

```java
// @RestController = @Controller + @ResponseBody
@RestController
public class UserController {

    @GetMapping("/users/{id}")
    public User getUser(@PathVariable Long id) {
        return userService.findById(id);  // 自动序列化为 JSON
    }
}
```

---

### 9. 如何实现全局异常处理？

**答案要点：**

```java
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(BusinessException.class)
    public Result<Void> handleBusinessException(BusinessException e) {
        return Result.error(e.getCode(), e.getMessage());
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public Result<Void> handleValidationException(MethodArgumentNotValidException e) {
        String message = e.getBindingResult().getFieldErrors().stream()
            .map(FieldError::getDefaultMessage)
            .collect(Collectors.joining(", "));
        return Result.error(400, message);
    }

    @ExceptionHandler(Exception.class)
    public Result<Void> handleException(Exception e) {
        log.error("系统异常", e);
        return Result.error(500, "系统繁忙，请稍后重试");
    }
}
```

**延伸：** 参考 [Spring Boot Web 开发](/docs/springboot/web-development)

---

## 🎯 数据访问

### 10. Spring Boot 如何配置多数据源？

**答案要点：**

```java
@Configuration
public class DataSourceConfig {

    @Bean
    @Primary
    @ConfigurationProperties("spring.datasource.primary")
    public DataSource primaryDataSource() {
        return DataSourceBuilder.create().build();
    }

    @Bean
    @ConfigurationProperties("spring.datasource.secondary")
    public DataSource secondaryDataSource() {
        return DataSourceBuilder.create().build();
    }
}
```

**配置文件：**

```yaml
spring:
  datasource:
    primary:
      url: jdbc:mysql://localhost:3306/db1
      username: root
      password: root
    secondary:
      url: jdbc:mysql://localhost:3306/db2
      username: root
      password: root
```

---

### 11. Spring Boot 事务管理如何配置？

**答案要点：**

```java
@Service
public class UserService {

    @Transactional(rollbackFor = Exception.class)
    public void createUser(User user) {
        userRepository.save(user);
        // 发生异常会回滚
    }

    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void logOperation(String operation) {
        // 独立事务，不受外部事务影响
    }
}
```

**事务传播行为：**

| 传播行为       | 说明                     |
| -------------- | ------------------------ |
| `REQUIRED`     | 默认，有则加入，无则新建 |
| `REQUIRES_NEW` | 总是新建事务             |
| `NESTED`       | 嵌套事务                 |
| `SUPPORTS`     | 有则加入，无则非事务执行 |

**延伸：** 参考 [Spring Boot 事务管理](/docs/springboot/transaction)

---

### 12. JPA 和 MyBatis 如何选择？

**答案要点：**

| 特性     | JPA/Hibernate | MyBatis  |
| -------- | ------------- | -------- |
| SQL 控制 | 自动生成      | 手写 SQL |
| 学习曲线 | 较陡          | 平缓     |
| 复杂查询 | 较弱          | 强       |
| 缓存     | 一级+二级缓存 | 需配置   |
| 适用场景 | 简单 CRUD     | 复杂查询 |

---

## 🎯 安全与监控

### 13. Spring Boot 如何集成 Spring Security？

**答案要点：**

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/public/**").permitAll()
                .requestMatchers("/admin/**").hasRole("ADMIN")
                .anyRequest().authenticated()
            )
            .formLogin(form -> form
                .loginPage("/login")
                .permitAll()
            )
            .logout(logout -> logout
                .logoutSuccessUrl("/")
            );
        return http.build();
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
}
```

**延伸：** 参考 [Spring Boot 安全](/docs/springboot/security)

---

### 14. Spring Boot Actuator 有哪些常用端点？

**答案要点：**

| 端点                   | 说明      |
| ---------------------- | --------- |
| `/actuator/health`     | 健康检查  |
| `/actuator/info`       | 应用信息  |
| `/actuator/metrics`    | 指标数据  |
| `/actuator/env`        | 环境变量  |
| `/actuator/beans`      | Bean 列表 |
| `/actuator/mappings`   | 请求映射  |
| `/actuator/threaddump` | 线程转储  |
| `/actuator/heapdump`   | 堆转储    |

**配置：**

```yaml
management:
  endpoints:
    web:
      exposure:
        include: health,info,metrics
  endpoint:
    health:
      show-details: always
```

**延伸：** 参考 [Spring Boot 健康监控](/docs/springboot/health-monitoring)

---

### 15. 如何自定义健康检查指标？

**答案要点：**

```java
@Component
public class DatabaseHealthIndicator implements HealthIndicator {

    @Autowired
    private DataSource dataSource;

    @Override
    public Health health() {
        try (Connection conn = dataSource.getConnection()) {
            if (conn.isValid(1)) {
                return Health.up()
                    .withDetail("database", "MySQL")
                    .withDetail("status", "Connected")
                    .build();
            }
        } catch (SQLException e) {
            return Health.down()
                .withException(e)
                .build();
        }
        return Health.down().build();
    }
}
```

---

## 🎯 部署与运维

### 16. Spring Boot 如何打包部署？

**答案要点：**

**打包方式：**

```bash
# 打包为可执行 JAR
mvn clean package

# 运行
java -jar app.jar

# 指定配置文件
java -jar app.jar --spring.profiles.active=prod

# 指定 JVM 参数
java -Xms512m -Xmx1024m -jar app.jar
```

**Docker 部署：**

```dockerfile
FROM openjdk:17-jdk-slim
COPY target/app.jar app.jar
ENTRYPOINT ["java", "-jar", "/app.jar"]
```

**延伸：** 参考 [Spring Boot 部署](/docs/springboot/deployment)

---

### 17. 如何实现配置文件的多环境管理？

**答案要点：**

**文件命名规则：**

```
application.yml          # 公共配置
application-dev.yml      # 开发环境
application-test.yml     # 测试环境
application-prod.yml     # 生产环境
```

**激活方式：**

```yaml
# application.yml
spring:
  profiles:
    active: dev
```

```bash
# 命令行激活
java -jar app.jar --spring.profiles.active=prod

# 环境变量激活
export SPRING_PROFILES_ACTIVE=prod
```

---

### 18. Spring Boot 如何优雅停机？

**答案要点：**

```yaml
# 配置优雅停机
server:
  shutdown: graceful

spring:
  lifecycle:
    timeout-per-shutdown-phase: 30s
```

```java
@PreDestroy
public void onShutdown() {
    // 清理资源
    log.info("应用正在关闭...");
}
```

---

## 🎯 性能优化

### 19. Spring Boot 应用如何优化启动速度？

**答案要点：**

1. **延迟初始化**

```yaml
spring:
  main:
    lazy-initialization: true
```

2. **排除不需要的自动配置**

```java
@SpringBootApplication(exclude = {
    DataSourceAutoConfiguration.class,
    SecurityAutoConfiguration.class
})
```

3. **使用 Spring Native（GraalVM）**

4. **减少组件扫描范围**

---

### 20. 如何监控 Spring Boot 应用性能？

**答案要点：**

**集成 Micrometer + Prometheus：**

```xml
<dependency>
    <groupId>io.micrometer</groupId>
    <artifactId>micrometer-registry-prometheus</artifactId>
</dependency>
```

```yaml
management:
  endpoints:
    web:
      exposure:
        include: prometheus
  metrics:
    tags:
      application: ${spring.application.name}
```

**自定义指标：**

```java
@Component
public class OrderMetrics {

    private final Counter orderCounter;

    public OrderMetrics(MeterRegistry registry) {
        this.orderCounter = Counter.builder("orders.created")
            .description("Number of orders created")
            .register(registry);
    }

    public void recordOrder() {
        orderCounter.increment();
    }
}
```

**延伸：** 参考 [Spring Boot 可观测性](/docs/springboot/observability)

---

## 📌 总结

### 学习建议

1. **掌握核心原理** - 自动配置、条件注解
2. **熟悉常用 Starter** - Web、Data、Security
3. **了解生产特性** - Actuator、配置管理
4. **实践项目经验** - 多环境部署、性能优化

### 相关资源

- [Spring Boot 官方文档](https://docs.spring.io/spring-boot/docs/current/reference/html/)
- [Spring Boot 核心概念](/docs/springboot/core-concepts)
- [Spring Boot 自动配置](/docs/springboot/auto-configuration)
- [Spring Boot 最佳实践](/docs/springboot/best-practices)
