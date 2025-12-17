---
id: configuration
title: Spring 配置与 Profiles
sidebar_label: 配置与Profiles
sidebar_position: 5
---

# Spring 配置与 Profiles

Spring Framework 的核心能力之一是“用一致的方式管理应用配置”：

- 基于 **JavaConfig**（`@Configuration` / `@Bean`）
- 基于 **组件扫描**（`@ComponentScan`）
- 基于 **外部化配置**（properties/yaml + `Environment`）
- 基于 **环境隔离**（Profiles）

本文聚焦 Spring Framework（非 Spring Boot 专属能力），在纯 Spring/Boot 两种场景都适用。

## 1. 配置方式总览

### 1.1 JavaConfig（推荐）

```java
@Configuration
public class AppConfig {

    @Bean
    public Clock clock() {
        return Clock.systemUTC();
    }

    @Bean
    public OrderService orderService(Clock clock) {
        return new OrderService(clock);
    }
}
```

要点：

- `@Configuration` 会让 Spring 对配置类做增强（CGLIB），保证同一个 `@Bean` 方法在容器内是单例语义。
- `@Bean` 方法参数可直接声明依赖，Spring 会自动解析。

### 1.2 组件扫描

```java
@Configuration
@ComponentScan(basePackages = "com.example")
public class AppConfig {
}
```

常用配套：

- `@Component` / `@Service` / `@Repository` / `@Controller`
- `@ConfigurationProperties`（更偏 Boot 常用，但底层也是 Spring 绑定能力）

### 1.3 XML（理解即可）

```xml
<beans>
  <bean id="clock" class="java.time.Clock" factory-method="systemUTC"/>
</beans>
```

在维护老项目时可能遇到，但新项目通常不再建议。

## 2. 导入与模块化配置

### 2.1 使用 `@Import`

```java
@Configuration
@Import({DataSourceConfig.class, WebConfig.class})
public class RootConfig {
}
```

### 2.2 使用 `@ImportResource`（兼容 XML）

```java
@Configuration
@ImportResource("classpath:spring/legacy-context.xml")
public class LegacyConfig {
}
```

## 3. 外部化配置：PropertySource 与 Environment

### 3.1 `@PropertySource`

```java
@Configuration
@PropertySource("classpath:application.properties")
public class PropertyConfig {
}
```

### 3.2 使用 `Environment` 读取属性

```java
@Component
public class FeatureFlags {

    private final Environment env;

    public FeatureFlags(Environment env) {
        this.env = env;
    }

    public boolean isNewCheckoutEnabled() {
        return env.getProperty("feature.checkout.v2", Boolean.class, false);
    }
}
```

### 3.3 `@Value` 注入（少量使用）

```java
@Component
public class HttpClientConfig {

    @Value("${http.connect-timeout-ms:3000}")
    private int connectTimeoutMs;
}
```

当注入项变多时，建议改用 `@ConfigurationProperties`（可读性更好）。

## 4. Profiles：隔离环境配置

Profile 用于区分不同环境（dev/test/prod）或不同部署形态（single/cluster）。

### 4.1 通过 `@Profile` 注册不同 Bean

```java
@Configuration
public class StorageConfig {

    @Bean
    @Profile("dev")
    public Storage storageDev() {
        return new InMemoryStorage();
    }

    @Bean
    @Profile("prod")
    public Storage storageProd() {
        return new S3Storage();
    }
}
```

### 4.2 激活 Profile 的方式

- JVM 启动参数：

```bash
-Dspring.profiles.active=dev
```

- 环境变量（常见于容器）：

```bash
SPRING_PROFILES_ACTIVE=prod
```

- 测试中：

```java
@ActiveProfiles("test")
class MyTest {
}
```

### 4.3 读取当前 Profile

```java
@Component
public class EnvInfo {

    private final Environment env;

    public EnvInfo(Environment env) {
        this.env = env;
    }

    public List<String> activeProfiles() {
        return List.of(env.getActiveProfiles());
    }
}
```

## 5. 常见坑与最佳实践

- **[self-invocation]**：同类内部直接调用带 AOP 的方法（如 `@Transactional`、`@Cacheable`）会绕过代理。
- **[配置类增强]**：`@Configuration` 与 `@Component` 的差异会影响 `@Bean` 方法的单例语义（一般不要把配置类只标成 `@Component`）。
- **[属性默认值]**：`Environment#getProperty` 可提供默认值，避免 NPE。
- **[Profile 命名]**：建议固定为 `dev/test/staging/prod`，避免随意扩散。

---

下一步建议：

- [SpEL 表达式](/docs/spring/spel)
- [缓存抽象](/docs/spring/caching)
- [参数校验](/docs/spring/validation)
