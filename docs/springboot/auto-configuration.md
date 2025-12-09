---
sidebar_position: 4
---

# 自动配置详解

> [!IMPORTANT]
> **Spring Boot 的核心特性**: 自动配置能根据 classpath 中的jar包和已定义的Bean自动配置Spring应用。理解自动配置原理对于深入掌握Spring Boot至关重要。

### 工作流程

```
1. 扫描 META-INF/spring.factories
   ↓
2. 加载自动配置类列表
   ↓
3. 逐一判断条件是否满足
   ↓
4. 条件满足则装配该配置
   ↓
5. 使用用户自定义配置覆盖
```

## @EnableAutoConfiguration

```java
@SpringBootApplication
public class Application {
    // @SpringBootApplication 包含 @EnableAutoConfiguration
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}

// 等价于

@Configuration
@EnableAutoConfiguration
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}
```

### 排除自动配置

```java
@SpringBootApplication(exclude = {DataSourceAutoConfiguration.class})
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}

// 或在配置文件中
spring.autoconfigure.exclude=\
  org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration,\
  org.springframework.boot.autoconfigure.orm.jpa.HibernateJpaAutoConfiguration
```

## 常见的自动配置类

### 1. Web 自动配置

当 classpath 中有 `spring-webmvc` 时自动配置：

```java
@Configuration
@ConditionalOnWebApplication(type = Type.SERVLET)
@ConditionalOnClass(DispatcherServlet.class)
@AutoConfigureOrder(Ordered.HIGHEST_PRECEDENCE)
public class DispatcherServletAutoConfiguration {
    
    @Bean
    @ConditionalOnMissingBean(DispatcherServlet.class)
    public DispatcherServlet dispatcherServlet(WebMvcProperties webMvcProperties) {
        DispatcherServlet servlet = new DispatcherServlet();
        servlet.setDispatchOptionsRequest(webMvcProperties.isDispatchOptionsRequest());
        return servlet;
    }
}
```

### 2. 数据库自动配置

```yaml
# application.yml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/mydb
    username: root
    password: password
    driver-class-name: com.mysql.cj.jdbc.Driver
  jpa:
    hibernate:
      ddl-auto: update
    show-sql: true
    properties:
      hibernate:
        dialect: org.hibernate.dialect.MySQL8Dialect
        format_sql: true
```

### 3. Web 容器自动配置

```java
// Tomcat 自动配置（默认）
@Configuration
@ConditionalOnClass(Tomcat.class)
public class EmbeddedTomcatWebServerFactoryCustomizer {
    // ...
}

// 可以通过 application.yml 定制
server:
  tomcat:
    max-threads: 200
    min-spare-threads: 10
    accept-count: 100
    connection-timeout: 60000
```

## 条件注解详解

### @ConditionalOnClass

当 classpath 中存在指定类时装配 Bean：

```java
@Configuration
@ConditionalOnClass(name = "com.mysql.jdbc.Driver")
public class MysqlAutoConfiguration {
    
    @Bean
    public DataSource dataSource() {
        // 仅当 MySQL JDBC 驱动在 classpath 中时创建
        return createMysqlDataSource();
    }
}
```

### @ConditionalOnMissingClass

当 classpath 中不存在指定类时装配：

```java
@Bean
@ConditionalOnMissingClass("org.springframework.security.core.Security")
public BasicSecurityConfiguration basicSecurityConfig() {
    // 当没有 Spring Security 时使用基本安全配置
}
```

### @ConditionalOnBean

当容器中存在指定 Bean 时装配：

```java
@Configuration
public class CacheConfiguration {
    
    @Bean
    @ConditionalOnBean(CacheManager.class)
    public CacheInterceptor cacheInterceptor(CacheManager cacheManager) {
        // 仅当 CacheManager 存在时创建拦截器
        return new CacheInterceptor(cacheManager);
    }
}
```

### @ConditionalOnMissingBean

当容器中不存在指定 Bean 时装配：

```java
@Configuration
public class DefaultServiceConfiguration {
    
    @Bean
    @ConditionalOnMissingBean
    public UserService defaultUserService() {
        // 当没有定义 UserService 时，使用默认实现
        return new DefaultUserService();
    }
}
```

### @ConditionalOnProperty

根据配置属性值装配：

```java
@Configuration
@ConditionalOnProperty(
    name = "app.cache.enabled",
    havingValue = "true",
    matchIfMissing = false
)
public class CacheConfiguration {
    
    @Bean
    public CacheManager cacheManager() {
        return new CaffeineCacheManager();
    }
}
```

`matchIfMissing = true` 表示当属性不存在时也装配。

### @ConditionalOnExpression

根据 SpEL 表达式装配：

```java
@Bean
@ConditionalOnExpression("${app.advanced.features:true}")
public AdvancedFeatureService advancedFeatureService() {
    return new AdvancedFeatureService();
}
```

### 自定义 @Conditional

```java
// 自定义条件
public class LinuxCondition implements Condition {
    @Override
    public boolean matches(ConditionContext context, 
                          AnnotatedTypeMetadata metadata) {
        String osName = System.getProperty("os.name").toLowerCase();
        return osName.contains("linux");
    }
}

// 使用自定义条件
@Configuration
public class OsSpecificConfiguration {
    
    @Bean
    @Conditional(LinuxCondition.class)
    public LinuxService linuxService() {
        return new LinuxService();
    }
}
```

## 自动配置顺序控制

### @AutoConfigureOrder

```java
@Configuration
@AutoConfigureOrder(Ordered.HIGHEST_PRECEDENCE)
public class HighPriorityAutoConfiguration {
    // 优先级最高
}

@Configuration
@AutoConfigureOrder(Ordered.LOWEST_PRECEDENCE)
public class LowPriorityAutoConfiguration {
    // 优先级最低
}
```

### @AutoConfigureBefore / @AutoConfigureAfter

```java
@Configuration
@AutoConfigureBefore(DataSourceAutoConfiguration.class)
public class CustomBeforeConfiguration {
    // 在 DataSourceAutoConfiguration 之前执行
}

@Configuration
@AutoConfigureAfter(DataSourceAutoConfiguration.class)
public class CustomAfterConfiguration {
    // 在 DataSourceAutoConfiguration 之后执行
}
```

## 自定义自动配置

### 1. 创建自动配置类

```java
@Configuration
@ConditionalOnClass(MyService.class)
@EnableConfigurationProperties(MyServiceProperties.class)
public class MyServiceAutoConfiguration {
    
    @Bean
    @ConditionalOnMissingBean
    public MyService myService(MyServiceProperties properties) {
        return new MyService(properties);
    }
}
```

### 2. 配置属性类

```java
@ConfigurationProperties(prefix = "myapp.service")
@Data
public class MyServiceProperties {
    
    private boolean enabled = true;
    private String name = "default";
    private int timeout = 30;
    
    // Getter 和 Setter 由 @Data 生成
}
```

### 3. 在 META-INF/spring.factories 中注册

```properties
# src/main/resources/META-INF/spring.factories
org.springframework.boot.autoconfigure.EnableAutoConfiguration=\
  com.example.MyServiceAutoConfiguration
```

### 4. 使用配置

```yaml
myapp:
  service:
    enabled: true
    name: custom-service
    timeout: 60
```

## 查看自动配置报告

### 启用调试日志

```yaml
logging:
  level:
    org.springframework.boot.autoconfigure: DEBUG
```

### 运行应用

```bash
java -jar app.jar --debug
```

或在 IDE 中运行时添加 `--debug` 参数。

### 输出示例

```
=== AUTO-CONFIGURATION REPORT ===

Positive matches:
-----------------
   DataSourceAutoConfiguration matched:
      - @ConditionalOnClass found required class 'javax.sql.DataSource' (OnClassCondition)

Negative matches:
-----------------
   MongoAutoConfiguration:
      - @ConditionalOnClass did not find required class 'com.mongodb.client.MongoClient' (OnClassCondition)
```

## 自动配置原理示例

完整的自动配置类示例：

```java
package com.example.autoconfigure;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import lombok.Data;

// 配置属性类
@ConfigurationProperties(prefix = "app.mail")
@Data
class MailProperties {
    private String from;
    private String host;
    private int port = 25;
    private String username;
    private String password;
}

// 邮件服务类
class MailService {
    private MailProperties properties;
    
    public MailService(MailProperties properties) {
        this.properties = properties;
    }
    
    public void sendMail(String to, String subject, String body) {
        System.out.println("Sending mail to: " + to);
    }
}

// 自动配置类
@AutoConfiguration
@ConditionalOnClass(MailService.class)
@ConditionalOnProperty(
    name = "app.mail.enabled",
    havingValue = "true",
    matchIfMissing = false
)
@EnableConfigurationProperties(MailProperties.class)
public class MailAutoConfiguration {
    
    @Bean
    @ConditionalOnMissingBean
    public MailService mailService(MailProperties properties) {
        return new MailService(properties);
    }
}
```

配置文件：

```yaml
app:
  mail:
    enabled: true
    from: noreply@example.com
    host: smtp.example.com
    port: 587
    username: user
    password: password
```

## 总结

Spring Boot 自动配置的关键点：

1. **自动检测** - 根据 classpath 和已有 Bean 自动配置
2. **条件装配** - 使用 @Conditional 系列注解控制装配逻辑
3. **优先级控制** - 使用 @AutoConfigureOrder 和 @AutoConfigureBefore/After 控制顺序
4. **可覆盖** - 用户定义的 Bean 总是优先于自动配置
5. **可排除** - 支持排除不需要的自动配置类

下一步学习 [项目结构与配置](./project-structure-config.md)。
