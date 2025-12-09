---
sidebar_position: 11
---

# 常见问题解答

> [!TIP]
> **快速导航**: 使用 `Ctrl+F` 查找关键词，快速定位您遇到的问题。如未找到解答，请查看 [最佳实践](./best-practices.md) 或 [快速参考](./quick-reference.md)。

## 常见配置问题

### Q: 如何修改服务器端口？

A: 在 `application.yml` 中修改：

```yaml
server:
  port: 9000
```

或启动时指定：

```bash
java -jar app.jar --server.port=9000
```

或环境变量：

```bash
export SERVER_PORT=9000
```

### Q: 如何修改上下文路径？

A: 配置文件中：

```yaml
server:
  servlet:
    context-path: /api
```

现在访问：`http://localhost:8080/api/...`

### Q: 如何禁用自动配置？

A: 在启动类上排除：

```java
@SpringBootApplication(exclude = {DataSourceAutoConfiguration.class})
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}
```

或配置文件中：

```yaml
spring:
  autoconfigure:
    exclude:
      - org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration
```

## 数据库相关问题

### Q: 连接超时问题怎么解决？

A: 调整连接池配置：

```yaml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/mydb?useSSL=false&serverTimezone=UTC&connectTimeout=30000
    hikari:
      connection-timeout: 30000
      idle-timeout: 600000
      max-lifetime: 1800000
```

### Q: 表自动创建失败？

A: 检查 JPA 配置：

```yaml
spring:
  jpa:
    hibernate:
      ddl-auto: create  # create, create-drop, update, validate
    properties:
      hibernate:
        dialect: org.hibernate.dialect.MySQL8Dialect
```

确保实体类有 `@Entity` 注解。

### Q: 如何在启动时执行 SQL 脚本？

A: 创建 `src/main/resources/schema.sql` 和 `data.sql`：

```sql
-- schema.sql
CREATE TABLE IF NOT EXISTS users (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  username VARCHAR(50) NOT NULL,
  email VARCHAR(100) NOT NULL UNIQUE
);

-- data.sql
INSERT INTO users VALUES (1, 'admin', 'admin@example.com');
```

Spring Boot 会自动执行这些脚本。

## 依赖和版本问题

### Q: 如何修改 Spring Boot 版本？

A: 在 `pom.xml` 中修改 parent 版本：

```xml
<parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>3.2.0</version>
</parent>
```

### Q: 如何处理依赖冲突？

A: 使用 `<exclusion>` 排除冲突的依赖：

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
    <exclusions>
        <exclusion>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-logging</artifactId>
        </exclusion>
    </exclusions>
</dependency>
```

或查看依赖树：

```bash
mvn dependency:tree
```

### Q: 如何使用自定义的 BOM（Bill of Materials）？

A: 在 dependencyManagement 中引入：

```xml
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>org.springframework.cloud</groupId>
            <artifactId>spring-cloud-dependencies</artifactId>
            <version>${spring-cloud.version}</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>
```

## Web 开发问题

### Q: 如何处理跨域（CORS）？

A: 配置 CORS：

```java
@Configuration
public class CorsConfig implements WebMvcConfigurer {
    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/api/**")
                .allowedOrigins("http://localhost:3000")
                .allowedMethods("*")
                .maxAge(3600);
    }
}
```

### Q: 如何上传文件？

A: Controller 代码：

```java
@PostMapping("/upload")
public ResponseEntity<String> upload(@RequestParam("file") MultipartFile file) {
    String filename = file.getOriginalFilename();
    // 保存文件
    file.transferTo(new File("upload/" + filename));
    return ResponseEntity.ok("File uploaded");
}
```

配置文件：

```yaml
spring:
  servlet:
    multipart:
      max-file-size: 10MB
      max-request-size: 100MB
```

### Q: 如何自定义错误页面？

A: 创建 `src/main/resources/templates/error.html`：

```html
<!DOCTYPE html>
<html>
<head>
    <title>Error</title>
</head>
<body>
    <h1>Something went wrong!</h1>
    <p>Status: <span th:text="${status}"></span></p>
    <p>Message: <span th:text="${error}"></span></p>
</body>
</html>
```

### Q: 如何处理静态资源？

A: 默认位置：`src/main/resources/static/` 和 `src/main/resources/public/`

或自定义：

```yaml
spring:
  web:
    resources:
      static-locations: classpath:/static/,classpath:/public/
```

## 日志问题

### Q: 如何修改日志级别？

A: 配置文件：

```yaml
logging:
  level:
    root: INFO
    com.example: DEBUG
    org.springframework.web: DEBUG
```

或启动参数：

```bash
java -jar app.jar --logging.level.root=DEBUG
```

### Q: 如何输出日志到文件？

A: 配置文件：

```yaml
logging:
  file:
    name: logs/app.log
    max-size: 10MB
    max-history: 30
  pattern:
    file: "%d{yyyy-MM-dd HH:mm:ss} [%thread] %-5level %logger{36} - %msg%n"
```

### Q: 如何使用 logback 配置？

A: 创建 `src/main/resources/logback-spring.xml`：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <property name="LOG_FILE" value="${LOG_FILE:-${LOG_PATH:-${LOG_TEMP:-${java.io.tmpdir:-/tmp}}/}spring.log}"/>
    
    <appender name="FILE" class="ch.qos.logback.core.rolling.RollingFileAppender">
        <file>${LOG_FILE}</file>
        <encoder>
            <pattern>%d{yyyy-MM-dd HH:mm:ss} - %msg%n</pattern>
        </encoder>
    </appender>
    
    <root level="INFO">
        <appender-ref ref="FILE"/>
    </root>
</configuration>
```

## 性能问题

### Q: 应用启动慢？

A: 可能原因和解决方案：

1. **减少扫描范围**：

```java
@SpringBootApplication(scanBasePackages = {"com.example"})
public class Application {}
```

2. **延迟初始化**：

```yaml
spring:
  jpa:
    hibernate:
      jdbc:
        batch_size: 20
```

3. **检查启动时间**：

```bash
java -jar app.jar --debug
```

### Q: 内存占用过高？

A: 调整堆内存：

```bash
java -Xms512m -Xmx1024m -jar app.jar
```

或检查连接池：

```yaml
spring:
  datasource:
    hikari:
      maximum-pool-size: 20
      minimum-idle: 5
```

### Q: 查询速度慢？

A:

1. **使用分页**：

```java
Pageable pageable = PageRequest.of(0, 20);
Page<User> users = userRepository.findAll(pageable);
```

2. **使用缓存**：

```java
@Cacheable("users")
public User getUser(Long id) { ... }
```

3. **优化 SQL**：使用 `@Query` 自定义查询，只查询需要的列。

## 部署问题

### Q: 如何打包成 JAR 文件？

A:

```bash
mvn clean package
```

运行：

```bash
java -jar target/app-0.0.1-SNAPSHOT.jar
```

### Q: 如何打包成 WAR 文件？

A: 修改 `pom.xml`：

```xml
<packaging>war</packaging>

<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-tomcat</artifactId>
    <scope>provided</scope>
</dependency>
```

修改启动类：

```java
@SpringBootApplication
public class Application extends SpringBootServletInitializer {
    @Override
    protected SpringApplicationBuilder configure(SpringApplicationBuilder builder) {
        return builder.sources(Application.class);
    }
}
```

### Q: 如何在 Docker 中运行？

A: 创建 `Dockerfile`：

```dockerfile
FROM openjdk:17-slim
COPY target/app.jar app.jar
ENTRYPOINT ["java","-jar","/app.jar"]
```

构建和运行：

```bash
docker build -t myapp .
docker run -p 8080:8080 myapp
```

### Q: 如何配置 HTTPS？

A: 生成密钥：

```bash
keytool -genkey -alias tomcat -keyalg RSA -keystore keystore.p12 -keypass password -storepass password -storetype PKCS12
```

配置文件：

```yaml
server:
  port: 8443
  ssl:
    key-store: classpath:keystore.p12
    key-store-password: password
    key-store-type: PKCS12
```

## 测试问题

### Q: 如何进行单元测试？

A:

```java
@SpringBootTest
public class UserServiceTest {
    
    @Autowired
    private UserService userService;
    
    @Test
    public void testGetUser() {
        User user = userService.getUser(1L);
        assertNotNull(user);
        assertEquals("John", user.getName());
    }
}
```

### Q: 如何 Mock 依赖？

A:

```java
@SpringBootTest
public class UserControllerTest {
    
    @MockBean
    private UserService userService;
    
    @Autowired
    private MockMvc mockMvc;
    
    @Test
    public void testGetUser() throws Exception {
        when(userService.getUser(1L))
            .thenReturn(new User(1L, "John"));
        
        mockMvc.perform(get("/api/users/1"))
            .andExpect(status().isOk());
    }
}
```

## 调试问题

### Q: 如何启用远程调试？

A: 运行时添加参数：

```bash
java -agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=5005 -jar app.jar
```

在 IDE 中配置远程调试，连接到 `localhost:5005`。

### Q: 如何查看应用配置？

A: 启用 Actuator 端点：

```yaml
management:
  endpoints:
    web:
      exposure:
        include: configprops,env
```

访问：`http://localhost:8080/actuator/env`

---

**提示**：更多问题和解决方案，请查看 [最佳实践](./best-practices.md)。
