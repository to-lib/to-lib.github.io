---
sidebar_position: 5
---

# 项目结构与配置

## 标准项目结构

```
my-springboot-app/
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/
│   │   │       └── example/
│   │   │           └── myapp/
│   │   │               ├── Application.java           # 启动类
│   │   │               ├── controller/               # 控制器层
│   │   │               │   └── UserController.java
│   │   │               ├── service/                  # 业务逻辑层
│   │   │               │   ├── UserService.java
│   │   │               │   └── impl/
│   │   │               │       └── UserServiceImpl.java
│   │   │               ├── repository/               # 数据访问层
│   │   │               │   └── UserRepository.java
│   │   │               ├── entity/                   # 实体类
│   │   │               │   └── User.java
│   │   │               ├── dto/                      # 数据传输对象
│   │   │               │   └── UserDTO.java
│   │   │               ├── config/                   # 配置类
│   │   │               │   └── WebConfig.java
│   │   │               ├── exception/                # 异常处理
│   │   │               │   └── GlobalExceptionHandler.java
│   │   │               └── util/                     # 工具类
│   │   │                   └── CommonUtils.java
│   │   └── resources/
│   │       ├── application.yml                       # 主配置文件
│   │       ├── application-dev.yml                   # 开发环境配置
│   │       ├── application-prod.yml                  # 生产环境配置
│   │       ├── logback-spring.xml                    # 日志配置
│   │       └── static/                               # 静态资源
│   │           ├── css/
│   │           ├── js/
│   │           └── img/
│   │       └── templates/                            # 模板文件（如 Thymeleaf）
│   │           └── index.html
│   └── test/
│       └── java/
│           └── com/example/myapp/
│               ├── controller/
│               │   └── UserControllerTest.java
│               ├── service/
│               │   └── UserServiceTest.java
│               └── ApplicationTests.java
├── pom.xml                                           # Maven 配置
├── mvnw                                              # Maven 包装脚本
├── mvnw.cmd
├── .gitignore
├── README.md
└── HELP.md
```

## POM.xml 配置详解

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <!-- 项目坐标 -->
    <groupId>com.example</groupId>
    <artifactId>my-springboot-app</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <name>My Spring Boot App</name>
    <description>A demo Spring Boot application</description>

    <!-- 继承 Spring Boot 父 POM -->
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.2.0</version>
        <relativePath/>
    </parent>

    <!-- 项目属性 -->
    <properties>
        <java.version>17</java.version>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>

    <!-- 依赖管理 -->
    <dependencies>
        <!-- Web 依赖 -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>

        <!-- 数据访问 -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-jpa</artifactId>
        </dependency>

        <!-- MySQL 驱动 -->
        <dependency>
            <groupId>com.mysql</groupId>
            <artifactId>mysql-connector-java</artifactId>
            <scope>runtime</scope>
        </dependency>

        <!-- Lombok -->
        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
            <optional>true</optional>
        </dependency>

        <!-- DevTools -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-devtools</artifactId>
            <scope>runtime</scope>
            <optional>true</optional>
        </dependency>

        <!-- 测试 -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <!-- 构建插件 -->
    <build>
        <plugins>
            <!-- Spring Boot Maven 插件 -->
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
                <configuration>
                    <excludes>
                        <exclude>
                            <groupId>org.projectlombok</groupId>
                            <artifactId>lombok</artifactId>
                        </exclude>
                    </excludes>
                </configuration>
            </plugin>

            <!-- 编译器插件 -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.8.1</version>
                <configuration>
                    <source>17</source>
                    <target>17</target>
                </configuration>
            </plugin>
        </plugins>
    </build>

</project>
```

## 配置文件详解

### application.yml（主配置文件）

```yaml
# 应用配置
spring:
  application:
    name: my-springboot-app
    version: 1.0.0
  
  # Web 配置
  web:
    locale: zh_CN
  
  # 数据库配置
  datasource:
    url: jdbc:mysql://localhost:3306/mydb?useSSL=false&serverTimezone=UTC&characterEncoding=utf8
    username: root
    password: password
    driver-class-name: com.mysql.cj.jdbc.Driver
    hikari:
      maximum-pool-size: 20
      minimum-idle: 5
      connection-timeout: 30000
      idle-timeout: 600000
  
  # JPA 配置
  jpa:
    hibernate:
      ddl-auto: update          # create, create-drop, update, validate
    show-sql: false
    properties:
      hibernate:
        dialect: org.hibernate.dialect.MySQL8Dialect
        format_sql: true
        use_sql_comments: true
  
  # Redis 配置
  redis:
    host: localhost
    port: 6379
    password: 
    database: 0
    timeout: 2000ms
    jedis:
      pool:
        max-active: 8
        max-idle: 8
        min-idle: 0
        max-wait: -1ms

# 服务器配置
server:
  port: 8080
  servlet:
    context-path: /api
  tomcat:
    max-threads: 200
    min-spare-threads: 10
    max-connections: 10000
    accept-count: 100
    connection-timeout: 60000

# 日志配置
logging:
  level:
    root: INFO
    com.example: DEBUG
  file:
    name: logs/application.log
    max-size: 10MB
    max-history: 30
  pattern:
    console: "%d{yyyy-MM-dd HH:mm:ss} - %msg%n"
    file: "%d{yyyy-MM-dd HH:mm:ss} [%thread] %-5level %logger{36} - %msg%n"

# 自定义配置
app:
  name: MyApp
  version: 1.0.0
  description: My awesome application
  features:
    cache:
      enabled: true
      type: redis
    auth:
      enabled: true
      jwt-secret: my-secret-key
      token-expiration: 3600000
```

### application-dev.yml（开发环境）

```yaml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/mydb_dev
  jpa:
    show-sql: true
  redis:
    host: localhost

server:
  port: 8080

logging:
  level:
    root: DEBUG
    com.example: DEBUG
    org.springframework.web: DEBUG
    org.hibernate: DEBUG

app:
  features:
    cache:
      enabled: false
```

### application-prod.yml（生产环境）

```yaml
spring:
  datasource:
    url: jdbc:mysql://prod-db:3306/mydb
    hikari:
      maximum-pool-size: 50
  jpa:
    show-sql: false
  redis:
    host: redis-prod
    password: prod-password

server:
  port: 8443
  ssl:
    key-store: classpath:keystore.p12
    key-store-password: password
    key-store-type: PKCS12

logging:
  level:
    root: WARN
  file:
    name: /var/logs/app.log

app:
  features:
    cache:
      enabled: true
```

## 配置属性注入

### @Value 注解

```java
@Component
public class AppConfig {
    
    @Value("\\${app.name}")
    private String appName;
    
    @Value("\\${app.version:1.0.0}")  // 默认值
    private String version;
    
    @Value("\\${app.features.cache.enabled}")
    private boolean cacheEnabled;
}
```

### @ConfigurationProperties

```java
@Configuration
@ConfigurationProperties(prefix = "app.features.cache")
@Data
public class CacheProperties {
    
    private boolean enabled;
    private String type;
    private long ttl = 3600;
    private int maxSize = 1000;
}

// 使用
@Component
public class MyService {
    
    @Autowired
    private CacheProperties cacheProperties;
    
    public void init() {
        if (cacheProperties.isEnabled()) {
            System.out.println("Cache type: " + cacheProperties.getType());
        }
    }
}
```

### Environment 对象

```java
@Component
public class EnvironmentService {
    
    @Autowired
    private Environment environment;
    
    public void printProperties() {
        String appName = environment.getProperty("app.name");
        String port = environment.getProperty("server.port", "8080");
        boolean cacheEnabled = environment.getProperty("app.features.cache.enabled", 
                                                       Boolean.class, false);
        
        System.out.println("App: " + appName);
        System.out.println("Port: " + port);
        System.out.println("Cache: " + cacheEnabled);
    }
}
```

## 多环境部署

### 激活 Profile

**方式 1：application.yml**
```yaml
spring:
  profiles:
    active: dev
```

**方式 2：命令行参数**
```bash
java -jar app.jar --spring.profiles.active=prod
```

**方式 3：环境变量**
```bash
export SPRING_PROFILES_ACTIVE=prod
java -jar app.jar
```

**方式 4：IDE 运行配置**
在 IntelliJ IDEA 中：
- Run → Edit Configurations
- 在 VM options 中添加：`-Dspring.profiles.active=dev`
- 或在 Program arguments 中添加：`--spring.profiles.active=dev`

### 启动脚本示例

```bash
#!/bin/bash

# 部署脚本
APP_NAME="my-springboot-app"
JAR_FILE="target/\\${APP_NAME}.jar"
PROFILES="prod"
PORT=8080

# 停止旧实例
echo "Stopping application..."
pkill -f "java.*\\${JAR_FILE}"

sleep 2

# 启动新实例
echo "Starting application..."
nohup java -jar \
    -Dspring.profiles.active=\\${PROFILES} \
    -Dserver.port=\\${PORT} \
    -Xms512m \
    -Xmx1024m \
    \\${JAR_FILE} > logs/app.log 2>&1 &

echo "Application started on port \\${PORT}"
```

## 配置优先级

从高到低：

1. 命令行参数：`--server.port=9000`
2. 系统属性：`-Dserver.port=9000`
3. 操作系统环境变量：`SERVER_PORT=9000`
4. 随机值：`${random.int}`
5. application-{profile}.yml / .properties
6. application.yml / .properties
7. @PropertySource 注解指定
8. 默认值

## 总结

- **结构规范** - 遵循标准的包结构和分层架构
- **配置集中** - 所有配置在 application.yml 中管理
- **多环境支持** - 支持 dev、test、prod 等多个环境
- **灵活注入** - 支持多种方式注入配置属性
- **易于部署** - 简单的部署脚本和命令行参数支持

下一步学习 [Web 开发](./web-development.md)。
