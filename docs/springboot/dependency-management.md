---
sidebar_position: 8
---

# 依赖管理

> [!TIP]
> **Starter 的力量**: Spring Boot Starter 简化了依赖管理,自动处理版本兼容。使用 dependencyManagement 统一管理版本。

## Spring Boot Starter

Spring Boot Starter 是一组方便的依赖描述符，可简化 Maven 配置。它们让开发者能够无需关心具体的版本号和配置细节。

### 常用 Starters

| Starter | 说明 |
|---------|------|
| `spring-boot-starter-web` | Web 应用（包含 Spring MVC、Tomcat） |
| `spring-boot-starter-webflux` | 响应式 Web 应用 |
| `spring-boot-starter-data-jpa` | 数据库访问（JPA/Hibernate） |
| `spring-boot-starter-data-mongodb` | MongoDB 支持 |
| `spring-boot-starter-data-redis` | Redis 支持 |
| `spring-boot-starter-security` | Spring Security |
| `spring-boot-starter-validation` | 数据验证 |
| `spring-boot-starter-logging` | 日志支持（默认 Logback） |
| `spring-boot-starter-actuator` | 应用监控和管理 |
| `spring-boot-starter-test` | 单元测试（JUnit、Mockito） |
| `spring-boot-starter-amqp` | RabbitMQ 支持 |
| `spring-boot-starter-kafka` | Kafka 支持 |

### Starter 依赖示例

```xml
<dependencies>
    <!-- Web 应用 -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>

    <!-- 数据库 -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-jpa</artifactId>
    </dependency>

    <!-- MySQL 驱动 -->
    <dependency>
        <groupId>com.mysql</groupId>
        <artifactId>mysql-connector-java</artifactId>
        <version>8.0.33</version>
        <scope>runtime</scope>
    </dependency>

    <!-- Redis -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-redis</artifactId>
    </dependency>

    <!-- 测试 -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-test</artifactId>
        <scope>test</scope>
    </dependency>

    <!-- DevTools -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-devtools</artifactId>
        <optional>true</optional>
        <scope>runtime</scope>
    </dependency>
</dependencies>
```

## 版本管理

### 继承 Spring Boot 父 POM

```xml
<parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>3.2.0</version>
    <relativePath/>
</parent>
```

### 不继承父 POM 的方式

如果不能继承父 POM，可以使用 `dependencyManagement`：

```xml
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-dependencies</artifactId>
            <version>3.2.0</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>

<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
        <!-- 无需指定版本，从 dependencyManagement 中继承 -->
    </dependency>
</dependencies>
```

## 版本兼容性

### Java 版本要求

| Spring Boot 版本 | Java 版本 |
|-----------------|----------|
| 3.2.x | 17+ |
| 3.1.x | 17+ |
| 3.0.x | 17+ |
| 2.7.x | 8+ |
| 2.6.x | 8+ |

### 依赖版本冲突解决

查看依赖树：

```bash
mvn dependency:tree
```

排除冲突的依赖：

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

## 常用第三方库

### 工具库

```xml
<!-- Lombok -->
<dependency>
    <groupId>org.projectlombok</groupId>
    <artifactId>lombok</artifactId>
    <optional>true</optional>
</dependency>

<!-- Apache Commons -->
<dependency>
    <groupId>org.apache.commons</groupId>
    <artifactId>commons-lang3</artifactId>
</dependency>

<!-- Guava -->
<dependency>
    <groupId>com.google.guava</groupId>
    <artifactId>guava</artifactId>
    <version>33.0.0-jre</version>
</dependency>

<!-- Jackson 额外模块 -->
<dependency>
    <groupId>com.fasterxml.jackson.dataformat</groupId>
    <artifactId>jackson-dataformat-xml</artifactId>
</dependency>
```

### ORM 和数据库

```xml
<!-- MyBatis -->
<dependency>
    <groupId>org.mybatis.spring.boot</groupId>
    <artifactId>mybatis-spring-boot-starter</artifactId>
    <version>3.0.3</version>
</dependency>

<!-- QueryDSL JPA -->
<dependency>
    <groupId>com.querydsl</groupId>
    <artifactId>querydsl-jpa</artifactId>
    <classifier>jakarta</classifier>
</dependency>

<!-- Liquibase 数据库版本控制 -->
<dependency>
    <groupId>org.liquibase</groupId>
    <artifactId>liquibase-core</artifactId>
</dependency>

<!-- Flyway 数据库迁移 -->
<dependency>
    <groupId>org.flywaydb</groupId>
    <artifactId>flyway-core</artifactId>
</dependency>
```

### Web 和 API

```xml
<!-- Swagger/OpenAPI -->
<dependency>
    <groupId>org.springdoc</groupId>
    <artifactId>springdoc-openapi-starter-webmvc-ui</artifactId>
    <version>2.0.4</version>
</dependency>

<!-- MapStruct 对象映射 -->
<dependency>
    <groupId>org.mapstruct</groupId>
    <artifactId>mapstruct</artifactId>
    <version>1.5.5.Final</version>
</dependency>

<!-- Modelmapper -->
<dependency>
    <groupId>org.modelmapper</groupId>
    <artifactId>modelmapper</artifactId>
    <version>3.2.0</version>
</dependency>
```

### 日志和监控

```xml
<!-- Logback extras -->
<dependency>
    <groupId>net.logstash.logback</groupId>
    <artifactId>logstash-logback-encoder</artifactId>
    <version>7.4</version>
</dependency>

<!-- Micrometer 监控 -->
<dependency>
    <groupId>io.micrometer</groupId>
    <artifactId>micrometer-core</artifactId>
</dependency>

<!-- Prometheus 指标 -->
<dependency>
    <groupId>io.micrometer</groupId>
    <artifactId>micrometer-registry-prometheus</artifactId>
</dependency>
```

### 缓存

```xml
<!-- Caffeine 本地缓存 -->
<dependency>
    <groupId>com.github.ben-manes.caffeine</groupId>
    <artifactId>caffeine</artifactId>
</dependency>

<!-- Jedis Redis 客户端 -->
<dependency>
    <groupId>redis.clients</groupId>
    <artifactId>jedis</artifactId>
</dependency>
```

### 消息队列

```xml
<!-- RabbitMQ -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-amqp</artifactId>
</dependency>

<!-- Kafka -->
<dependency>
    <groupId>org.springframework.kafka</groupId>
    <artifactId>spring-kafka</artifactId>
</dependency>
```

## 使用 BOM 管理版本

创建自己的 BOM：

```xml
<project>
    <groupId>com.example</groupId>
    <artifactId>myapp-bom</artifactId>
    <version>1.0.0</version>
    <packaging>pom</packaging>

    <dependencyManagement>
        <dependencies>
            <!-- 第三方库版本 -->
            <dependency>
                <groupId>com.google.guava</groupId>
                <artifactId>guava</artifactId>
                <version>33.0.0-jre</version>
            </dependency>
        </dependencies>
    </dependencyManagement>
</project>
```

在项目中使用：

```xml
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>com.example</groupId>
            <artifactId>myapp-bom</artifactId>
            <version>1.0.0</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>
```

## Maven 插件配置

### Spring Boot Maven 插件

```xml
<build>
    <plugins>
        <plugin>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-maven-plugin</artifactId>
            <executions>
                <execution>
                    <goals>
                        <goal>repackage</goal>
                    </goals>
                </execution>
            </executions>
            <configuration>
                <excludes>
                    <exclude>
                        <groupId>org.projectlombok</groupId>
                        <artifactId>lombok</artifactId>
                    </exclude>
                </excludes>
            </configuration>
        </plugin>
    </plugins>
</build>
```

### 编译器配置

```xml
<properties>
    <java.version>17</java.version>
    <maven.compiler.source>17</maven.compiler.source>
    <maven.compiler.target>17</maven.compiler.target>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
</properties>
```

## 依赖安全

### 检查已知漏洞

```bash
# 使用 OWASP 依赖检查
mvn dependency-check:check
```

### 定期更新依赖

```bash
# 检查可用的更新
mvn versions:display-dependency-updates

# 更新到最新版本
mvn versions:use-latest-releases
```

## 最佳实践

1. **使用 Spring Boot Starters** - 简化配置，统一管理版本
2. **明确指定版本** - 避免版本冲突和不确定性
3. **定期更新** - 及时获取安全补丁和性能优化
4. **排除不需要的依赖** - 减少应用大小，避免冲突
5. **使用 Maven BOM** - 统一管理企业级项目的依赖版本
6. **监控安全漏洞** - 定期检查依赖中的已知漏洞

下一步学习 [缓存管理](./cache-management.md)。
