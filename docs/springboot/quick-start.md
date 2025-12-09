---
sidebar_position: 2
---

# 快速开始

## 5 分钟创建第一个 Spring Boot 应用

> [!TIP]
> **首次使用？** 推荐使用 Spring Initializr [https://start.spring.io](https://start.spring.io) 创建项目，无需手动配置！

### 前置条件

- ✅ JDK 11 或更高版本
- ✅ Maven 3.6+ 或 Gradle 7.0+
- ✅ IDE（如 IntelliJ IDEA 或 VS Code）

> [!NOTE]
> **版本推荐**: 生产环境使用 JDK 17 或 21，Spring Boot 3.x 版本。

### 方法一：使用 Spring Initializr（推荐）

1. **访问官方网站**
   打开 [https://start.spring.io](https://start.spring.io)

2. **配置项目**
   - Project: Maven Project
   - Language: Java
   - Spring Boot: 3.x.x (最新版本)
   - Project Metadata:
     - Group: com.example
     - Artifact: hello-world
     - Name: hello-world
     - Description: My first Spring Boot app

3. **选择依赖**
   - Spring Web
   - Spring Boot DevTools

4. **生成并导入项目**

### 方法二：使用 Maven 命令

```bash
mvn archetype:generate \
  -DgroupId=com.example \
  -DartifactId=hello-world \
  -DarchetypeGroupId=org.apache.maven.archetypes \
  -DarchetypeArtifactId=maven-archetype-quickstart \
  -DinteractiveMode=false

cd hello-world
```

### 方法三：使用 IDE 创建项目

**IntelliJ IDEA:**

- File → New → Project
- 选择 Spring Boot
- 配置项目信息
- 选择 Spring Web 依赖

## 项目结构

创建项目后的标准结构：

```text
hello-world/
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/example/helloworld/
│   │   │       └── HelloWorldApplication.java
│   │   └── resources/
│   │       └── application.properties
│   └── test/
│       └── java/
│           └── com/example/helloworld/
│               └── HelloWorldApplicationTests.java
├── pom.xml
└── README.md
```

## 编写第一个 Controller

打开 `src/main/java/com/example/helloworld/HelloWorldApplication.java`，修改内容：

```java
package com.example.helloworld;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@SpringBootApplication
public class HelloWorldApplication {

    public static void main(String[] args) {
        SpringApplication.run(HelloWorldApplication.class, args);
    }
}

@RestController
class HelloController {
    
    @GetMapping("/")
    public String hello() {
        return "Hello, Spring Boot!";
    }
    
    @GetMapping("/api/greeting")
    public Greeting greeting(String name) {
        return new Greeting("Hello, " + (name != null ? name : "World") + "!");
    }
}

record Greeting(String message) {}
```

## 运行应用

### 使用 IDE

点击 IDE 中的 Run 按钮，或按 `Shift + F10` (IntelliJ) 或 `Ctrl + F5` (VS Code)

### 使用 Maven

```bash
mvn spring-boot:run
```

### 打包并运行

```bash
# 打包
mvn clean package

# 运行 JAR 文件
java -jar target/hello-world-0.0.1-SNAPSHOT.jar
```

## 测试应用

1. **访问应用**

   ```text
   http://localhost:8080/
   ```

   您应该看到：`Hello, Spring Boot!`

2. **测试 API**

   ```text
   http://localhost:8080/api/greeting?name=Spring Boot
   ```

   响应：`{"message":"Hello, Spring Boot!"}`

## 配置应用

编辑 `src/main/resources/application.properties`：

```properties
# 服务器端口
server.port=8080

# 应用名称
spring.application.name=hello-world

# 日志级别
logging.level.root=INFO
logging.level.com.example=DEBUG
```

或使用 YAML 格式 `application.yml`：

```yaml
server:
  port: 8080

spring:
  application:
    name: hello-world

logging:
  level:
    root: INFO
    com.example: DEBUG
```

## 添加更多依赖

编辑 `pom.xml`，在 `<dependencies>` 中添加：

```xml
<!-- 数据库支持 -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-jpa</artifactId>
</dependency>
<dependency>
    <groupId>com.h2database</groupId>
    <artifactId>h2</artifactId>
    <scope>runtime</scope>
</dependency>

<!-- 日志 -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-logging</artifactId>
</dependency>
```

## 常见问题

**Q: 如何关闭应用？**  
A: 按 `Ctrl + C` 关闭终端中运行的应用

**Q: 端口已被占用？**  
A: 在 `application.properties` 中修改：`server.port=8081`

**Q: 需要 Tomcat？**  
A: Spring Boot 已内置 Tomcat，无需额外配置

## 下一步

恭喜！你已经成功创建了第一个 Spring Boot 应用。接下来可以：

- 📚 学习 [核心概念](./core-concepts.md) - 深入理解 Spring Boot
- 🚀 构建 [RESTful API](./web-development.md) - 实现真实的 Web 服务
- 💾 集成 [数据库](./data-access.md) - 持久化数据

---

> [!TIP]
> **开发技巧**: Spring Boot DevTools 会在文件更改时自动重启应用，大大提高开发效率！通过添加 `spring-boot-devtools` 依赖启用。
