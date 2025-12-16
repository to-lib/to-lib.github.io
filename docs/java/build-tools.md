---
sidebar_position: 3
title: Maven/Gradle 构建与编译（JDK 1.8）
---

# Maven/Gradle 构建与编译（JDK 1.8）

> [!IMPORTANT]
> 本目录默认以 **JDK 1.8 (Java 8)** 为基线。本页重点是让你的项目在本地/CI 上能够稳定地以 Java 8 编译、测试、打包。

## 你需要统一的“Java 版本”到底是什么

在工程化场景中，通常要同时约束三件事：

- **JDK 运行版本**：你用什么版本的 `java/javac` 来编译（例如 JDK 8）
- **源码级别（source）**：你写的语法上限（例如 `1.8`）
- **字节码级别（target）**：编译产物能在什么 JRE 上运行（例如 `1.8`）

只要 `target=1.8`，产物原则上就能在 Java 8 运行时运行。

## Maven（推荐：固定 source/target=1.8）

### 最小可用配置：maven-compiler-plugin

在 `pom.xml` 中增加：

```xml
<properties>
    <maven.compiler.source>1.8</maven.compiler.source>
    <maven.compiler.target>1.8</maven.compiler.target>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
</properties>

<build>
    <plugins>
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-compiler-plugin</artifactId>
            <version>3.11.0</version>
            <configuration>
                <source>${maven.compiler.source}</source>
                <target>${maven.compiler.target}</target>
                <encoding>${project.build.sourceEncoding}</encoding>
            </configuration>
        </plugin>
    </plugins>
</build>
```

### 让 CI / 团队更一致：Maven Wrapper

建议为项目生成 wrapper：

- `mvn -N -q wrapper:wrapper`

生成后使用：

- `./mvnw -v`
- `./mvnw clean test`

## Gradle（固定 sourceCompatibility=1.8）

### Groovy DSL（build.gradle）

```groovy
plugins {
    id 'java'
}

group = 'com.example'
version = '1.0.0'

java {
    sourceCompatibility = JavaVersion.VERSION_1_8
    targetCompatibility = JavaVersion.VERSION_1_8
}

tasks.withType(JavaCompile) {
    options.encoding = 'UTF-8'
}
```

### Kotlin DSL（build.gradle.kts）

```kotlin
plugins {
    java
}

group = "com.example"
version = "1.0.0"

java {
    sourceCompatibility = JavaVersion.VERSION_1_8
    targetCompatibility = JavaVersion.VERSION_1_8
}

tasks.withType<JavaCompile> {
    options.encoding = "UTF-8"
}
```

### Gradle Wrapper（强烈建议）

- `gradle wrapper`

然后在团队/CI 中统一用：

- `./gradlew test`
- `./gradlew build`

## 打包（Jar）

### Maven 打普通 Jar

- `mvn clean package`

产物通常在：

- `target/*.jar`

### Maven 打可执行 Fat Jar（Shade Plugin）

如果你需要把依赖一起打进 jar（常见于命令行工具/简单服务）：

```xml
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-shade-plugin</artifactId>
    <version>3.5.0</version>
    <executions>
        <execution>
            <phase>package</phase>
            <goals>
                <goal>shade</goal>
            </goals>
            <configuration>
                <transformers>
                    <transformer implementation="org.apache.maven.plugins.shade.resource.ManifestResourceTransformer">
                        <mainClass>com.example.Main</mainClass>
                    </transformer>
                </transformers>
            </configuration>
        </execution>
    </executions>
</plugin>
```

> [!TIP]
> `mainClass` 需要替换为你的程序入口类。

## 常见问题排查

### 1) Unsupported major.minor version / Unsupported class file major version

- **现象**：运行时报 “Unsupported major.minor version” 或 “Unsupported class file major version xx”
- **原因**：你用更高版本 JDK 编译出了更高字节码版本，但运行时还是 Java 8
- **处理**：确保构建配置 `target=1.8`（或 Gradle 的 `targetCompatibility=1.8`），并检查 CI 的 JDK 版本

### 2) 编译通过但线上异常：依赖与运行时 JDK 不兼容

- **现象**：依赖里使用了更高版本 JDK 才有的 API（例如 `HttpClient` 是 JDK 11+）
- **处理**：对外发布/运行在 Java 8 的项目，依赖选择要保证 Java 8 兼容；新特性建议通过“多模块/多版本”方式隔离

## 下一步

完成构建配置后，建议继续阅读：

- [开发环境搭建](/docs/java/environment-setup)
- [Java 最佳实践](/docs/java/best-practices)
- [Java 快速参考](/docs/java/quick-reference)
