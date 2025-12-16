---
sidebar_position: 1
title: 环境配置指南
description: JDK 1.8 环境配置与开发工具设置
---

# 环境配置指南

本指南帮助你配置 Java 开发环境，以便运行设计模式示例代码。

## JDK 版本要求

本文档所有代码基于 **JDK 1.8 (Java 8)** 编写，请确保安装了正确的 JDK 版本。

:::info 为什么选择 JDK 1.8？

- **广泛使用**: JDK 1.8 是企业级应用最广泛使用的 Java 版本
- **新特性支持**: 支持 Lambda 表达式、Stream API、函数式接口等现代特性
- **稳定可靠**: 自 2014 年发布，经过多年验证，非常稳定
- **长期支持**: Oracle 和多个供应商提供长期支持（LTS）
  :::

## JDK 安装

### Windows

1. **下载 JDK**

   - 访问 [Oracle JDK 下载页面](https://www.oracle.com/java/technologies/javase/javase8-archive-downloads.html) 或 [Adoptium](https://adoptium.net/)
   - 选择 Windows x64 版本

2. **安装 JDK**

   - 运行下载的安装程序
   - 默认安装路径为 `C:\Program Files\Java\jdk1.8.0_xxx`

3. **配置环境变量**

   ```batch
   # 设置 JAVA_HOME
   setx JAVA_HOME "C:\Program Files\Java\jdk1.8.0_xxx"

   # 添加到 PATH
   setx PATH "%PATH%;%JAVA_HOME%\bin"
   ```

4. **验证安装**
   ```batch
   java -version
   javac -version
   ```

### macOS

1. **使用 Homebrew 安装**

   ```bash
   # 安装 Homebrew (如果尚未安装)
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

   # 安装 JDK 8
   brew install --cask temurin@8
   ```

2. **配置环境变量**

   ```bash
   # 编辑 ~/.zshrc 或 ~/.bash_profile
   export JAVA_HOME=$(/usr/libexec/java_home -v 1.8)
   export PATH=$JAVA_HOME/bin:$PATH
   ```

3. **验证安装**
   ```bash
   java -version
   javac -version
   ```

### Linux (Ubuntu/Debian)

1. **安装 OpenJDK 8**

   ```bash
   sudo apt update
   sudo apt install openjdk-8-jdk
   ```

2. **配置环境变量**

   ```bash
   # 编辑 ~/.bashrc
   export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
   export PATH=$JAVA_HOME/bin:$PATH

   # 使配置生效
   source ~/.bashrc
   ```

3. **验证安装**
   ```bash
   java -version
   javac -version
   ```

## IDE 配置

### IntelliJ IDEA

1. **创建项目**

   - File → New → Project
   - 选择 Java，Project SDK 选择 1.8

2. **配置项目 SDK**

   - File → Project Structure → Project
   - Project SDK: 选择 1.8
   - Project language level: 选择 8 - Lambdas, type annotations etc.

3. **推荐插件**
   - **Lombok**: 简化 POJO 类
   - **PlantUML Integration**: UML 图渲染
   - **CheckStyle-IDEA**: 代码风格检查

### Eclipse

1. **配置 JDK**

   - Window → Preferences → Java → Installed JREs
   - Add → Standard VM → 选择 JDK 1.8 路径

2. **创建项目**

   - File → New → Java Project
   - JRE 选择 JavaSE-1.8

3. **编译器设置**
   - Window → Preferences → Java → Compiler
   - Compiler compliance level: 1.8

### VS Code

1. **安装扩展**

   - Extension Pack for Java
   - Debugger for Java
   - Language Support for Java

2. **配置 settings.json**
   ```json
   {
     "java.configuration.runtimes": [
       {
         "name": "JavaSE-1.8",
         "path": "/path/to/jdk1.8",
         "default": true
       }
     ]
   }
   ```

## Maven 项目配置

创建 `pom.xml` 文件：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.example</groupId>
    <artifactId>design-patterns</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>jar</packaging>

    <name>Java Design Patterns</name>
    <description>23种设计模式示例代码</description>

    <properties>
        <maven.compiler.source>1.8</maven.compiler.source>
        <maven.compiler.target>1.8</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>

    <dependencies>
        <!-- JUnit 5 for testing -->
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.9.3</version>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.11.0</version>
                <configuration>
                    <source>1.8</source>
                    <target>1.8</target>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

## Gradle 项目配置

创建 `build.gradle` 文件：

```groovy
plugins {
    id 'java'
}

group = 'com.example'
version = '1.0-SNAPSHOT'

java {
    sourceCompatibility = JavaVersion.VERSION_1_8
    targetCompatibility = JavaVersion.VERSION_1_8
}

repositories {
    mavenCentral()
}

dependencies {
    testImplementation 'org.junit.jupiter:junit-jupiter:5.9.3'
}

tasks.withType(JavaCompile) {
    options.encoding = 'UTF-8'
}

test {
    useJUnitPlatform()
}
```

## 验证环境

运行以下代码验证环境配置正确：

```java
public class EnvironmentCheck {
    public static void main(String[] args) {
        System.out.println("Java 版本: " + System.getProperty("java.version"));
        System.out.println("Java 供应商: " + System.getProperty("java.vendor"));
        System.out.println("Java Home: " + System.getProperty("java.home"));

        // 测试 Lambda 表达式 (JDK 8 特性)
        Runnable r = () -> System.out.println("Lambda 表达式正常工作！");
        r.run();

        // 测试 Stream API (JDK 8 特性)
        java.util.Arrays.asList("设计模式", "工厂模式", "单例模式")
            .stream()
            .filter(s -> s.contains("模式"))
            .forEach(System.out::println);

        System.out.println("\n✅ 环境配置正确，可以开始学习设计模式了！");
    }
}
```

## 常见问题

### Q: 如何切换多个 JDK 版本？

**Windows**: 使用 `scoop` 或手动修改 `JAVA_HOME` 环境变量

**macOS**: 使用 `jenv` 工具

```bash
brew install jenv
jenv add /path/to/jdk
jenv global 1.8
```

**Linux**: 使用 `update-alternatives`

```bash
sudo update-alternatives --config java
```

### Q: IDEA 提示 Language Level 不匹配？

确保 Project Structure → Modules → Sources 中的 Language Level 设置为 8。

### Q: Maven 编译报错 "source option 5 is no longer supported"？

在 `pom.xml` 中明确指定编译版本：

```xml
<properties>
    <maven.compiler.source>1.8</maven.compiler.source>
    <maven.compiler.target>1.8</maven.compiler.target>
</properties>
```

---

环境配置完成后，可以开始学习 [设计模式概述](overview)！
