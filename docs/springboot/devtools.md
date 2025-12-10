---
sidebar_position: 23
---

# 开发者工具

> [!TIP]
> **DevTools 的作用**: Spring Boot DevTools 提供了快速重启、LiveReload、远程调试等功能，大幅提升开发效率。

## 添加依赖

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-devtools</artifactId>
    <scope>runtime</scope>
    <optional>true</optional>
</dependency>
```

## 核心功能

### 1. 自动重启

当 classpath 文件发生变化时，应用会自动重启。

```yaml
spring:
  devtools:
    restart:
      enabled: true
      additional-paths: src/main/resources
      exclude: static/**,public/**,META-INF/maven/**,META-INF/resources/**
```

**工作原理**：

- DevTools 使用两个类加载器
- **base classloader**: 加载不变的类（第三方jar）
- **restart classloader**: 加载开发中的类
- 重启时只重新创建 restart classloader，速度快

### 2. LiveReload

自动刷新浏览器页面。

```yaml
spring:
  devtools:
    livereload:
      enabled: true
      port: 35729
```

安装浏览器插件：

- Chrome: LiveReload Extension
- Firefox: LiveReload Add-on

### 3. 全局配置

在用户主目录创建 `.spring-boot-devtools.properties`：

```properties
# Windows: C:\Users\username\.spring-boot-devtools.properties
# Linux/Mac: ~/.spring-boot-devtools.properties

spring.devtools.restart.additional-paths=../other-project/src/main/java
spring.devtools.restart.exclude=static/**,public/**
```

### 4. 远程调试

```yaml
spring:
  devtools:
    remote:
      secret: mysecret
      restart:
        enabled: true
```

启动远程应用：

```bash
java -jar myapp.jar --spring.devtools.remote.secret=mysecret
```

在 IDE 中运行：

```java
org.springframework.boot.devtools.RemoteSpringApplication
```

## IntelliJ IDEA 配置

### 启用自动编译

1. **Settings** → **Build, Execution, Deployment** → **Compiler**
2. 勾选 **Build project automatically**

### 启用运行时自动编译

1. **Help** → **Find Action** (Ctrl+Shift+A / Cmd+Shift+A)
2. 搜索 **Registry**
3. 勾选 **compiler.automake.allow.when.app.running**

## 最佳实践

> [!TIP]
> **DevTools 使用技巧**：
>
> 1. **仅在开发环境使用** - 生产环境自动禁用
> 2. **排除不需要重启的资源** - 如静态文件
> 3. **配合 Lombok** - 修改实体类自动生成getter/setter
> 4. **使用 LiveReload** - 前端开发时自动刷新浏览器

## 总结

- **自动重启** - 修改代码自动重启应用
- **LiveReload** - 自动刷新浏览器
- **远程调试** - 调试远程应用
- **提升效率** - 开发体验显著提升
