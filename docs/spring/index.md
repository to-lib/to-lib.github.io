---
id: spring-index
title: Spring Framework 学习指南
sidebar_label: 概览
sidebar_position: 1
---

# Spring Framework 学习指南

> [!TIP] > **Spring 框架基础**: Spring 是 Java 企业应用开发的基础框架。深入理解 IoC、DI 和 AOP 是掌握 Spring 的关键，建议从 [核心概念](/docs/spring/core-concepts) 开始学习。

## 📚 学习路径

### 基础部分

- **[Spring 核心概念](/docs/spring/core-concepts)** - IoC、DI、Bean 生命周期等基础概念
- **[依赖注入详解](/docs/spring/dependency-injection)** - 深入理解 DI 的各种方式和最佳实践
- **[Bean 管理](/docs/spring/bean-management)** - Bean 的定义、作用域、生命周期管理
- **[配置与 Profiles](/docs/spring/configuration)** - JavaConfig、外部化配置、环境隔离与模块化配置

### 核心特性

- **[面向切面编程(AOP)](/docs/spring/aop)** - 切点、通知、代理等 AOP 核心概念
- **[事务管理](/docs/spring/transactions)** - 事务特性、传播行为、隔离级别
- **[事件机制](/docs/spring/events)** - 事件发布与监听、异步事件、事务事件
- **[资源管理](/docs/spring/resource-management)** - Resource 接口、ResourceLoader、配置文件加载
- **[SpEL 表达式](/docs/spring/spel)** - 表达式语法、注解场景（缓存/事件/注入）与最佳实践
- **[缓存抽象](/docs/spring/caching)** - Spring Cache 核心注解、key/condition/unless 与常见坑

### Web 应用开发

- **[Spring MVC](/docs/spring/spring-mvc)** - MVC 架构、请求处理流程、视图解析

### 数据访问

- **[Spring Data](/docs/spring/spring-data)** - Repository 模式、查询方法、分页排序、审计功能

### 安全与测试

- **[安全基础](/docs/spring/security-basics)** - Spring Security 认证授权、用户管理、方法安全
- **[测试](/docs/spring/testing)** - 单元测试、集成测试、MockMvc、数据层测试
- **[参数校验](/docs/spring/validation)** - Bean Validation、方法级校验、分组校验与自定义约束

### 快速参考

- **[快速参考](/docs/spring/quick-reference)** - 常用注解、配置方式、代码片段
- **[常见问题解答](/docs/spring/faq)** - 常见问题及解决方案
- **[最佳实践](/docs/spring/best-practices)** - Spring 应用开发的最佳实践
- **[面试题集](/docs/interview/spring-interview-questions)** - Spring 相关面试题精选

## 🎯 核心概念速览

### IoC (Inversion of Control)

控制反转 - 将对象的创建和管理权交给容器，而不是由程序员手动管理。

### DI (Dependency Injection)

依赖注入 - IoC 的具体实现方式，通过注入依赖而不是在类内部创建依赖。

### AOP (Aspect-Oriented Programming)

面向切面编程 - 将横切关注点（如日志、事务）从业务逻辑中分离出来。

### Bean

Spring 中的对象，由容器管理其生命周期、依赖关系等。

## 🔧 常用注解速览

| 注解                     | 说明              |
| ------------------------ | ----------------- |
| `@Configuration`         | 标记配置类        |
| `@Bean`                  | 声明 Bean         |
| `@Component`             | 通用组件注解      |
| `@Service`               | 业务逻辑层组件    |
| `@Repository`            | 数据访问层组件    |
| `@Controller`            | 控制层组件        |
| `@Autowired`             | 自动装配          |
| `@Qualifier`             | 指定要装配的 Bean |
| `@Scope`                 | 指定 Bean 作用域  |
| `@Transactional`         | 声明式事务        |
| `@Aspect`                | 标记切面类        |
| `@Before/@After/@Around` | 通知类型          |

## 📖 学习资源

- [Spring 官方文档](https://spring.io/projects/spring-framework)
- [Spring GitHub 仓库](https://github.com/spring-projects/spring-framework)
- [Spring 社区论坛](https://spring.io/community)

## 🚀 下一步

选择上面的任意主题开始学习，建议按照学习路径的顺序进行学习。

---

**最后更新**: 2025 年 12 月  
**版本**: Spring Framework 6.x
