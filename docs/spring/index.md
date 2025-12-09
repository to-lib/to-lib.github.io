---
id: spring-index
title: Spring Framework 学习指南
sidebar_label: 概览
sidebar_position: 1
---

# Spring Framework 学习指南

> [!TIP]
> **Spring 框架基础**: Spring 是 Java 企业应用开发的基础框架。深入理解 IoC、DI 和 AOP 是掌握 Spring 的关键，建议从 [核心概念](./core-concepts.md) 开始学习。

## 📚 学习路径

### 基础部分

- **[Spring核心概念](./core-concepts.md)** - IoC、DI、Bean生命周期等基础概念
- **[依赖注入详解](./dependency-injection.md)** - 深入理解DI的各种方式和最佳实践
- **[Bean管理](./bean-management.md)** - Bean的定义、作用域、生命周期管理

### 核心特性

- **[面向切面编程(AOP)](./aop.md)** - 切点、通知、代理等AOP核心概念
- **[事务管理](./transactions.md)** - 事务特性、传播行为、隔离级别

### Web应用开发

- **[Spring MVC](./spring-mvc.md)** - MVC架构、请求处理流程、视图解析

### 快速参考

- **[快速参考](./quick-reference.md)** - 常用注解、配置方式、代码片段
- **[常见问题解答](./faq.md)** - 常见问题及解决方案
- **[最佳实践](./best-practices.md)** - Spring应用开发的最佳实践

## 🎯 核心概念速览

### IoC (Inversion of Control)

控制反转 - 将对象的创建和管理权交给容器，而不是由程序员手动管理。

### DI (Dependency Injection)

依赖注入 - IoC的具体实现方式，通过注入依赖而不是在类内部创建依赖。

### AOP (Aspect-Oriented Programming)

面向切面编程 - 将横切关注点（如日志、事务）从业务逻辑中分离出来。

### Bean

Spring中的对象，由容器管理其生命周期、依赖关系等。

## 🔧 常用注解速览

| 注解 | 说明 |
|------|------|
| `@Configuration` | 标记配置类 |
| `@Bean` | 声明Bean |
| `@Component` | 通用组件注解 |
| `@Service` | 业务逻辑层组件 |
| `@Repository` | 数据访问层组件 |
| `@Controller` | 控制层组件 |
| `@Autowired` | 自动装配 |
| `@Qualifier` | 指定要装配的Bean |
| `@Scope` | 指定Bean作用域 |
| `@Transactional` | 声明式事务 |
| `@Aspect` | 标记切面类 |
| `@Before/@After/@Around` | 通知类型 |

## 📖 学习资源

- [Spring官方文档](https://spring.io/projects/spring-framework)
- [Spring GitHub仓库](https://github.com/spring-projects/spring-framework)
- [Spring社区论坛](https://spring.io/community)

## 🚀 下一步

选择上面的任意主题开始学习，建议按照学习路径的顺序进行学习。

---

**最后更新**: 2025年12月  
**版本**: Spring Framework 6.x
