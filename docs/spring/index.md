---
id: spring-index
title: Spring Framework 学习指南
sidebar_label: 概览
sidebar_position: 1
---

# Spring Framework 学习指南

Spring是一个开源的Java企业级应用开发框架，提供了全方位的基础设施支持。本学习指南涵盖了Spring框架的核心概念、最常用的功能模块和最佳实践。

## 📚 学习路径

### 基础部分
- **[Spring核心概念](./core-concepts.md)** - IoC、DI、Bean生命周期等基础概念
- **[依赖注入详解](./dependency-injection.md)** - 深入理解DI的各种方式和最佳实践
- **[Bean管理](./bean-management.md)** - Bean的定义、作用域、生命周期管理

### 核心特性
- **[面向切面编程(AOP)](./aop.md)** - 切点、通知、代理等AOP核心概念
- **[事务管理](./transactions.md)** - 事务特性、传播行为、隔离级别
- **[数据访问](./data-access.md)** - JdbcTemplate、ORM集成、事务处理

### Web应用开发
- **[Spring MVC](./spring-mvc.md)** - MVC架构、请求处理流程、视图解析
- **[REST API开发](./rest-api.md)** - RESTful设计原则、请求映射、数据绑定
- **[异常处理](./exception-handling.md)** - 全局异常处理、自定义异常

### 高级特性
- **[配置与属性](./configuration.md)** - JavaConfig、注解配置、属性绑定
- **[事件与监听](./events.md)** - 应用事件、事件监听器、事件发布
- **[Spring Boot集成](./spring-boot-integration.md)** - 与Spring Boot的集成、自动配置

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
