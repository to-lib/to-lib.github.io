---
sidebar_position: 1
title: Java 编程概述
---

# Java 编程

欢迎来到 Java 编程完整学习指南！本指南涵盖了 Java 编程语言的核心知识和实践技巧。

> [!IMPORTANT]
> 本教程基于 **JDK 1.8 (Java 8)** 版本，这是目前企业中最广泛使用的 Java 版本。开始学习前，请先完成 [开发环境搭建](/docs/java/environment-setup)。

## 📚 学习内容

### 基础知识

- **基础语法** - 数据类型、变量、运算符、流程控制、字符串、数组
- **面向对象** - 类、对象、封装、继承、多态、内部类、枚举、注解
- **异常处理** - try-catch、自定义异常、最佳实践

### 核心特性

- **集合框架** - List、Set、Map 及其实现类
- **泛型编程** - 泛型类、泛型方法、通配符
- **正则表达式** - Pattern、Matcher、常见验证场景
- **反射机制** - Class 类、动态代理、元数据处理

### 日期时间与字符串

- **字符串处理** - String、StringBuilder、StringBuffer
- **日期时间 API** - LocalDate、LocalTime、LocalDateTime、格式化

### 高级主题

- **多线程** - Thread、线程同步、线程池、并发工具类
- **IO 流** - 字节流、字符流、缓冲流、NIO
- **函数式编程** - Lambda 表达式、Stream API
- **反射与注解** - 反射 API、自定义注解、注解处理
- **JVM 基础** - JVM 架构、类加载、内存模型、垃圾回收
- **性能优化** - 代码优化、并发优化、IO 优化、调优技巧
- **JDK 8 新特性** - Lambda、Stream、Optional、新日期时间 API
- **JDK 11 新特性** - HTTP Client、String 增强、文件读写增强
- **JDK 17 新特性** - 密封类、记录类型、模式匹配、文本块
- **JDK 21 新特性** - 虚拟线程、序列化集合、增强的模式匹配

## 🚀 快速开始

如果你是 Java 初学者，建议按以下顺序学习：

1. [开发环境搭建](/docs/java/environment-setup) - 安装配置 JDK 1.8
2. [构建与编译（Maven/Gradle）](/docs/java/build-tools) - 固定 Java 8 编译配置，打包与常用命令
3. [基础语法](/docs/java/basic-syntax) - 掌握 Java 基本语法和字符串处理
4. [面向对象](/docs/java/oop) - 理解面向对象编程思想、枚举和注解
5. [集合框架](/docs/java/collections) - 学习常用数据结构
6. [异常处理](/docs/java/exception-handling) - 掌握异常处理机制
7. [多线程](/docs/java/multithreading) - 了解并发编程基础

## 📖 学习路径

> [!TIP]
> 查看完整的 [**Java 学习路线**](/docs/java/learning-path)，获取从零到高级后端的详细学习计划。

### 阶段概览

| 阶段       | 内容                                     | 预计时间 |
| ---------- | ---------------------------------------- | -------- |
| **阶段 1** | Java 基础语法                            | 1-2 周   |
| **阶段 2** | Java 核心进阶（JVM、泛型、反射、Stream） | 2-4 周   |
| **阶段 3** | Java 并发编程                            | 4-6 周   |
| **阶段 4** | 数据库与持久化（MySQL、MyBatis）         | 2-4 周   |
| **阶段 5** | Spring 全家桶                            | 4-8 周   |
| **阶段 6** | Netty 与网络编程                         | 4-6 周   |
| **阶段 7** | 分布式系统                               | 6-12 周  |
| **阶段 8** | 工程化与 DevOps                          | 持续提升 |
| **阶段 9** | 高频面试方向                             | 随时补充 |

### 快速定位

- **初级开发者**（2-3 个月）：阶段 1-2 + Spring Boot 基础
- **中级开发者**（6-8 个月）：阶段 1-5 + 基础分布式
- **高级开发者**（12-18 个月）：全部阶段 + 源码阅读 + 系统设计

## 💡 最佳实践

本指南不仅包含理论知识，还提供：

- ✅ 实用代码示例
- ✅ 常见问题解答
- ✅ 性能优化建议
- ✅ 编码规范指南
- ✅ 实战项目案例

## 📚 完整学习资源

| 主题                                                 | 描述                                              |
| ---------------------------------------------------- | ------------------------------------------------- |
| [🧭 学习路线](/docs/java/learning-path)              | **从零到高级后端的完整学习路径**                  |
| [基础语法](/docs/java/basic-syntax)                  | 数据类型、变量、运算符、流程控制、字符串处理      |
| [构建与编译（Maven/Gradle）](/docs/java/build-tools) | Java 8 工程构建、编译配置、打包与常见问题排查     |
| [面向对象](/docs/java/oop)                           | 类、对象、继承、多态、内部类、枚举、注解          |
| [集合框架](/docs/java/collections)                   | List、Set、Map 详解和性能对比                     |
| [异常处理](/docs/java/exception-handling)            | 异常分类、处理机制、最佳实践                      |
| [泛型编程](/docs/java/generics)                      | 泛型类、泛型方法、通配符、类型擦除                |
| [多线程](/docs/java/multithreading)                  | 线程创建、同步、通信、线程池、并发工具类          |
| [IO 流](/docs/java/io-streams)                       | 字节流、字符流、缓冲流、NIO 详解                  |
| [函数式编程](/docs/java/functional-programming)      | Lambda、Stream API、函数式接口                    |
| [字符串与日期](/docs/java/date-time)                 | LocalDate、LocalTime、格式化、时间计算            |
| [反射与注解](/docs/java/reflection-annotation)       | 反射 API、动态代理、自定义注解                    |
| [正则表达式](/docs/java/regex)                       | 正则语法、Pattern、Matcher、常见验证              |
| [JVM 基础](/docs/java/jvm-basics)                    | JVM 架构、类加载、内存模型、垃圾回收、调优        |
| [性能优化](/docs/java/performance)                   | 代码优化、并发优化、缓存策略、监控分析            |
| [JDK 17 新特性](/docs/java/jdk17-features)           | 密封类、记录类型、模式匹配、文本块、Switch 表达式 |
| [JDK 21 新特性](/docs/java/jdk21-features)           | 虚拟线程、序列化集合、增强模式匹配、记录模式      |

## 🔗 相关资源

### 框架与中间件

- [Java 设计模式](/docs/java-design-patterns) - 23 种设计模式详解
- [Spring Framework](/docs/spring) - Spring 核心框架
- [Spring Boot](/docs/springboot) - 快速构建企业应用
- [Spring Cloud](/docs/springcloud) - 微服务架构
- [Netty](/docs/netty) - 高性能网络框架

### 数据库与缓存

- [MySQL](/docs/mysql) - 关系型数据库
- [MyBatis](/docs/mybatis) - 持久层框架
- [Redis](/docs/redis) - 高性能缓存
- [PostgreSQL](/docs/postgres) - 开源数据库

### 消息队列

- [Kafka](/docs/kafka) - 分布式消息系统
- [RocketMQ](/docs/rocketmq) - 阿里消息中间件
- [RabbitMQ](/docs/rabbitmq) - 消息代理

### 容器与运维

- [Docker](/docs/docker) - 容器化技术
- [Kubernetes](/docs/kubernetes) - 容器编排
- [Linux](/docs/linux) - 服务器基础

开始你的 Java 学习之旅吧！🚀
