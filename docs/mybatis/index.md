---
sidebar_position: 1
title: MyBatis 持久层框架学习指南
---

# MyBatis 持久层框架

欢迎来到 MyBatis 持久层框架完整学习指南！本指南涵盖了 MyBatis 从基础到高级的核心知识和实践技巧。

> [!TIP]
> **MyBatis 是什么？** MyBatis 是一款优秀的持久层框架，它支持自定义 SQL、存储过程以及高级映射。MyBatis 免除了几乎所有的 JDBC 代码以及设置参数和获取结果集的工作。建议从 [核心概念](/docs/mybatis/core-concepts) 开始学习。

## 📚 学习内容

### 基础知识

- **核心概念** - MyBatis 架构、SqlSessionFactory、SqlSession、Mapper 接口
- **配置详解** - mybatis-config.xml 配置文件结构和常用配置项
- **XML 映射** - Mapper XML 文件、CRUD 操作、resultMap 映射

### 核心特性

- **动态 SQL** - if、choose、where、set、foreach 等动态标签
- **注解映射** - @Select、@Insert、@Update、@Delete 注解方式
- **缓存机制** - 一级缓存、二级缓存原理与配置
- **插件机制** - 拦截器原理、自定义插件开发

### 企业应用

- **Spring 集成** - MyBatis-Spring、Spring Boot Starter 配置
- **最佳实践** - Mapper 设计规范、SQL 编写规范、性能优化

## 🎯 核心特性

| 特性 | 描述 |
|------|------|
| **灵活的 SQL 控制** | 直接编写原生 SQL，完全掌控 SQL 执行 |
| **强大的映射能力** | 支持复杂的结果集映射，包括一对一、一对多关联 |
| **动态 SQL** | 根据条件动态生成 SQL，避免字符串拼接 |
| **插件扩展** | 通过拦截器机制扩展功能，如分页、SQL 监控 |
| **缓存支持** | 内置一级、二级缓存，支持第三方缓存集成 |
| **Spring 集成** | 与 Spring/Spring Boot 无缝集成 |

## 🚀 快速开始

如果你是 MyBatis 初学者，建议按以下顺序学习：

1. [核心概念](/docs/mybatis/core-concepts) - 了解 MyBatis 架构和核心组件
2. [配置详解](/docs/mybatis/configuration) - 掌握配置文件结构
3. [XML 映射](/docs/mybatis/xml-mapping) - 学习 Mapper XML 编写
4. [动态 SQL](/docs/mybatis/dynamic-sql) - 掌握动态 SQL 标签
5. [Spring 集成](/docs/mybatis/spring-integration) - 在 Spring Boot 中使用 MyBatis

## 📖 学习路径

### 初级开发者（1-2 周）

- MyBatis 核心概念和架构
- 基本配置和环境搭建
- XML 映射基础（CRUD 操作）
- 简单的 resultMap 配置
- Spring Boot 集成基础

### 中级开发者（2-4 周）

- 动态 SQL 完整掌握
- 复杂映射（一对一、一对多）
- 注解方式开发
- 一级缓存和二级缓存
- 事务管理配合
- 多数据源配置

### 高级开发者（4-6 周）

- 插件机制和自定义插件
- 源码原理分析
- 性能优化策略
- MyBatis-Plus 扩展
- 分页插件原理
- 批量操作优化

## 📚 完整学习资源

| 主题 | 描述 |
|------|------|
| [核心概念](/docs/mybatis/core-concepts) | MyBatis 架构、核心组件、工作流程 |
| [配置详解](/docs/mybatis/configuration) | 配置文件结构、常用配置项、环境配置 |
| [XML 映射](/docs/mybatis/xml-mapping) | Mapper XML、CRUD 标签、resultMap、参数映射 |
| [动态 SQL](/docs/mybatis/dynamic-sql) | if、choose、where、set、foreach、sql 片段 |
| [注解映射](/docs/mybatis/annotations) | @Select、@Results、@Provider 系列注解 |
| [缓存机制](/docs/mybatis/caching) | 一级缓存、二级缓存、第三方缓存集成 |
| [Spring 集成](/docs/mybatis/spring-integration) | MyBatis-Spring、Spring Boot Starter、事务管理 |
| [插件机制](/docs/mybatis/plugins) | 拦截器原理、自定义插件、PageHelper、MyBatis-Plus |
| [最佳实践](/docs/mybatis/best-practices) | Mapper 设计规范、SQL 规范、性能优化 |
| [快速参考](/docs/mybatis/quick-reference) | 配置速查、标签速查、注解速查、代码片段 |
| [常见问题](/docs/mybatis/faq) | 配置问题、映射问题、性能问题、集成问题 |
| [面试题集](/docs/interview/mybatis-interview-questions) | MyBatis 常见面试题和答案详解 |

## 🔧 MyBatis vs JDBC vs JPA

| 特性 | JDBC | MyBatis | JPA/Hibernate |
|------|------|---------|---------------|
| SQL 控制 | 完全手写 | 手写 SQL | 自动生成 |
| 学习曲线 | 低 | 中 | 高 |
| 开发效率 | 低 | 中 | 高 |
| 灵活性 | 高 | 高 | 中 |
| 复杂查询 | 繁琐 | 方便 | 需要 JPQL/原生 SQL |
| 缓存支持 | 无 | 一级/二级缓存 | 一级/二级缓存 |
| 适用场景 | 简单项目 | 复杂 SQL 场景 | 标准 CRUD 场景 |

## 💡 为什么选择 MyBatis？

1. **SQL 可控** - 直接编写 SQL，便于优化和调试
2. **学习成本低** - 相比 JPA/Hibernate，更容易上手
3. **灵活性高** - 适合复杂查询和报表场景
4. **生态丰富** - PageHelper、MyBatis-Plus 等插件支持
5. **国内主流** - 国内企业广泛使用，面试必备

## 🔗 相关资源

### Java 基础

- [Java 编程](/docs/java) - Java 语言基础
- [Java 设计模式](/docs/java-design-patterns) - 设计模式详解

### Spring 生态

- [Spring Framework](/docs/spring) - Spring 核心框架
- [Spring Boot](/docs/springboot) - 快速构建企业应用
- [Spring Cloud](/docs/springcloud) - 微服务架构

### 数据库

- [MySQL](/docs/mysql) - MySQL 数据库
- [PostgreSQL](/docs/postgres) - PostgreSQL 数据库
- [Redis](/docs/redis) - 缓存数据库

### 面试准备

- [MyBatis 面试题](/docs/interview/mybatis-interview-questions) - MyBatis 面试题集
- [Java 面试题](/docs/interview/java-interview-questions) - Java 面试题集
- [Spring 面试题](/docs/interview/spring-interview-questions) - Spring 面试题集

## 📖 推荐学习资源

- [MyBatis 官方文档](https://mybatis.org/mybatis-3/zh/index.html)
- [MyBatis-Spring 官方文档](https://mybatis.org/spring/)
- [MyBatis-Plus 官方文档](https://baomidou.com/)
- [MyBatis GitHub 仓库](https://github.com/mybatis/mybatis-3)

---

**最后更新**: 2025 年 12 月  
**版本**: MyBatis 3.5.x

开始你的 MyBatis 学习之旅吧！🚀
