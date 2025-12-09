---
id: springcloud-index
title: Spring Cloud 学习指南
sidebar_label: 概览
sidebar_position: 1
---

# Spring Cloud 学习指南

> [!TIP] > **微服务架构基础**: Spring Cloud 是构建微服务架构的完整工具集。掌握服务注册发现、配置管理、API 网关是微服务开发的关键，建议从 [核心概念](./core-concepts) 开始学习。

## 📚 学习路径

### 基础部分

- **[核心概念](./core-concepts)** - 微服务架构、服务治理、Spring Cloud 组件体系
- **[Eureka 服务注册与发现](./eureka)** - 服务注册、服务发现、高可用配置
- **[Config 配置中心](./config)** - 集中配置管理、动态刷新、配置加密

### 核心组件

- **[Gateway API 网关](./gateway)** - 路由配置、过滤器、限流熔断
- **[Feign 声明式调用](./feign)** - HTTP 客户端、服务调用、负载均衡
- **[Ribbon 负载均衡](./ribbon)** - 负载均衡策略、自定义配置

### 高级特性

- **[Hystrix 熔断器](./hystrix)** - 服务降级、熔断、限流、隔离
- **[Sleuth 链路追踪](./sleuth)** - 分布式追踪、日志关联、性能分析

## 🎯 核心组件速览

### Eureka - 服务注册与发现

服务注册中心，实现服务的自动注册和发现，支持高可用集群。

### Config - 配置中心

集中管理微服务配置，支持配置动态刷新、多环境配置、配置加密。

### Gateway - API 网关

统一的 API 入口，提供路由、过滤、限流、认证等功能。

### Feign - 声明式 HTTP 客户端

通过声明式接口实现 HTTP 调用，集成负载均衡和熔断器。

### Ribbon - 客户端负载均衡

提供多种负载均衡策略，支持自定义规则。

### Hystrix - 熔断器

实现服务降级、熔断、限流等容错机制，保护系统稳定性。

### Sleuth - 链路追踪

分布式系统的请求追踪，实现调用链路可视化。

## 🔧 常用注解速览

| 注解                    | 说明               |
| ----------------------- | ------------------ |
| `@EnableEurekaServer`   | 启用 Eureka 服务端 |
| `@EnableEurekaClient`   | 启用 Eureka 客户端 |
| `@EnableConfigServer`   | 启用配置中心服务端 |
| `@RefreshScope`         | 支持配置动态刷新   |
| `@EnableFeignClients`   | 启用 Feign 客户端  |
| `@FeignClient`          | 声明 Feign 客户端  |
| `@LoadBalanced`         | 启用负载均衡       |
| `@HystrixCommand`       | 定义熔断方法       |
| `@EnableCircuitBreaker` | 启用熔断器         |

## 📊 Spring Cloud 组件对比

| 功能         | Netflix 组件 | 阿里巴巴组件 | 说明                     |
| ------------ | ------------ | ------------ | ------------------------ |
| 服务注册发现 | Eureka       | Nacos        | Nacos 功能更强大         |
| 配置中心     | Config       | Nacos        | Nacos 同时支持注册和配置 |
| 负载均衡     | Ribbon       | Dubbo        | 都支持多种策略           |
| 服务调用     | Feign        | Dubbo        | Dubbo 性能更高           |
| 熔断降级     | Hystrix      | Sentinel     | Sentinel 更灵活          |
| API 网关     | Zuul/Gateway | Gateway      | Gateway 是推荐方案       |

## 📖 学习资源

- [Spring Cloud 官方文档](https://spring.io/projects/spring-cloud)
- [Spring Cloud GitHub](https://github.com/spring-cloud)
- [Spring Cloud 中文文档](https://www.springcloud.cc/)

## 🚀 下一步

选择上面的任意主题开始学习，建议按照学习路径的顺序进行学习。如果你已经熟悉 Spring Boot，可以直接从 [Eureka](./eureka) 或 [Gateway](./gateway) 开始。

对于国内项目，也可以考虑学习 **Spring Cloud Alibaba**，它提供了更适合国内场景的解决方案。

---

**最后更新**: 2025 年 12 月  
**版本**: Spring Cloud 2023.x
