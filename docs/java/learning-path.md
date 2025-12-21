---
sidebar_position: 0
title: Java 学习路线
---

# 🧭 Java 学习路线（从零到高级后端）

这条路线适合想成为 Java 后端工程师的开发者，涵盖从基础语法到分布式系统的完整知识体系。每个阶段都有明确的目标、时间规划和实践建议。

> [!TIP]
> 本学习路线基于 **JDK 1.8 (Java 8)** 版本，这是目前企业中最广泛使用的 Java 版本。建议按顺序学习，扎实掌握每个阶段后再进入下一阶段。

---

## 阶段 1：Java 基础语法（1–2 周）

**🎯 目标**：能写基础程序，理解语言核心机制

### 📌 必学内容

- **基础语法**：变量、数据类型、运算符、流程控制（if/switch/for/while）
- **面向对象**：类、对象、封装、继承、多态、接口、抽象类
- **常用 API**：String、StringBuilder、Math、Arrays
- **异常机制**：try/catch/finally、throw/throws、自定义异常
- **I/O 基础**：File、InputStream、OutputStream、Reader、Writer

### 📚 推荐文档

| 主题                                      | 描述                               |
| ----------------------------------------- | ---------------------------------- |
| [基础语法](/docs/java/basic-syntax)       | 数据类型、变量、运算符、流程控制   |
| [面向对象](/docs/java/oop)                | 类、对象、继承、多态、内部类、枚举 |
| [异常处理](/docs/java/exception-handling) | 异常分类、处理机制、最佳实践       |
| [IO 流](/docs/java/io-streams)            | 字节流、字符流、缓冲流基础         |
| [常用类](/docs/java/common-classes)       | String、Math、包装类等常用类       |

### 🛠 实践项目

- 写一个**学生管理系统**（控制台版）：增删改查、文件存储
- 写一个**个人记账本**：收支记录、统计分析、数据持久化
- 熟练使用 **IntelliJ IDEA**：快捷键、调试、重构

### ✅ 阶段目标检验

- [ ] 能独立编写 200+ 行的控制台程序
- [ ] 理解面向对象三大特性
- [ ] 能正确处理异常并写入日志文件

---

## 阶段 2：Java 核心进阶（2–4 周）

**🎯 目标**：掌握 Java 运行机制，能写高质量代码

### 📌 必学内容

- **JVM 基础**：类加载机制、内存结构（堆/栈/方法区）、GC 原理
- **泛型编程**：泛型类、泛型方法、通配符、类型擦除
- **注解与反射**：自定义注解、反射 API、动态代理
- **集合框架深入**：ArrayList/LinkedList 原理、HashMap/TreeMap 实现、Collections 工具类
- **Lambda 与 Stream**：函数式接口、Lambda 表达式、Stream API
- **单元测试**：JUnit 5、断言、Mock 测试

### 📚 推荐文档

| 主题                                            | 描述                                     |
| ----------------------------------------------- | ---------------------------------------- |
| [JVM 基础](/docs/java/jvm-basics)               | JVM 架构、类加载、内存模型、垃圾回收     |
| [泛型编程](/docs/java/generics)                 | 泛型类、泛型方法、通配符、类型擦除       |
| [反射与注解](/docs/java/reflection-annotation)  | 反射 API、动态代理、自定义注解           |
| [集合框架](/docs/java/collections)              | List、Set、Map 详解和性能对比            |
| [函数式编程](/docs/java/functional-programming) | Lambda、Stream API、函数式接口           |
| [JDK 8 新特性](/docs/java/jdk8-features)        | Lambda、Stream、Optional、新日期时间 API |

### 🛠 实践项目

- 用 **Stream API** 重构阶段 1 的项目
- 写一个简单的 **JSON 解析器**（手写递归下降解析）
- 写一个 **文件扫描工具**（使用反射和注解）
- 实现一个 **简易 IoC 容器**（理解 Spring 原理）

### ✅ 阶段目标检验

- [ ] 能画出 JVM 内存结构图
- [ ] 能用 Stream 写复杂的数据处理逻辑
- [ ] 能用反射实现简单的框架功能

---

## 阶段 3：Java 并发编程（4–6 周）

**🎯 目标**：理解并发底层原理，能写高性能线程代码

> [!IMPORTANT]
> 并发编程是 Java 后端的核心技能，需要投入足够的时间深入理解。这部分内容在面试中出现频率极高。

### 📌 必学内容

- **线程模型**：Thread、Runnable、Callable、Future
- **线程池**：ThreadPoolExecutor 原理、核心参数、拒绝策略
- **JUC 核心**：
  - Lock 接口与 ReentrantLock
  - AQS（AbstractQueuedSynchronizer）原理
  - CAS 原理与原子类（AtomicInteger、AtomicReference）
  - Condition 条件变量
- **并发容器**：ConcurrentHashMap、CopyOnWriteArrayList、BlockingQueue
- **内存模型**：
  - JMM（Java Memory Model）
  - volatile 语义与实现
  - happens-before 规则
- **锁机制深入**：
  - synchronized 底层实现
  - 对象头与 Mark Word
  - 锁升级（偏向锁 → 轻量级锁 → 重量级锁）

### 📚 推荐文档

| 主题                                | 描述                                     |
| ----------------------------------- | ---------------------------------------- |
| [多线程](/docs/java/multithreading) | 线程创建、同步、通信、线程池、并发工具类 |
| [JVM 基础](/docs/java/jvm-basics)   | 内存模型、对象头、锁机制                 |
| [性能优化](/docs/java/performance)  | 并发优化、锁优化策略                     |

### 🛠 实践项目

- 实现一个 **简易线程池**（理解 ThreadPoolExecutor 原理）
- 用 **Disruptor 思路** 写高性能无锁队列
- 实现 **读写锁**（参考 ReentrantReadWriteLock）
- 分析 **ConcurrentHashMap** 源码（JDK 7 vs JDK 8）

### ✅ 阶段目标检验

- [ ] 能讲清 synchronized 和 ReentrantLock 的区别
- [ ] 能画出 AQS 的核心数据结构
- [ ] 能分析并解决死锁问题

---

## 阶段 4：数据库与持久化（2–4 周）

**🎯 目标**：能写高性能 SQL，理解事务与索引原理

### 📌 必学内容

- **MySQL 核心**：
  - 索引原理（B+ 树、聚簇索引、二级索引）
  - 事务 ACID 与隔离级别
  - 锁机制（行锁、表锁、间隙锁）
  - 执行计划分析（EXPLAIN）
  - 慢查询优化
- **JDBC 基础**：Connection、Statement、ResultSet、连接池
- **ORM 框架**：
  - MyBatis：Mapper、动态 SQL、缓存机制
  - JPA/Hibernate：实体映射、关联关系
- **连接池**：HikariCP 原理与配置

### 📚 推荐文档

| 主题                                                   | 描述                 |
| ------------------------------------------------------ | -------------------- |
| [MySQL 基础](/docs/mysql/basic-concepts)               | 数据库基本概念       |
| [MySQL 索引](/docs/mysql/indexes)                      | 索引原理与优化       |
| [MySQL 事务](/docs/mysql/transactions)                 | 事务 ACID、隔离级别  |
| [MySQL 锁](/docs/mysql/locks)                          | 行锁、表锁、死锁处理 |
| [MySQL 性能优化](/docs/mysql/performance-optimization) | 慢查询分析、索引优化 |

### 🛠 实践项目

- 写一个 **小型 CRUD 服务**（Spring Boot + MyBatis）
- 分析慢 SQL 并 **优化索引**
- 实现 **分页查询优化**（深分页问题）
- 模拟 **高并发下的库存扣减**（解决超卖问题）

### ✅ 阶段目标检验

- [ ] 能讲清 B+ 树索引的原理
- [ ] 能分析 EXPLAIN 执行计划
- [ ] 能正确处理事务边界

---

## 阶段 5：Spring 全家桶（4–8 周）

**🎯 目标**：掌握 Spring 生态，能写企业级后端服务

> [!TIP]
> Spring 是 Java 后端的核心框架，建议先理解原理再大量实践。

### 📌 必学内容

#### Spring Framework

- **IoC / DI 原理**：BeanFactory、ApplicationContext、Bean 生命周期
- **AOP 原理**：动态代理（JDK / CGLib）、切面、通知类型
- **事务管理**：声明式事务、事务传播机制、事务失效场景

#### Spring Boot

- **自动装配原理**：@EnableAutoConfiguration、spring.factories、条件注解
- **Starter 机制**：自定义 Starter 开发
- **配置体系**：application.yml、Profile、配置优先级

#### Spring MVC

- **DispatcherServlet 流程**：请求处理全流程
- **参数解析**：@RequestParam、@PathVariable、@RequestBody
- **拦截器与过滤器**：区别与使用场景

#### Spring Cloud（可选进阶）

- 注册中心：Nacos / Eureka
- 服务调用：Feign
- 网关：Gateway
- 限流熔断：Sentinel
- 配置中心：Config / Nacos

### 📚 推荐文档

| 主题                                                        | 描述                   |
| ----------------------------------------------------------- | ---------------------- |
| [Spring 核心概念](/docs/spring/core-concepts)               | IoC、DI、Bean 管理     |
| [Spring AOP](/docs/spring/aop)                              | 面向切面编程原理与实践 |
| [Spring 事务](/docs/spring/transactions)                    | 事务管理、传播机制     |
| [Spring MVC](/docs/spring/spring-mvc)                       | Web 开发、请求处理     |
| [Spring Boot 入门](/docs/springboot/quick-start)            | 快速开始 Spring Boot   |
| [Spring Boot 自动装配](/docs/springboot/auto-configuration) | 自动装配原理           |
| [Spring Cloud 入门](/docs/springcloud)                      | 微服务架构入门         |

### 🛠 实践项目

- 写一个 **完整的 RESTful 服务**（用户管理、权限控制）
- 自己实现一个 **Spring Boot Starter**（如日志 Starter）
- 用 **AOP 实现** 统一日志 / 监控框架
- 实现 **自定义注解** 实现接口限流

### ✅ 阶段目标检验

- [ ] 能画出 Spring Bean 生命周期流程图
- [ ] 能讲清动态代理的两种实现方式
- [ ] 能独立完成一个企业级后端服务

---

## 阶段 6：Netty 与网络编程（4–6 周）

**🎯 目标**：掌握高性能网络编程，能写自定义协议服务

### 📌 必学内容

- **IO 模型**：BIO / NIO / AIO 对比
- **NIO 核心**：Buffer、Channel、Selector
- **Reactor 模型**：单线程 / 多线程 / 主从 Reactor
- **Netty 核心组件**：
  - EventLoop、EventLoopGroup
  - Channel、ChannelPipeline
  - ChannelHandler、ChannelHandlerContext
- **编解码**：ByteBuf、Codec、MessageToByteEncoder
- **粘包拆包**：DelimiterBasedFrameDecoder、LengthFieldBasedFrameDecoder
- **心跳机制**：IdleStateHandler

### 📚 推荐文档

| 主题                                          | 描述                         |
| --------------------------------------------- | ---------------------------- |
| [Netty 概述](/docs/netty/overview)            | Netty 简介与核心概念         |
| [Netty 核心组件](/docs/netty/core-components) | EventLoop、Channel、Pipeline |
| [Netty ByteBuf](/docs/netty/bytebuf)          | 缓冲区管理                   |
| [Netty 编解码](/docs/netty/codec)             | 编解码器原理与实践           |
| [Netty 高级特性](/docs/netty/advanced)        | 高级用法与优化               |
| [Netty 实战](/docs/netty/practical-examples)  | 实战案例                     |

### 🛠 实践项目

- 写一个 **IM 即时通讯系统**（群聊、私聊、在线状态）
- 写一个 **RPC 框架**（服务注册、负载均衡、序列化）
- 实现 **自定义协议**（类似 Redis 的 RESP 协议）
- 写一个 **HTTP 服务器**（简化版 Tomcat）

### ✅ 阶段目标检验

- [ ] 能画出 Netty 的线程模型
- [ ] 能正确处理粘包拆包问题
- [ ] 能实现自定义的应用层协议

---

## 阶段 7：分布式系统（6–12 周）

**🎯 目标**：掌握分布式系统核心原理，能设计高可用架构

> [!IMPORTANT]
> 分布式系统是高级后端工程师的必备技能，需要理论与实践相结合。

### 📌 必学内容

- **分布式理论**：CAP 定理、BASE 理论、一致性模型
- **RPC 原理**：序列化、网络传输、服务发现、负载均衡
- **注册中心**：Nacos / Zookeeper / Consul
- **配置中心**：Nacos Config / Apollo
- **分布式锁**：Redis 分布式锁、Zookeeper 分布式锁、Redisson
- **分布式事务**：2PC、3PC、TCC、SAGA、Seata
- **消息队列**：
  - Kafka：高吞吐、分区、消费者组
  - RocketMQ：事务消息、延迟消息、顺序消息
  - RabbitMQ：Exchange 类型、死信队列
- **缓存**：
  - Redis 数据结构与应用场景
  - 缓存穿透、击穿、雪崩
  - 缓存一致性
- **限流熔断降级**：Sentinel、Hystrix、Resilience4j

### 📚 推荐文档

| 主题                                              | 描述                 |
| ------------------------------------------------- | -------------------- |
| [Redis 入门](/docs/redis/introduction)            | Redis 基础与数据结构 |
| [Redis 缓存策略](/docs/redis/cache-strategies)    | 缓存设计与问题处理   |
| [Redis 分布式锁](/docs/redis/practical-examples)  | 分布式锁实现         |
| [Kafka 入门](/docs/kafka)                         | Kafka 核心概念       |
| [RocketMQ 入门](/docs/rocketmq)                   | RocketMQ 核心概念    |
| [RabbitMQ 入门](/docs/rabbitmq)                   | RabbitMQ 核心概念    |
| [微服务架构](/docs/microservices)                 | 微服务设计与实践     |
| [Spring Cloud](/docs/springcloud)                 | Spring Cloud 组件    |
| [Spring Cloud Alibaba](/docs/springcloud-alibaba) | Nacos、Sentinel 等   |

### 🛠 实践项目

- 实现 **分布式 ID 生成器**（Snowflake 算法）
- 实现 **Redis 分布式锁**（支持续期、可重入）
- 用 **Kafka 实现异步削峰**（订单系统）
- 实现 **分布式事务**（TCC 模式）
- 设计一个 **秒杀系统**（高并发、防超卖）

### ✅ 阶段目标检验

- [ ] 能讲清 CAP 定理的含义
- [ ] 能设计一个高可用的分布式系统
- [ ] 能正确使用消息队列解决实际问题

---

## 阶段 8：工程化与 DevOps（持续提升）

**🎯 目标**：掌握工程化能力，能独立完成项目全流程

### 📌 必学内容

- **版本控制**：Git 高级用法、分支策略、GitHub Actions
- **构建工具**：Maven 生命周期、Gradle 配置
- **容器化**：
  - Docker：镜像构建、容器编排
  - Docker Compose：多容器管理
- **容器编排**：
  - Kubernetes：Pod、Deployment、Service
  - Helm：Chart 管理
- **日志体系**：
  - ELK（Elasticsearch + Logstash + Kibana）
  - Loki + Grafana
- **监控告警**：
  - Prometheus + Grafana
  - SkyWalking / Zipkin（链路追踪）

### 📚 推荐文档

| 主题                                | 描述                 |
| ----------------------------------- | -------------------- |
| [构建工具](/docs/java/build-tools)  | Maven/Gradle 配置    |
| [Docker 入门](/docs/docker)         | Docker 基础与实践    |
| [Kubernetes 入门](/docs/kubernetes) | K8s 核心概念与实践   |
| [Linux 基础](/docs/linux)           | Linux 常用命令与配置 |
| [Nginx 入门](/docs/nginx)           | Nginx 配置与优化     |

### 🛠 实践项目

- 给项目配置 **CI/CD 流水线**（GitHub Actions / Jenkins）
- 用 **Docker** 部署 Spring Boot 服务
- 用 **K8s** 部署微服务集群
- 搭建 **完整的监控体系**（Prometheus + Grafana）

### ✅ 阶段目标检验

- [ ] 能独立完成项目的 Docker 化
- [ ] 能配置 CI/CD 自动化流水线
- [ ] 能搭建基本的监控告警系统

---

## 阶段 9：高频面试方向（随时补充）

**🎯 目标**：系统准备面试，查漏补缺

### 📌 核心面试题方向

| 方向          | 重点内容                                   |
| ------------- | ------------------------------------------ |
| **Java 并发** | 线程池、锁机制、JMM、volatile、CAS、AQS    |
| **JVM**       | 内存模型、GC 算法、类加载、调优            |
| **MySQL**     | 索引原理、事务隔离级别、锁、SQL 优化       |
| **Redis**     | 数据结构、持久化、集群、分布式锁、缓存问题 |
| **Spring**    | IoC/AOP 原理、Bean 生命周期、事务失效场景  |
| **分布式**    | CAP、分布式锁、分布式事务、消息队列        |
| **设计模式**  | 单例、工厂、代理、策略、模板方法、观察者   |
| **系统设计**  | 秒杀系统、短链系统、feed 流、IM 系统       |

### 📚 推荐文档

| 主题                                        | 描述                |
| ------------------------------------------- | ------------------- |
| [Java 面试题](/docs/interview)              | 高频面试题汇总      |
| [Java 设计模式](/docs/java-design-patterns) | 23 种设计模式详解   |
| [最佳实践](/docs/java/best-practices)       | Java 编码规范与实践 |
| [常见问题](/docs/java/faq)                  | 常见问题解答        |

---

## 📚 总结：最精简的 Java 学习路径图

```text
┌────────────────────────────────────────────────────────────────┐
│                    Java 后端工程师学习路径                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  阶段 1    阶段 2    阶段 3    阶段 4    阶段 5                  │
│    │        │        │        │        │                       │
│    ▼        ▼        ▼        ▼        ▼                       │
│  基础语法 → 核心进阶 → 并发编程 → 数据库 → Spring                 │
│  (1-2周)   (2-4周)   (4-6周)  (2-4周)  (4-8周)                  │
│                                                                │
│                         │                                      │
│                         ▼                                      │
│              ┌─────────────────────┐                           │
│              │      阶段 6         │                           │
│              │   Netty 网络编程    │                           │
│              │     (4-6周)         │                           │
│              └─────────────────────┘                           │
│                         │                                      │
│                         ▼                                      │
│              ┌─────────────────────┐                           │
│              │      阶段 7         │                           │
│              │    分布式系统       │                           │
│              │     (6-12周)        │                           │
│              └─────────────────────┘                           │
│                         │                                      │
│                         ▼                                      │
│              ┌─────────────────────┐                           │
│              │      阶段 8         │                           │
│              │   工程化 DevOps     │                           │
│              │     (持续提升)       │                           │
│              └─────────────────────┘                           │
│                         │                                      │
│                         ▼                                      │
│              ┌─────────────────────┐                           │
│              │      阶段 9         │                           │
│              │    面试准备         │                           │
│              │     (随时补充)       │                           │
│              └─────────────────────┘                           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 🎯 学习建议

> [!TIP] > **高效学习的关键**
>
> 1. **动手实践** > 单纯看书/视频
> 2. **理解原理** > 死记硬背
> 3. **写项目** > 只刷算法题
> 4. **看源码** > 只用框架

### 时间规划建议

| 目标           | 预计时间   | 重点                           |
| -------------- | ---------- | ------------------------------ |
| **初级开发者** | 2-3 个月   | 阶段 1-2 + Spring Boot 基础    |
| **中级开发者** | 6-8 个月   | 阶段 1-5 + 基础分布式          |
| **高级开发者** | 12-18 个月 | 全部阶段 + 源码阅读 + 系统设计 |

### 推荐学习资源

- **官方文档**：[Java Documentation](https://docs.oracle.com/javase/8/docs/)
- **书籍**：《Java 核心技术》《深入理解 Java 虚拟机》《Java 并发编程实战》
- **实践**：GitHub 开源项目、LeetCode 算法练习

---

## 🔗 相关资源

- [Java 设计模式](/docs/java-design-patterns) - 23 种设计模式详解
- [Spring Framework](/docs/spring) - Spring 核心框架
- [Spring Boot](/docs/springboot) - Spring Boot 全教程
- [Spring Cloud](/docs/springcloud) - 微服务架构
- [Netty](/docs/netty) - 高性能网络框架
- [MySQL](/docs/mysql) - MySQL 数据库
- [Redis](/docs/redis) - Redis 缓存
- [Kafka](/docs/kafka) - 消息队列

> 祝你学习顺利，早日成为优秀的 Java 后端工程师！🚀
