# Requirements Document

## Introduction

创建一份全面的Java高级开发工程师面试题文档，涵盖JVM深度、并发编程高级、分布式系统、性能调优、架构设计等高级主题。该文档将作为Java高级开发工程师面试准备的权威参考资料，与现有的基础面试题文档形成互补。

## Glossary

- **Java_Senior_Interview_Doc**: Java高级开发工程师面试题文档
- **JVM_Deep_Dive**: JVM深度知识，包括内存模型、GC调优、类加载机制等
- **Concurrency_Advanced**: 高级并发编程，包括JUC工具类、锁优化、无锁编程等
- **Performance_Tuning**: 性能调优，包括JVM调优、SQL优化、缓存策略等
- **Architecture_Design**: 架构设计，包括设计模式应用、微服务架构、分布式系统等
- **Source_Code_Analysis**: 源码分析，包括Spring、MyBatis等框架核心源码

## Requirements

### Requirement 1: JVM深度面试题

**User Story:** As a Java高级开发工程师候选人, I want 深入的JVM面试题, so that 我能展示对JVM底层原理的深刻理解。

#### Acceptance Criteria

1. THE Java_Senior_Interview_Doc SHALL 包含JVM内存模型详解题目（堆、栈、方法区、直接内存）
2. THE Java_Senior_Interview_Doc SHALL 包含垃圾回收器对比题目（CMS、G1、ZGC、Shenandoah）
3. THE Java_Senior_Interview_Doc SHALL 包含GC调优实战题目（参数配置、日志分析、问题排查）
4. THE Java_Senior_Interview_Doc SHALL 包含类加载机制题目（双亲委派、自定义类加载器、热部署）
5. THE Java_Senior_Interview_Doc SHALL 包含JIT编译优化题目（逃逸分析、内联优化、锁消除）

### Requirement 2: 高级并发编程面试题

**User Story:** As a Java高级开发工程师候选人, I want 高级并发编程面试题, so that 我能展示对多线程和并发的深入掌握。

#### Acceptance Criteria

1. THE Java_Senior_Interview_Doc SHALL 包含JUC工具类深度题目（AQS原理、ReentrantLock源码、Semaphore实现）
2. THE Java_Senior_Interview_Doc SHALL 包含线程池高级题目（核心参数调优、拒绝策略选择、监控方案）
3. THE Java_Senior_Interview_Doc SHALL 包含锁优化题目（偏向锁、轻量级锁、锁升级过程）
4. THE Java_Senior_Interview_Doc SHALL 包含无锁编程题目（CAS原理、Atomic类、LongAdder优化）
5. THE Java_Senior_Interview_Doc SHALL 包含并发设计模式题目（生产者消费者、读写锁、Future模式）

### Requirement 3: 性能调优面试题

**User Story:** As a Java高级开发工程师候选人, I want 性能调优面试题, so that 我能展示解决生产环境性能问题的能力。

#### Acceptance Criteria

1. THE Java_Senior_Interview_Doc SHALL 包含JVM调优实战题目（内存泄漏排查、CPU飙高分析、线程死锁诊断）
2. THE Java_Senior_Interview_Doc SHALL 包含性能监控工具题目（JProfiler、Arthas、JMC使用）
3. THE Java_Senior_Interview_Doc SHALL 包含代码级优化题目（集合选择、字符串处理、IO优化）
4. THE Java_Senior_Interview_Doc SHALL 包含数据库优化题目（索引优化、SQL调优、连接池配置）
5. THE Java_Senior_Interview_Doc SHALL 包含缓存策略题目（本地缓存、分布式缓存、缓存一致性）

### Requirement 4: 架构设计面试题

**User Story:** As a Java高级开发工程师候选人, I want 架构设计面试题, so that 我能展示系统设计和架构能力。

#### Acceptance Criteria

1. THE Java_Senior_Interview_Doc SHALL 包含设计模式高级应用题目（模式组合、框架中的模式、反模式）
2. THE Java_Senior_Interview_Doc SHALL 包含分布式系统题目（CAP理论、分布式事务、一致性算法）
3. THE Java_Senior_Interview_Doc SHALL 包含微服务架构题目（服务拆分、服务治理、链路追踪）
4. THE Java_Senior_Interview_Doc SHALL 包含高可用设计题目（限流熔断、降级策略、容灾方案）
5. THE Java_Senior_Interview_Doc SHALL 包含系统设计题目（秒杀系统、消息队列、分布式ID）

### Requirement 5: 框架源码分析面试题

**User Story:** As a Java高级开发工程师候选人, I want 框架源码分析面试题, so that 我能展示对主流框架底层原理的理解。

#### Acceptance Criteria

1. THE Java_Senior_Interview_Doc SHALL 包含Spring核心源码题目（IoC容器启动、Bean生命周期、AOP实现）
2. THE Java_Senior_Interview_Doc SHALL 包含Spring Boot源码题目（自动配置原理、启动流程、条件装配）
3. THE Java_Senior_Interview_Doc SHALL 包含MyBatis源码题目（SQL解析、动态代理、缓存机制）
4. THE Java_Senior_Interview_Doc SHALL 包含Netty源码题目（Reactor模型、ByteBuf设计、Pipeline机制）
5. THE Java_Senior_Interview_Doc SHALL 包含中间件源码题目（Redis客户端、Kafka生产者、RPC框架）

### Requirement 6: 文档格式与结构

**User Story:** As a 文档维护者, I want 统一的文档格式, so that 文档与现有Java面试题文档保持一致。

#### Acceptance Criteria

1. THE Java_Senior_Interview_Doc SHALL 使用与现有interview-questions.md相同的Markdown格式
2. THE Java_Senior_Interview_Doc SHALL 包含难度分级标识（高级、专家级）
3. THE Java_Senior_Interview_Doc SHALL 每道题目包含答案要点、代码示例、延伸阅读
4. THE Java_Senior_Interview_Doc SHALL 包含总结与学习建议章节
5. THE Java_Senior_Interview_Doc SHALL 在sidebars.ts中正确配置导航
