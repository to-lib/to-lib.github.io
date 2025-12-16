---
title: JDK / Spring 中的设计模式速查
sidebar_position: 6
description: 从 JDK 与 Spring 的真实实现出发，快速建立“模式 -> 场景 -> 代码位置”的映射
---

# JDK / Spring 中的设计模式速查

本文聚焦一个目标：把 23 种经典设计模式，和你在 **JDK / Spring** 里每天会遇到的“真实代码”对应起来，帮助你更快地“看懂框架源码 / 写出可维护代码”。

## 速查表：模式 -> 典型实现

| 模式 | 在 JDK / Spring 中的典型位置 | 你在业务中最常见的用法 |
| --- | --- | --- |
| Singleton | Spring `singleton` scope（默认）；JDK `Runtime.getRuntime()` | 全局唯一的配置、连接池、缓存管理器（谨慎使用） |
| Factory Method | Spring `BeanFactory` / `FactoryBean`；JDK `Calendar.getInstance()` | 把“创建逻辑”从业务代码中抽离，屏蔽复杂构建与选择 |
| Abstract Factory | Spring `BeanFactory` + 条件装配（按 profile/条件产生一组 bean） | 需要“成套替换”的组件族（如不同存储实现） |
| Builder | JDK `StringBuilder`；Spring `UriComponentsBuilder`（理念类似） | 参数很多、可选项多的对象构建（避免长参数列表） |
| Prototype | Spring `prototype` scope；`clone()`（谨慎） | 复制对象模板后再修改（注意深拷贝/不可变对象） |
| Adapter | Spring MVC `HandlerAdapter`；JDK `InputStreamReader` | “旧接口/第三方接口”适配到你系统的统一接口 |
| Proxy | Spring AOP（JDK 动态代理/CGLIB） | 认证、鉴权、日志、缓存、事务等横切能力 |
| Decorator | JDK `InputStream` 体系（`BufferedInputStream` 等） | 在不改原类的前提下叠加功能（但别嵌套太深） |
| Facade | Spring `ApplicationContext`（对复杂子系统的门面） | 给复杂子系统提供更简单的调用入口 |
| Template Method | Spring `JdbcTemplate`、`RestTemplate`（典型思想） | 固定流程 + 可变步骤（比如导入/导出、对账流程） |
| Strategy | Spring `@Autowired Map<String, X>` 按 key 选择实现 | 消除大量 if-else（支付、路由、规则、限流算法） |
| Observer | Spring `ApplicationEventPublisher` / `ApplicationListener` | 领域事件、异步解耦通知、审计 |
| Chain of Responsibility | Servlet Filter Chain；Spring Security Filter Chain | 认证/鉴权/限流/日志等按顺序处理 |

> 说明：表格只列最常用、最好“落地”的部分模式；其余模式在框架中也存在，但使用频率或识别成本更高。

## 你如何在项目里“正确使用”这些模式

## 1) Proxy（Spring AOP）最常见：事务、日志、权限

- **你想要的效果**：不污染业务代码，把横切逻辑集中管理
- **典型场景**：
  - `@Transactional`
  - 方法耗时统计
  - 权限校验

**判断是否适合 Proxy：**

- 这段逻辑是否横跨多个业务类？
- 是否可以做到“对业务透明”？
- 是否需要在调用前/后做增强？

## 2) Strategy + Factory：用配置/注册表消灭 if-else

当你遇到类似：

- “根据类型选择不同实现”
- “规则经常变”

优先考虑：

- 用 Strategy 表达“算法/规则族”
- 用 Factory 或注册表管理“如何拿到策略”

## 3) Template Method：业务流程固定，但步骤可变

适用于：

- 数据导入导出
- 订单处理流程
- 风控/审核流程

不要滥用：

- 如果流程本身经常变，Template Method 可能把变化锁死；此时更适合用 Strategy 或流程编排（如状态机）。

## 4) Adapter：把“外部世界”隔离在边界

- 第三方 SDK
- 历史遗留模块
- 多版本接口

把不稳定、不可控的接口用 Adapter 统一起来，你的业务层就会更稳定。

## 常见误区（建议你重点避开）

- **把工具类都做成 Singleton**：会导致隐藏依赖、难测试。
- **Decorator 无限套娃**：链路过深后调试困难，性能也更差。
- **为了“看起来高级”强行上模式**：优先写清晰的业务代码，必要时再重构引入。

## 相关文档

- [设计模式概述](./overview)
- [快速参考](./quick-reference)
- [最佳实践](./best-practices)
- [模式选择指南](./selection-guide)
