---
sidebar_position: 0
slug: /design-patterns-guide
---

# Java 设计模式完全指南

欢迎来到Java设计模式完全学习指南！本文档包含了23种经典设计模式的详细讲解、实际应用示例和最佳实践。

## 📚 文档组织结构

本指南分为以下几个部分：

### 📖 基础入门
- [设计模式概述](java-design-patterns/overview) - 理解什么是设计模式及其分类
- [快速参考](java-design-patterns/quick-reference) - 23种模式的快速查询表

### 🏭 创建型模式 (5种)
构造对象，隐藏创建逻辑

1. [单例模式 (Singleton)](java-design-patterns/singleton-pattern) - 确保类仅有一个实例
2. [工厂方法 (Factory Method)](java-design-patterns/factory-pattern) - 创建对象的接口
3. [抽象工厂 (Abstract Factory)](java-design-patterns/abstract-factory-pattern) - 创建产品族
4. [建造者模式 (Builder)](java-design-patterns/builder-pattern) - 分步构建复杂对象
5. [原型模式 (Prototype)](java-design-patterns/prototype-pattern) - 通过克隆创建对象

### 🔧 结构型模式 (7种)
组合对象，形成更大的结构

6. [适配器模式 (Adapter)](java-design-patterns/adapter-pattern) - 转换接口实现兼容
7. [装饰器模式 (Decorator)](java-design-patterns/decorator-pattern) - 动态添加功能
8. [外观模式 (Facade)](java-design-patterns/facade-pattern) - 简化复杂子系统
9. [代理模式 (Proxy)](java-design-patterns/proxy-pattern) - 控制对象访问
10. [组合模式 (Composite)](java-design-patterns/composite-pattern) - 树形结构处理
11. [享元模式 (Flyweight)](java-design-patterns/flyweight-pattern) - 共享细粒度对象
12. [桥接模式 (Bridge)](java-design-patterns/bridge-pattern) - 分离抽象和实现

### 🎯 行为型模式 (11种)
处理对象间的通信和职责分配

13. [观察者模式 (Observer)](java-design-patterns/observer-pattern) - 一对多通知
14. [策略模式 (Strategy)](java-design-patterns/strategy-pattern) - 可互换的算法
15. [状态模式 (State)](java-design-patterns/state-pattern) - 状态改变行为
16. [命令模式 (Command)](java-design-patterns/command-pattern) - 将请求对象化
17. [模板方法 (Template Method)](java-design-patterns/template-method-pattern) - 定义算法骨架
18. [责任链 (Chain of Responsibility)](java-design-patterns/chain-of-responsibility-pattern) - 请求沿链传递
19. [迭代器 (Iterator)](java-design-patterns/iterator-pattern) - 遍历集合元素
20. [中介者 (Mediator)](java-design-patterns/mediator-pattern) - 集中管理对象通信
21. [备忘录 (Memento)](java-design-patterns/memento-pattern) - 保存/恢复状态
22. [访问者 (Visitor)](java-design-patterns/visitor-pattern) - 为对象添加操作
23. [解释器 (Interpreter)](java-design-patterns/interpreter-pattern) - 解析和执行语言

### 💡 最佳实践
- [设计模式最佳实践](java-design-patterns/best-practices) - SOLID原则、反模式、应用清单

## 🎓 学习路径建议

### 🌱 初级 (1-2周)
如果你是设计模式新手，建议按以下顺序学习：

1. **理解基础概念**
   - [设计模式概述](java-design-patterns/overview)
   - [快速参考](java-design-patterns/quick-reference)

2. **学习最常用的5个模式**
   - [单例模式](java-design-patterns/singleton-pattern) - 最简单
   - [工厂方法](java-design-patterns/factory-pattern) - 对象创建
   - [策略模式](java-design-patterns/strategy-pattern) - 算法选择
   - [观察者模式](java-design-patterns/observer-pattern) - 事件系统
   - [装饰器模式](java-design-patterns/decorator-pattern) - 功能增强

### 🌿 中级 (2-4周)
掌握基础后，继续学习：

6. [建造者模式](java-design-patterns/builder-pattern)
7. [适配器模式](java-design-patterns/adapter-pattern)
8. [代理模式](java-design-patterns/proxy-pattern)
9. [模板方法](java-design-patterns/template-method-pattern)
10. [状态模式](java-design-patterns/state-pattern)
11. [命令模式](java-design-patterns/command-pattern)

### 🌳 高级 (4周+)
深入理解剩余的模式：

12. [抽象工厂](java-design-patterns/abstract-factory-pattern)
13. [原型模式](java-design-patterns/prototype-pattern)
14. [外观模式](java-design-patterns/facade-pattern)
15. [组合模式](java-design-patterns/composite-pattern)
16. [责任链](java-design-patterns/chain-of-responsibility-pattern)
17. [迭代器](java-design-patterns/iterator-pattern)
18. [中介者](java-design-patterns/mediator-pattern)
19. [备忘录](java-design-patterns/memento-pattern)
20. [访问者](java-design-patterns/visitor-pattern)
21. [解释器](java-design-patterns/interpreter-pattern)
22. [享元模式](java-design-patterns/flyweight-pattern)

最后阅读[最佳实践](java-design-patterns/best-practices)总结所学。

## 📊 设计模式分类速览

### 按使用频率
**高频** ⭐⭐⭐
- Singleton（无处不在）
- Factory Method（对象创建）
- Strategy（算法选择）
- Observer（事件系统）
- Decorator（功能增强）

**中频** ⭐⭐
- Abstract Factory
- Builder
- Template Method
- Command
- State

**低频** ⭐
- 其他11种模式（特定场景）

### 按复杂度
- **简单** - 容易理解：Singleton, Factory Method, Strategy, Adapter, Observer
- **中等** - 需要经验：Abstract Factory, Builder, Decorator, Proxy, State, Command, Template Method
- **复杂** - 需要深入理解：Bridge, Composite, Facade, Flyweight, Chain of Responsibility, Mediator, Memento, Iterator, Visitor, Interpreter, Prototype

## 🔍 快速查询

**我需要创建对象？**
- 单一对象 → [Singleton](java-design-patterns/singleton-pattern)
- 多种类型 → [Factory Method](java-design-patterns/factory-pattern)
- 产品族 → [Abstract Factory](java-design-patterns/abstract-factory-pattern)
- 复杂对象 → [Builder](java-design-patterns/builder-pattern)
- 克隆对象 → [Prototype](java-design-patterns/prototype-pattern)

**我需要组织对象结构？**
- 树形结构 → [Composite](java-design-patterns/composite-pattern)
- 动态功能 → [Decorator](java-design-patterns/decorator-pattern)
- 简化系统 → [Facade](java-design-patterns/facade-pattern)
- 隐藏实现 → [Proxy](java-design-patterns/proxy-pattern)
- 转换接口 → [Adapter](java-design-patterns/adapter-pattern)
- 分享对象 → [Flyweight](java-design-patterns/flyweight-pattern)

**我需要处理对象交互？**
- 一对多通知 → [Observer](java-design-patterns/observer-pattern)
- 可换算法 → [Strategy](java-design-patterns/strategy-pattern)
- 状态转换 → [State](java-design-patterns/state-pattern)
- 多对多通信 → [Mediator](java-design-patterns/mediator-pattern)
- 其他...

## 💻 每个文档包含

每个设计模式文档都包括：

- ✅ **模式定义** - 清楚的概念解释
- ✅ **问题分析** - 为什么需要这个模式
- ✅ **解决方案** - 模式的核心思想
- ✅ **代码实现** - 完整的Java实现示例
- ✅ **实际应用** - 多个真实场景例子
- ✅ **优缺点** - 权衡分析
- ✅ **适用场景** - 何时使用
- ✅ **最佳实践** - 应用建议
- ✅ **与其他模式的关系** - 模式间的联系

## 🚀 如何使用本指南

### 方式1：按顺序学习
从基础到高级，完整学习所有模式。

### 方式2：按需查询
使用[快速参考](java-design-patterns/quick-reference)快速找到所需模式。

### 方式3：按场景学习
根据实际需求查找相关模式（如"我要实现undo功能"）。

## 📝 学习建议

1. **理论 + 实践** - 不仅要读代码，要写代码
2. **对比学习** - 理解相似模式的区别
3. **项目应用** - 在真实项目中实践
4. **代码审查** - 学习他人代码中的模式应用
5. **定期复习** - 设计模式需要不断巩固

## 🎯 本指南的目标

- ✨ 帮助你理解23种经典设计模式
- 📚 提供清晰的讲解和完整的代码示例
- 💡 展示实际的应用场景
- 🔧 指导如何在项目中使用
- 📊 提供快速参考和决策树

## 🤝 贡献

如果你发现任何错误或想改进本指南，欢迎贡献！

---

**现在就开始学习吧！** 👉 [从概述开始](java-design-patterns/overview)
