---
sidebar_position: 25
---

# 设计模式快速参考

## 23种设计模式完整列表

### 创建型模式 (5种)

创建对象的各种方式，隐藏创建逻辑，使代码更灵活。

| 模式 | 目的 | 关键特性 | 适用场景 |
|------|------|--------|--------|
| **Singleton**<br/>单例模式 | 确保类仅有一个实例 | 全局访问点 | 数据库连接、日志、配置 |
| **Factory Method**<br/>工厂方法 | 创建对象的接口 | 子类决定产品 | 多种产品，需要切换 |
| **Abstract Factory**<br/>抽象工厂 | 创建产品族 | 一族相关对象 | 产品族相关，需一致性 |
| **Builder**<br/>建造者 | 分步骤构建复杂对象 | 链式调用 | 复杂对象，可选参数多 |
| **Prototype**<br/>原型 | 通过克隆创建对象 | 对象复制 | 大量相似对象，复制代价低 |

### 结构型模式 (7种)

处理对象和类的组合，形成更复杂的结构。

| 模式 | 目的 | 关键特性 | 适用场景 |
|------|------|--------|--------|
| **Adapter**<br/>适配器 | 转换接口实现兼容 | 接口转换 | 第三方库集成，接口不兼容 |
| **Bridge**<br/>桥接 | 分离抽象和实现 | 抽象-实现分离 | 多维度变化 |
| **Composite**<br/>组合 | 树形结构中处理 | 统一接口 | 文件系统、菜单、UI组件 |
| **Decorator**<br/>装饰器 | 动态添加功能 | 功能组合 | IO流、UI增强 |
| **Facade**<br/>外观 | 简化复杂子系统 | 统一接口 | 封装复杂系统、提供简单API |
| **Flyweight**<br/>享元 | 共享细粒度对象 | 对象复用 | 大量相似对象，内存优化 |
| **Proxy**<br/>代理 | 控制对象访问 | 替身代理 | 延迟加载、访问控制、日志 |

### 行为型模式 (11种)

处理对象间的通信、职责分配和算法的分离。

| 模式 | 目的 | 关键特性 | 适用场景 |
|------|------|--------|--------|
| **Chain of Responsibility**<br/>责任链 | 请求沿链传递 | 处理者链 | 多级审批、日志系统 |
| **Command**<br/>命令 | 将请求对象化 | 请求-对象化 | 撤销/重做、队列、事务 |
| **Iterator**<br/>迭代器 | 遍历集合元素 | 统一遍历 | 集合框架、各种遍历方式 |
| **Mediator**<br/>中介者 | 集中管理对象通信 | 通信中介 | 复杂对象交互、协调 |
| **Memento**<br/>备忘录 | 保存/恢复状态 | 状态快照 | 撤销/重做、游戏存档 |
| **Observer**<br/>观察者 | 一对多通知 | 事件发布-订阅 | 事件系统、MVC |
| **State**<br/>状态 | 状态改变行为 | 状态转换 | 订单流程、游戏状态 |
| **Strategy**<br/>策略 | 可互换的算法 | 算法家族 | 排序、支付方式、算法选择 |
| **Template Method**<br/>模板方法 | 定义算法骨架 | 步骤模板 | 数据处理流程、导出方式 |
| **Visitor**<br/>访问者 | 为对象添加操作 | 双重分派 | 编译器、代码生成 |
| **Interpreter**<br/>解释器 | 解析和执行语言 | 语法解析 | SQL、表达式求值、配置文件 |

## 选择设计模式的决策树

### 需要创建对象？

```
创建对象
├─ 单一对象、全局访问 → Singleton
├─ 多种类型产品
│  ├─ 产品族 → Abstract Factory
│  └─ 单一产品 → Factory Method
├─ 复杂对象
│  ├─ 分步构建 → Builder
│  └─ 克隆创建 → Prototype
```

### 需要处理对象结构？

```
处理对象结构
├─ 树形结构 → Composite
├─ 动态添加功能 → Decorator
├─ 简化复杂系统 → Facade
├─ 共享对象 → Flyweight
├─ 控制访问 → Proxy
├─ 接口转换 → Adapter
└─ 分离抽象实现 → Bridge
```

### 需要处理对象交互？

```
对象交互
├─ 一对多通知 → Observer
├─ 多对多通信 → Mediator
├─ 链式传递 → Chain of Responsibility
├─ 状态转换 → State
├─ 可换算法 → Strategy
├─ 处理请求 → Command
├─ 遍历集合 → Iterator
├─ 保存状态 → Memento
├─ 添加操作 → Visitor
├─ 定义流程 → Template Method
└─ 解析语言 → Interpreter
```

## 按复杂度分类

### 简单 ⭐

适合初学者，容易理解实现

- Singleton（单例）
- Factory Method（工厂方法）
- Adapter（适配器）
- Strategy（策略）
- Observer（观察者）

### 中等 ⭐⭐

需要一定经验，要理解关键概念

- Abstract Factory（抽象工厂）
- Builder（建造者）
- Decorator（装饰器）
- Proxy（代理）
- State（状态）
- Command（命令）
- Template Method（模板方法）

### 复杂 ⭐⭐⭐

需要深入理解，应用场景特殊

- Bridge（桥接）
- Composite（组合）
- Facade（外观）
- Flyweight（享元）
- Chain of Responsibility（责任链）
- Mediator（中介者）
- Memento（备忘录）
- Iterator（迭代器）
- Visitor（访问者）
- Interpreter（解释器）
- Prototype（原型）

## 按使用频率排序

### 高频 (项目中经常用到)

1. Singleton（单例）- 无处不在
2. Factory Method（工厂方法）- 对象创建
3. Strategy（策略）- 算法选择
4. Observer（观察者）- 事件系统
5. Decorator（装饰器）- 功能增强
6. Adapter（适配器）- 接口适配
7. Builder（建造者）- 复杂对象
8. Proxy（代理）- 访问控制

### 中频 (根据需要)

- Abstract Factory（产品族）
- Template Method（流程定义）
- Command（请求对象化）
- State（状态转换）
- Iterator（集合遍历）

### 低频 (特定场景)

- Prototype（对象克隆）
- Facade（简化接口）
- Composite（树形结构）
- Flyweight（对象共享）
- Chain of Responsibility（责任链）
- Mediator（对象协调）
- Memento（状态保存）
- Visitor（对象操作）
- Interpreter（语言解析）
- Bridge（抽象实现分离）

## 模式间的关系

### 经常一起使用

- **Factory + Singleton** - 工厂创建单例
- **Strategy + Factory** - 工厂创建策略对象
- **Observer + Mediator** - 对象协调和通知
- **Decorator + Factory** - 工厂创建装饰器
- **Builder + Factory** - 构建复杂对象
- **Template Method + Strategy** - 流程和算法
- **Composite + Iterator** - 遍历树形结构
- **Command + Memento** - 撤销和重做
- **State + Strategy** - 状态和行为

### 替代关系

- **Strategy vs State** - 策略由客户端选择，状态自动转换
- **Factory Method vs Abstract Factory** - 单产品vs产品族
- **Decorator vs Proxy** - 装饰添加功能，代理控制访问
- **Adapter vs Facade** - 适配器转换接口，外观简化使用
- **Iterator vs Visitor** - 迭代遍历vs对象操作

## 最常见的使用场景

### Web框架

- Spring IoC: Factory + Singleton
- Spring AOP: Proxy + Decorator
- MVC: Observer + Template Method

### 数据库

- Connection Pool: Flyweight + Singleton
- ORM: Visitor + Iterator

### 游戏开发

- GameObject: Composite + Visitor
- Animation: State + Strategy
- Audio Manager: Singleton + Facade

### 编译器/IDE

- AST: Visitor + Interpreter + Composite
- Lexer: Iterator + Visitor

## 学习建议

### 第一阶段：理解基础

1. Singleton - 最简单的模式
2. Factory Method - 基础创建模式
3. Strategy - 基础行为模式
4. Observer - 基础交互模式

### 第二阶段：掌握常用

5. Builder - 复杂对象创建
6. Decorator - 功能增强
7. Adapter - 接口适配
8. Template Method - 流程定义
9. Proxy - 访问控制
10. State - 状态管理

### 第三阶段：深入理解

11. Abstract Factory - 产品族
12. Composite - 树形结构
13. Chain of Responsibility - 请求链
14. Command - 请求对象化
15. Visitor - 对象操作
16. Iterator - 集合遍历
17. 其他模式...

## 反模式警告

### 过度使用设计模式

- 问题：为简单问题使用复杂模式
- 症状：代码反而变复杂
- 解决：根据实际需求选择

### 滥用继承

- 问题：优先使用继承而非组合
- 症状：类层次过深，难以维护
- 解决：优先选择组合

### 不一致的应用

- 问题：同一模式应用不一致
- 症状：代码风格混乱
- 解决：制定团队规范

### 忽视性能

- 问题：为了"优雅"忽视性能
- 症状：系统响应缓慢
- 解决：在模式和性能间平衡

## 快速查询表

| 我需要... | 使用这个模式 |
|---------|-----------|
| 保证类只有一个实例 | Singleton |
| 创建对象但隐藏创建逻辑 | Factory Method |
| 创建相关的对象族 | Abstract Factory |
| 构建复杂对象 | Builder |
| 复制对象 | Prototype |
| 转换接口 | Adapter |
| 分离抽象和实现 | Bridge |
| 树形对象结构 | Composite |
| 动态添加功能 | Decorator |
| 简化复杂系统 | Facade |
| 共享对象节省内存 | Flyweight |
| 控制对象访问 | Proxy |
| 请求沿链传递 | Chain of Responsibility |
| 请求对象化 | Command |
| 遍历集合 | Iterator |
| 集中管理对象通信 | Mediator |
| 保存/恢复状态 | Memento |
| 一对多通知 | Observer |
| 状态改变行为 | State |
| 可换算法 | Strategy |
| 定义算法骨架 | Template Method |
| 为对象添加操作 | Visitor |
| 解析表达式 | Interpreter |
