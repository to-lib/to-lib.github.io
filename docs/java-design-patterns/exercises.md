---
sidebar_position: 8
title: 设计模式练习题
description: 23种设计模式练习题与实践项目
---

# 设计模式练习题

通过实践练习巩固设计模式知识。每道题都有明确的需求描述和参考提示。

## 创建型模式练习

### 练习 1: 单例模式 - 日志管理器

**需求描述**：
创建一个线程安全的日志管理器 `LogManager`，要求：

- 全局只有一个实例
- 支持不同级别的日志（INFO、WARN、ERROR）
- 日志输出到控制台，包含时间戳

**提示**：使用静态内部类或枚举实现

```java
// 期望用法
LogManager.getInstance().info("应用启动");
LogManager.getInstance().error("发生错误");
```

---

### 练习 2: 工厂方法模式 - 图形工厂

**需求描述**：
设计一个图形绘制系统，要求：

- 支持创建 Circle、Rectangle、Triangle 三种图形
- 每种图形都有 `draw()` 和 `getArea()` 方法
- 使用工厂方法模式，便于扩展新图形

**提示**：定义 Shape 接口和对应的 ShapeFactory 抽象类

```java
// 期望用法
ShapeFactory factory = new CircleFactory();
Shape circle = factory.createShape();
circle.draw();
System.out.println("面积: " + circle.getArea());
```

---

### 练习 3: 建造者模式 - 电脑配置器

**需求描述**：
创建一个电脑配置构建器 `ComputerBuilder`，要求：

- 可选配置：CPU、内存、硬盘、显卡、电源
- 支持链式调用
- 构建完成后返回 Computer 对象

**提示**：使用静态内部 Builder 类

```java
// 期望用法
Computer computer = new Computer.Builder()
    .cpu("Intel i7-13700K")
    .ram("32GB DDR5")
    .storage("1TB SSD")
    .gpu("RTX 4080")
    .build();
```

---

### 练习 4: 抽象工厂模式 - UI 主题工厂

**需求描述**：
设计一个跨平台 UI 组件库，要求：

- 支持 Light（浅色）和 Dark（深色）两套主题
- 每套主题包含 Button、TextField、Checkbox 组件
- 同一主题的组件风格统一

**提示**：抽象工厂创建产品族

```java
// 期望用法
UIFactory factory = new DarkThemeFactory();
Button btn = factory.createButton();
TextField tf = factory.createTextField();
```

---

### 练习 5: 原型模式 - 文档克隆

**需求描述**：
实现一个文档克隆系统，要求：

- 支持深拷贝文档对象（包含标题、内容、附件列表）
- 提供文档模板注册和获取功能
- 克隆后的文档修改不影响原文档

**提示**：实现 Cloneable 接口，注意深拷贝

```java
// 期望用法
Document template = DocumentRegistry.getTemplate("report");
Document myDoc = template.clone();
myDoc.setTitle("我的报告");
```

---

## 结构型模式练习

### 练习 6: 适配器模式 - 数据格式转换

**需求描述**：
系统需要使用第三方库读取 XML 数据，但你的系统只接受 JSON 格式，要求：

- 创建适配器将 XML 数据转换为 JSON 格式
- 不修改原有的 XML 解析器和 JSON 处理器代码

```java
// 期望用法
JsonDataProcessor processor = new XmlToJsonAdapter(xmlParser);
processor.processJson(data);
```

---

### 练习 7: 装饰器模式 - 咖啡订单

**需求描述**：
设计一个咖啡店订单系统，要求：

- 基础咖啡：Espresso（￥ 15）、Latte（￥ 20）
- 可选配料：牛奶（+￥ 3）、糖（+￥ 1）、摩卡（+￥ 5）
- 动态计算价格和描述

**提示**：装饰器动态添加功能

```java
// 期望用法
Beverage order = new Mocha(new Milk(new Espresso()));
System.out.println(order.getDescription()); // Espresso + 牛奶 + 摩卡
System.out.println(order.cost()); // 23.0
```

---

### 练习 8: 代理模式 - 图片延迟加载

**需求描述**：
实现一个图片查看器，要求：

- 图片只在首次显示时加载（延迟加载）
- 记录图片访问次数
- 提供访问控制（可选）

```java
// 期望用法
Image image = new ProxyImage("large_photo.jpg");
image.display(); // 首次调用时加载
image.display(); // 直接显示，不再加载
```

---

### 练习 9: 外观模式 - 家庭影院

**需求描述**：
设计一个家庭影院系统，包含多个子系统：

- 投影仪、音响、灯光、播放器
- 创建外观类提供简化接口：watchMovie()、endMovie()

```java
// 期望用法
HomeTheaterFacade theater = new HomeTheaterFacade();
theater.watchMovie("阿凡达"); // 一键开启所有设备
theater.endMovie(); // 一键关闭所有设备
```

---

### 练习 10: 组合模式 - 文件系统

**需求描述**：
模拟文件系统结构，要求：

- 支持文件和文件夹
- 文件夹可以包含文件和子文件夹
- 统一的 `getSize()` 方法计算大小

```java
// 期望用法
Folder root = new Folder("root");
root.add(new File("readme.txt", 100));
root.add(new Folder("src"));
System.out.println(root.getSize()); // 递归计算总大小
```

---

### 练习 11: 享元模式 - 棋子工厂

**需求描述**：
实现围棋棋子管理，要求：

- 棋子颜色（黑/白）作为内部状态共享
- 棋子位置作为外部状态
- 使用享元工厂管理棋子对象

```java
// 期望用法
ChessPiece black1 = ChessPieceFactory.getPiece("black");
ChessPiece black2 = ChessPieceFactory.getPiece("black");
System.out.println(black1 == black2); // true，同一对象
black1.display(1, 1);
black2.display(2, 3);
```

---

### 练习 12: 桥接模式 - 消息发送系统

**需求描述**：
设计消息发送系统，要求：

- 消息类型：普通消息、紧急消息
- 发送方式：Email、SMS、微信
- 消息类型和发送方式可自由组合

```java
// 期望用法
Message msg = new UrgentMessage(new EmailSender());
msg.send("服务器宕机了！");
```

---

## 行为型模式练习

### 练习 13: 观察者模式 - 气象站

**需求描述**：
实现气象监测系统，要求：

- 气象站发布天气数据（温度、湿度）
- 多个显示器订阅并显示数据
- 支持动态订阅和取消订阅

```java
// 期望用法
WeatherStation station = new WeatherStation();
station.addObserver(new TemperatureDisplay());
station.addObserver(new HumidityDisplay());
station.setMeasurements(25.0, 65.0); // 自动通知所有观察者
```

---

### 练习 14: 策略模式 - 导航系统

**需求描述**：
设计导航应用的路线规划，要求：

- 支持多种出行策略：步行、骑行、驾车、公交
- 每种策略有不同的时间和路线计算方式
- 运行时可切换策略

```java
// 期望用法
Navigator nav = new Navigator();
nav.setStrategy(new DrivingStrategy());
nav.navigate("北京", "上海");
nav.setStrategy(new PublicTransitStrategy());
nav.navigate("北京", "上海");
```

---

### 练习 15: 命令模式 - 遥控器

**需求描述**：
实现一个万能遥控器，要求：

- 支持控制多种电器（电视、空调、灯）
- 每个按钮可配置不同命令
- 支持撤销操作

```java
// 期望用法
RemoteControl remote = new RemoteControl();
remote.setCommand(0, new LightOnCommand(light), new LightOffCommand(light));
remote.pressButton(0); // 开灯
remote.undo(); // 关灯
```

---

### 练习 16: 状态模式 - 订单状态

**需求描述**：
实现电商订单状态管理，要求：

- 订单状态：待支付、已支付、已发货、已完成、已取消
- 不同状态下操作行为不同
- 状态自动流转

```java
// 期望用法
Order order = new Order();
order.pay(); // 待支付 -> 已支付
order.ship(); // 已支付 -> 已发货
order.complete(); // 已发货 -> 已完成
```

---

### 练习 17: 责任链模式 - 审批流程

**需求描述**：
实现请假审批系统，要求：

- 组长审批 1-3 天
- 经理审批 4-7 天
- 总监审批 8-15 天
- 超过 15 天需 CEO 审批

```java
// 期望用法
Approver chain = new TeamLeader()
    .setNext(new Manager())
    .setNext(new Director())
    .setNext(new CEO());
chain.approve(new LeaveRequest(5)); // 经理审批
```

---

### 练习 18: 模板方法模式 - 数据导出

**需求描述**：
实现数据导出功能，要求：

- 固定流程：连接数据源、获取数据、格式化、输出
- 支持导出为 CSV、JSON、XML 格式
- 子类只需实现格式化方法

```java
// 期望用法
DataExporter csvExporter = new CsvExporter();
csvExporter.export(); // 执行完整流程
```

---

### 练习 19: 迭代器模式 - 自定义集合

**需求描述**：
实现一个环形数组集合 `CircularArray`，要求：

- 支持从任意位置开始遍历
- 实现 `Iterable` 接口
- 使用 for-each 循环遍历

```java
// 期望用法
CircularArray<String> arr = new CircularArray<>(5);
arr.add("A"); arr.add("B"); arr.add("C");
arr.setStartIndex(1);
for (String s : arr) {
    System.out.println(s); // B, C, A
}
```

---

### 练习 20: 中介者模式 - 聊天室

**需求描述**：
实现一个聊天室系统，要求：

- 用户通过聊天室发送消息
- 消息可发送给所有人或特定用户
- 用户之间不直接通信

```java
// 期望用法
ChatRoom room = new ChatRoom();
User alice = new User("Alice", room);
User bob = new User("Bob", room);
alice.send("大家好！"); // 广播消息
alice.sendTo("Bob", "你好 Bob！"); // 私聊
```

---

### 练习 21: 备忘录模式 - 文本编辑器

**需求描述**：
实现支持撤销/重做的文本编辑器，要求：

- 保存编辑历史
- 支持多次撤销和重做
- 限制历史记录数量（如最多 10 条）

```java
// 期望用法
TextEditor editor = new TextEditor();
editor.write("Hello");
editor.write(" World");
editor.undo(); // 回到 "Hello"
editor.redo(); // 回到 "Hello World"
```

---

### 练习 22: 访问者模式 - 文件分析

**需求描述**：
分析文件系统中不同类型文件，要求：

- 支持 .txt、.java、.xml 文件
- 不同访问者执行不同操作（统计行数、代码检查等）
- 新增分析功能不需要修改文件类

```java
// 期望用法
FileVisitor lineCounter = new LineCountVisitor();
FileVisitor codeChecker = new CodeCheckVisitor();
for (FileElement file : files) {
    file.accept(lineCounter);
}
```

---

### 练习 23: 解释器模式 - 计算器

**需求描述**：
实现一个简单的数学表达式解释器，要求：

- 支持加减乘除运算
- 支持括号
- 解析字符串表达式并计算结果

```java
// 期望用法
Expression exp = new Parser().parse("(5 + 3) * 2");
System.out.println(exp.interpret()); // 16
```

---

## 综合练习

### 综合练习 1: 电商订单系统

结合以下模式设计电商订单系统：

- **单例模式**: 配置管理器
- **工厂模式**: 订单创建
- **策略模式**: 支付方式
- **状态模式**: 订单状态
- **观察者模式**: 订单状态变更通知

---

### 综合练习 2: 游戏角色系统

设计一个 RPG 游戏角色系统：

- **原型模式**: 克隆怪物
- **装饰器模式**: 装备增益
- **策略模式**: 攻击技能
- **状态模式**: 角色状态（正常、中毒、眩晕）
- **命令模式**: 技能释放和撤销

---

### 综合练习 3: 日志框架

模拟 Log4j 实现日志框架：

- **单例模式**: LoggerFactory
- **工厂模式**: 创建不同类型 Logger
- **装饰器模式**: 日志格式化
- **策略模式**: 日志输出策略
- **责任链模式**: 日志级别过滤

---

## 答案与参考

> 建议先独立完成练习，再参考各设计模式的详细文档中的代码示例。

每道练习都可以在对应模式的文档中找到类似的实现思路：

- [创建型模式文档](/docs/java-design-patterns/singleton-pattern)
- [结构型模式文档](/docs/java-design-patterns/adapter-pattern)
- [行为型模式文档](/docs/java-design-patterns/observer-pattern)

---

完成练习后，推荐阅读 [最佳实践](best-practices) 总结设计模式的应用经验！
